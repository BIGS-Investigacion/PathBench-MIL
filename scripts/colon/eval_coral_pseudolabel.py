#!/usr/bin/env python3
"""
eval_coral_pseudolabel.py

CORAL condicional con pseudo-etiquetas: simula el escenario real donde
no se conocen las etiquetas de CPTAC/TCGA en el momento de la inferencia.

Pipeline:
  1. Ajusta PCA(128) + LR sobre Macarena (slide means)
  2. Predice pseudo-etiquetas MSS/MSI para CPTAC y TCGA
  3. Aplica CORAL condicional usando esas pseudo-etiquetas
  4. Evalúa con la misma LR
  5. Opcionalmente itera (mejores pseudo-etiquetas → mejor CORAL → ...)

Compara cuatro variantes:
  A) Sin alineación
  B) CORAL estándar
  C) CORAL condicional — etiquetas verdaderas (límite superior)
  D) CORAL condicional — pseudo-etiquetas LR  (escenario real)
  E) CORAL condicional — pseudo-etiquetas iterado N veces

Uso:
  python scripts/colon/eval_coral_pseudolabel.py [--n_pca 128] [--iters 3] [--seed 42]
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

warnings.filterwarnings('ignore')

ROOT     = Path(__file__).resolve().parents[2]
BAGS_DIR = Path('/home/PARADIS/colon/tfrecords/shared_bags/256_128_none_h_optimus_0')
ANNOT    = ROOT / 'config/annotations/annotations_colon_mss_msi_all.csv'
OUT_DIR  = ROOT / 'results/colon/coral_pseudolabel'

DATASET_LABELS = {'macarena': 'Macarena', 'cptac_coad': 'CPTAC-COAD', 'tcga_coad': 'TCGA-COAD'}


# ── Carga ─────────────────────────────────────────────────────────────────────

def load_features(ann):
    vecs, rows = [], []
    for _, row in ann.iterrows():
        p = BAGS_DIR / f'{row["slide"]}.pt'
        if not p.exists():
            continue
        feat = torch.load(p, map_location='cpu', weights_only=True).float().numpy()
        vecs.append(feat.mean(axis=0))
        rows.append({'slide': row['slide'], 'dataset': row['dataset'],
                     'category': int(row['category'])})
    return np.stack(vecs).astype(np.float32), pd.DataFrame(rows)


# ── CORAL ─────────────────────────────────────────────────────────────────────

def coral(X_src, X_tgt, alpha):
    mu_s = X_src.mean(axis=0);  mu_t = X_tgt.mean(axis=0)
    d = X_src.shape[1]
    Cs = np.cov((X_src - mu_s).T) + alpha * np.eye(d)
    Ct = np.cov((X_tgt - mu_t).T) + alpha * np.eye(d)
    Us, Ss, _ = np.linalg.svd(Cs)
    Ut, St, _ = np.linalg.svd(Ct)
    W = Us @ np.diag(1/np.sqrt(np.maximum(Ss, 1e-10))) @ Us.T @ \
        Ut @ np.diag(np.sqrt(np.maximum(St, 0))) @ Ut.T
    return (X_src - mu_s) @ W + mu_t


def apply_coral_std(X, meta, alpha):
    X_out = X.copy()
    X_mac = X[(meta['dataset'] == 'macarena').values]
    for ds in ['cptac_coad', 'tcga_coad']:
        mask = (meta['dataset'] == ds).values
        X_out[mask] = coral(X[mask], X_mac, alpha)
    return X_out


def apply_coral_cond(X, meta, labels, alpha):
    """
    CORAL condicional usando `labels` como pseudo-etiquetas.
    labels: array de longitud len(meta) con 0/1 para todos los slides.
    Macarena usa sus etiquetas verdaderas; CPTAC/TCGA usan las pseudo-etiquetas.
    """
    X_out = X.copy()
    d = X.shape[1]

    for ds in ['cptac_coad', 'tcga_coad']:
        for cls_id in [0, 1]:
            mask_src = ((meta['dataset'] == ds).values) & (labels == cls_id)
            mask_tgt = ((meta['dataset'] == 'macarena').values) & \
                       (meta['category'].values == cls_id)

            n_src, n_tgt = mask_src.sum(), mask_tgt.sum()
            if n_src < 5 or n_tgt < 5:
                continue

            alpha_eff = alpha * max(1.0, d / min(n_src, n_tgt))
            X_out[mask_src] = coral(X[mask_src], X[mask_tgt], alpha_eff)

    return X_out


# ── Métricas ──────────────────────────────────────────────────────────────────

def evaluate(lr, X, meta, tag, rows):
    for ds in ['macarena', 'cptac_coad', 'tcga_coad']:
        mask  = (meta['dataset'] == ds).values
        y_true = meta.loc[meta['dataset'] == ds, 'category'].values
        y_pred = lr.predict(X[mask])

        mss = y_true == 0;  msi = y_true == 1
        acc_mss = (y_pred[mss] == 0).mean() if mss.sum() > 0 else float('nan')
        acc_msi = (y_pred[msi] == 1).mean() if msi.sum() > 0 else float('nan')
        bal     = balanced_accuracy_score(y_true, y_pred)

        print(f'  [{tag:35s}] {DATASET_LABELS[ds]:12s}  '
              f'MSS={acc_mss:.3f} (n={mss.sum()})  '
              f'MSI={acc_msi:.3f} (n={msi.sum()})  bal={bal:.3f}')
        rows.append({'tag': tag, 'dataset': DATASET_LABELS[ds],
                     'acc_mss': acc_mss, 'acc_msi': acc_msi, 'bal_acc': bal})


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_pca', type=int,   default=128)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--iters', type=int,   default=3,
                        help='Iteraciones de pseudo-etiqueta → CORAL condicional')
    parser.add_argument('--seed',  type=int,   default=42)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Cargar features
    print('Cargando features...')
    ann = pd.read_csv(ANNOT)
    X_raw, meta = load_features(ann)

    mask_mac = (meta['dataset'] == 'macarena').values
    for ds in DATASET_LABELS:
        m = meta['dataset'] == ds
        print(f'  {DATASET_LABELS[ds]:12s}: {m.sum()} slides  '
              f'(MSS={(meta.loc[m,"category"]==0).sum()}, '
              f'MSI={(meta.loc[m,"category"]==1).sum()})')

    # 2. PCA ajustado sobre Macarena
    print(f'\nPCA({args.n_pca}) + StandardScaler ajustados sobre Macarena...')
    scaler = StandardScaler().fit(X_raw[mask_mac])
    pca    = PCA(n_components=args.n_pca, random_state=args.seed).fit(
                 scaler.transform(X_raw[mask_mac]))
    X_pca  = pca.transform(scaler.transform(X_raw))
    print(f'  Varianza explicada: {pca.explained_variance_ratio_.sum():.1%}')

    # 3. LR ajustada sobre Macarena
    print('LR ajustada sobre Macarena...')
    lr = LogisticRegression(max_iter=2000, random_state=args.seed, C=1.0)
    lr.fit(X_pca[mask_mac], meta.loc[mask_mac, 'category'].values)

    rows = []
    print('\nResultados:')
    print(f'  {"variante":35s} {"dataset":12s}  MSS             MSI             bal-acc')
    print(f'  {"-"*85}')

    # A) Sin alineación
    evaluate(lr, X_pca, meta, 'A) Sin alineación', rows)
    print()

    # B) CORAL estándar
    X_std = apply_coral_std(X_pca, meta, args.alpha)
    evaluate(lr, X_std, meta, 'B) CORAL estándar', rows)
    print()

    # C) CORAL condicional — etiquetas verdaderas (límite superior)
    true_labels = meta['category'].values
    X_cond_true = apply_coral_cond(X_pca, meta, true_labels, args.alpha)
    evaluate(lr, X_cond_true, meta, 'C) CORAL cond. — labels verdaderas', rows)
    print()

    # D) CORAL condicional — pseudo-etiquetas (1ª iteración)
    pseudo = lr.predict(X_pca).copy()
    # Macarena mantiene sus etiquetas verdaderas
    pseudo[mask_mac] = meta.loc[mask_mac, 'category'].values
    X_cond_ps = apply_coral_cond(X_pca, meta, pseudo, args.alpha)

    n_correct = (pseudo[~mask_mac] == meta.loc[~mask_mac, 'category'].values).mean()
    print(f'  Pseudo-etiquetas iter 1: accuracy={n_correct:.3f} '
          f'({int(n_correct*(~mask_mac).sum())}/{(~mask_mac).sum()} correctas)')
    evaluate(lr, X_cond_ps, meta, 'D) CORAL cond. — pseudo-labels iter 1', rows)
    print()

    # E) Iteraciones adicionales
    X_iter = X_cond_ps.copy()
    for it in range(2, args.iters + 1):
        pseudo = lr.predict(X_iter).copy()
        pseudo[mask_mac] = meta.loc[mask_mac, 'category'].values
        X_iter = apply_coral_cond(X_pca, meta, pseudo, args.alpha)

        n_correct = (pseudo[~mask_mac] == meta.loc[~mask_mac, 'category'].values).mean()
        print(f'  Pseudo-etiquetas iter {it}: accuracy={n_correct:.3f} '
              f'({int(n_correct*(~mask_mac).sum())}/{(~mask_mac).sum()} correctas)')
        evaluate(lr, X_iter, meta, f'E) CORAL cond. — pseudo-labels iter {it}', rows)
        print()

    # Guardar
    pd.DataFrame(rows).to_csv(OUT_DIR / f'results_pca{args.n_pca}.csv', index=False)
    print(f'Resultados en: {OUT_DIR}')


if __name__ == '__main__':
    main()