#!/usr/bin/env python3
"""
eval_coral_holdout.py

Evalúa el efecto de CORAL estándar en el escenario real de hold-out:
  - Train: Macarena (referencia CORAL + clasificador LR)
  - Test:  CPTAC-COAD y TCGA-COAD (alineadas a Macarena en inferencia)

Pipeline:
  1. Carga features mean-pooled por slide
  2. Fit PCA(128) + CORAL sobre Macarena (train)
  3. Aplica el transform a CPTAC y TCGA (test) — sin usar sus etiquetas
  4. Entrena LR sobre Macarena, evalúa sobre CPTAC y TCGA
  5. Compara accuracy por clase (MSS/MSI) con y sin CORAL
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import argparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report

warnings.filterwarnings('ignore')

ROOT     = Path(__file__).resolve().parents[2]
BAGS_DIR = Path('/home/PARADIS/colon/tfrecords/shared_bags/256_128_none_h_optimus_0')
ANNOT    = ROOT / 'config/annotations/annotations_colon_mss_msi_all.csv'
OUT_DIR  = ROOT / 'results/colon/coral_holdout'

DATASET_LABELS = {'macarena': 'Macarena', 'cptac_coad': 'CPTAC-COAD', 'tcga_coad': 'TCGA-COAD'}


# ── Carga ─────────────────────────────────────────────────────────────────────

def load_features(ann: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
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

def fit_coral(X_ref: np.ndarray, alpha: float):
    """Calcula las matrices de transform CORAL sobre el dominio de referencia."""
    mu  = X_ref.mean(axis=0)
    C   = np.cov((X_ref - mu).T) + alpha * np.eye(X_ref.shape[1])
    U, S, _ = np.linalg.svd(C)
    W_color = U @ np.diag(np.sqrt(np.maximum(S, 0.0))) @ U.T
    W_white = U @ np.diag(1.0 / np.sqrt(np.maximum(S, 1e-10))) @ U.T
    return mu, W_white, W_color


def apply_coral(X_src: np.ndarray, X_ref_mu, W_white_ref, W_color_ref, alpha: float):
    """Alinea X_src a la distribución de referencia."""
    mu_s = X_src.mean(axis=0)
    Xs   = X_src - mu_s
    C_s  = np.cov(Xs.T) + alpha * np.eye(X_src.shape[1])
    Us, Ss, _ = np.linalg.svd(C_s)
    W_white_s = Us @ np.diag(1.0 / np.sqrt(np.maximum(Ss, 1e-10))) @ Us.T
    return Xs @ W_white_s @ W_color_ref + X_ref_mu


# ── Evaluación ────────────────────────────────────────────────────────────────

def report(y_true, y_pred, dataset_name, tag):
    mss_mask = y_true == 0
    msi_mask = y_true == 1
    acc_mss = (y_pred[mss_mask] == 0).mean() if mss_mask.sum() > 0 else float('nan')
    acc_msi = (y_pred[msi_mask] == 1).mean() if msi_mask.sum() > 0 else float('nan')
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    print(f'  [{tag:20s}] {dataset_name:12s}  '
          f'MSS={acc_mss:.3f} (n={mss_mask.sum()})  '
          f'MSI={acc_msi:.3f} (n={msi_mask.sum()})  '
          f'bal-acc={bal_acc:.3f}')
    return {'dataset': dataset_name, 'tag': tag,
            'acc_mss': acc_mss, 'acc_msi': acc_msi, 'bal_acc': bal_acc,
            'n_mss': int(mss_mask.sum()), 'n_msi': int(msi_mask.sum())}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_pca', type=int, default=128)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--seed',  type=int, default=42)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Cargar features
    print('Cargando features...')
    ann = pd.read_csv(ANNOT)
    X_all, meta = load_features(ann)

    mask_mac  = meta['dataset'] == 'macarena'
    mask_cptac = meta['dataset'] == 'cptac_coad'
    mask_tcga  = meta['dataset'] == 'tcga_coad'

    X_mac   = X_all[mask_mac.values];   y_mac   = meta.loc[mask_mac,  'category'].values
    X_cptac = X_all[mask_cptac.values]; y_cptac = meta.loc[mask_cptac,'category'].values
    X_tcga  = X_all[mask_tcga.values];  y_tcga  = meta.loc[mask_tcga, 'category'].values

    print(f'  Macarena:   {len(X_mac)} slides  (MSS={( y_mac==0).sum()}, MSI={(y_mac==1).sum()})')
    print(f'  CPTAC-COAD: {len(X_cptac)} slides  (MSS={(y_cptac==0).sum()}, MSI={(y_cptac==1).sum()})')
    print(f'  TCGA-COAD:  {len(X_tcga)} slides  (MSS={(y_tcga==0).sum()}, MSI={(y_tcga==1).sum()})')

    # 2. PCA ajustado sobre Macarena
    print(f'\nPCA({args.n_pca}) ajustado sobre Macarena...')
    scaler = StandardScaler().fit(X_mac)
    pca    = PCA(n_components=args.n_pca, random_state=args.seed).fit(scaler.transform(X_mac))
    print(f'  Varianza explicada: {pca.explained_variance_ratio_.sum():.1%}')

    X_mac_pca   = pca.transform(scaler.transform(X_mac))
    X_cptac_pca = pca.transform(scaler.transform(X_cptac))
    X_tcga_pca  = pca.transform(scaler.transform(X_tcga))

    # 3. CORAL ajustado sobre Macarena PCA
    print(f'CORAL ajustado sobre Macarena (alpha={args.alpha})...')
    mu_mac, W_white_mac, W_color_mac = fit_coral(X_mac_pca, args.alpha)

    # Macarena no se transforma (es la referencia)
    X_cptac_coral = apply_coral(X_cptac_pca, mu_mac, W_white_mac, W_color_mac, args.alpha)
    X_tcga_coral  = apply_coral(X_tcga_pca,  mu_mac, W_white_mac, W_color_mac, args.alpha)

    # 4. Clasificador LR entrenado sobre Macarena
    print('\nEntrenando LR sobre Macarena...')
    lr = LogisticRegression(max_iter=2000, random_state=args.seed, C=1.0)
    lr.fit(X_mac_pca, y_mac)

    # 5. Resultados
    print('\nResultados (train=Macarena):')
    print(f'  {"":20s} {"dataset":12s}  MSS acc         MSI acc         bal-acc')
    print(f'  {"-"*75}')

    rows = []
    # Sin CORAL
    for X_test, y_test, name in [
            (X_mac_pca,   y_mac,   'Macarena'),
            (X_cptac_pca, y_cptac, 'CPTAC-COAD'),
            (X_tcga_pca,  y_tcga,  'TCGA-COAD')]:
        pred = lr.predict(X_test)
        rows.append(report(y_test, pred, name, 'Sin CORAL'))

    print()
    # Con CORAL
    for X_test, y_test, name in [
            (X_mac_pca,    y_mac,   'Macarena'),
            (X_cptac_coral, y_cptac, 'CPTAC-COAD'),
            (X_tcga_coral,  y_tcga,  'TCGA-COAD')]:
        pred = lr.predict(X_test)
        rows.append(report(y_test, pred, name, 'Con CORAL estándar'))

    # Guardar
    results = pd.DataFrame(rows)
    results.to_csv(OUT_DIR / f'results_pca{args.n_pca}.csv', index=False)
    print(f'\nResultados guardados en: {OUT_DIR}')


if __name__ == '__main__':
    main()