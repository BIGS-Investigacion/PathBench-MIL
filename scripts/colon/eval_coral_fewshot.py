#!/usr/bin/env python3
"""
eval_coral_fewshot.py

Curva de aprendizaje para CORAL condicional con few-shot labels:
  Para cada N (slides etiquetados por clase) y cada semilla,
  selecciona N slides por clase del dominio test, aplica CORAL condicional,
  evalúa balanced accuracy con LR entrenada en Macarena.

Resultado: media ± std de balanced accuracy en función de N.

Uso:
  python scripts/colon/eval_coral_fewshot.py [--n_pca 128] [--seeds 20] [--seed0 0]
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

warnings.filterwarnings('ignore')

ROOT     = Path(__file__).resolve().parents[2]
BAGS_DIR = Path('/home/PARADIS/colon/tfrecords/shared_bags/256_128_none_h_optimus_0')
ANNOT    = ROOT / 'config/annotations/annotations_colon_mss_msi_all.csv'
OUT_DIR  = ROOT / 'results/colon/coral_fewshot'

DATASET_LABELS = {'macarena': 'Macarena', 'cptac_coad': 'CPTAC-COAD', 'tcga_coad': 'TCGA-COAD'}
N_VALUES = [5, 10, 20, 30, 50]


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
    mu_s = X_src.mean(0);  mu_t = X_tgt.mean(0)
    d = X_src.shape[1]
    Cs = np.cov((X_src - mu_s).T) + alpha * np.eye(d)
    Ct = np.cov((X_tgt - mu_t).T) + alpha * np.eye(d)
    Us, Ss, _ = np.linalg.svd(Cs);  Ut, St, _ = np.linalg.svd(Ct)
    W = Us @ np.diag(1/np.sqrt(np.maximum(Ss, 1e-10))) @ Us.T @ \
        Ut @ np.diag(np.sqrt(np.maximum(St, 0))) @ Ut.T
    return (X_src - mu_s) @ W + mu_t


def apply_coral_cond_fewshot(X, meta, ds, labeled_idx, alpha):
    """
    CORAL condicional usando solo los slides en labeled_idx como referencia
    de clase para el dominio ds. Alinea todos los slides del dominio ds.
    """
    X_out = X.copy()
    d = X.shape[1]
    mask_ds = (meta['dataset'] == ds).values

    for cls_id in [0, 1]:
        # Slides etiquetados del dominio test con esta clase
        labeled_cls = [i for i in labeled_idx if meta.iloc[i]['category'] == cls_id]
        # Referencia: slides de Macarena con esta clase
        mask_mac_cls = ((meta['dataset'] == 'macarena') &
                        (meta['category'] == cls_id)).values

        n_s = len(labeled_cls)
        n_t = mask_mac_cls.sum()
        if n_s < 2 or n_t < 2:
            continue

        alpha_eff = alpha * max(1.0, d / min(n_s, n_t))

        # Ajusta CORAL sobre los labeled samples, aplica a todos los slides del dominio
        X_src_labeled = X[labeled_cls]
        X_tgt         = X[mask_mac_cls]

        mu_s = X_src_labeled.mean(0);  mu_t = X_tgt.mean(0)
        Cs = np.cov((X_src_labeled - mu_s).T) + alpha_eff * np.eye(d)
        Ct = np.cov((X_tgt - mu_t).T)         + alpha_eff * np.eye(d)
        Us, Ss, _ = np.linalg.svd(Cs);  Ut, St, _ = np.linalg.svd(Ct)
        W = Us @ np.diag(1/np.sqrt(np.maximum(Ss, 1e-10))) @ Us.T @ \
            Ut @ np.diag(np.sqrt(np.maximum(St, 0))) @ Ut.T

        # Aplica el transform a TODOS los slides de la clase (no solo los labeled)
        mask_ds_cls = mask_ds & (meta['category'] == cls_id).values
        X_out[mask_ds_cls] = (X[mask_ds_cls] - mu_s) @ W + mu_t

    return X_out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_pca',  type=int,   default=128)
    parser.add_argument('--alpha',  type=float, default=1.0)
    parser.add_argument('--seeds',  type=int,   default=20,
                        help='Número de semillas por cada N')
    parser.add_argument('--seed0',  type=int,   default=0,
                        help='Semilla base para PCA y LR')
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Cargar y proyectar
    print('Cargando features...')
    ann = pd.read_csv(ANNOT)
    X_raw, meta = load_features(ann)
    mask_mac = (meta['dataset'] == 'macarena').values

    scaler = StandardScaler().fit(X_raw[mask_mac])
    pca    = PCA(n_components=args.n_pca, random_state=args.seed0).fit(
                 scaler.transform(X_raw[mask_mac]))
    X_pca  = pca.transform(scaler.transform(X_raw))
    print(f'  Varianza explicada PCA: {pca.explained_variance_ratio_.sum():.1%}')

    # 2. LR sobre Macarena
    y_mac = meta.loc[mask_mac, 'category'].values
    lr = LogisticRegression(max_iter=2000, random_state=args.seed0, C=1.0)
    lr.fit(X_pca[mask_mac], y_mac)

    # Baselines (sin etiquetas del test)
    baselines = {}
    for ds in ['cptac_coad', 'tcga_coad']:
        mask  = (meta['dataset'] == ds).values
        y_t   = meta.loc[meta['dataset'] == ds, 'category'].values
        y_p   = lr.predict(X_pca[mask])
        baselines[ds] = balanced_accuracy_score(y_t, y_p)
    print(f'\n  Baseline sin alineación:  CPTAC={baselines["cptac_coad"]:.3f}  '
          f'TCGA={baselines["tcga_coad"]:.3f}')

    # Límite superior: CORAL condicional con todas las etiquetas
    from sklearn.metrics import balanced_accuracy_score as bas
    def coral_full(X, meta, alpha):
        X_out = X.copy(); d = X.shape[1]
        for ds in ['cptac_coad', 'tcga_coad']:
            for cls_id in [0, 1]:
                ms = (meta['dataset'] == ds).values & (meta['category'] == cls_id).values
                mt = (meta['dataset'] == 'macarena').values & (meta['category'] == cls_id).values
                ns, nt = ms.sum(), mt.sum()
                if ns < 2 or nt < 2: continue
                ae = alpha * max(1.0, d / min(ns, nt))
                X_out[ms] = coral(X[ms], X[mt], ae)
        return X_out

    X_oracle = coral_full(X_pca, meta, args.alpha)
    oracles = {}
    for ds in ['cptac_coad', 'tcga_coad']:
        mask = (meta['dataset'] == ds).values
        y_t  = meta.loc[meta['dataset'] == ds, 'category'].values
        y_p  = lr.predict(X_oracle[mask])
        oracles[ds] = bas(y_t, y_p)
    print(f'  Oracle (todas etiquetas): CPTAC={oracles["cptac_coad"]:.3f}  '
          f'TCGA={oracles["tcga_coad"]:.3f}')

    # 3. Curva few-shot
    print(f'\nCurva few-shot ({args.seeds} semillas × {len(N_VALUES)} valores de N)...')
    records = []

    for ds in ['cptac_coad', 'tcga_coad']:
        mask_ds = (meta['dataset'] == ds).values
        idx_ds  = np.where(mask_ds)[0]
        y_ds    = meta.loc[mask_ds, 'category'].values
        y_true  = meta.loc[meta['dataset'] == ds, 'category'].values

        idx_mss = idx_ds[y_ds == 0]
        idx_msi = idx_ds[y_ds == 1]

        for N in N_VALUES:
            if N > min(len(idx_mss), len(idx_msi)):
                print(f'  Skip N={N} para {DATASET_LABELS[ds]}: '
                      f'solo {len(idx_msi)} MSI disponibles')
                continue

            accs = []
            for seed in range(args.seed0, args.seed0 + args.seeds):
                rng = np.random.default_rng(seed)
                labeled = np.concatenate([
                    rng.choice(idx_mss, N, replace=False),
                    rng.choice(idx_msi, N, replace=False),
                ])
                X_aligned = apply_coral_cond_fewshot(X_pca, meta, ds, labeled, args.alpha)
                y_pred = lr.predict(X_aligned[mask_ds])
                accs.append(bas(y_true, y_pred))

            mu, sd = np.mean(accs), np.std(accs)
            print(f'  {DATASET_LABELS[ds]:12s}  N={N:2d}/clase  '
                  f'bal-acc={mu:.3f} ± {sd:.3f}')
            records.append({'dataset': DATASET_LABELS[ds], 'N': N,
                            'mean': mu, 'std': sd,
                            'baseline': baselines[ds], 'oracle': oracles[ds]})

    # Guardar
    df = pd.DataFrame(records)
    df.to_csv(OUT_DIR / f'fewshot_pca{args.n_pca}.csv', index=False)

    # Figura
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, ds_name in zip(axes, ['CPTAC-COAD', 'TCGA-COAD']):
        sub = df[df['dataset'] == ds_name]
        if sub.empty:
            continue
        ax.plot(sub['N'], sub['mean'], 'o-', color='#1565C0', lw=2, label='CORAL cond. few-shot')
        ax.fill_between(sub['N'],
                        sub['mean'] - sub['std'],
                        sub['mean'] + sub['std'],
                        alpha=0.2, color='#1565C0')
        ax.axhline(sub['baseline'].iloc[0], ls='--', color='#757575',
                   label=f'Sin alineación ({sub["baseline"].iloc[0]:.3f})')
        ax.axhline(sub['oracle'].iloc[0],   ls='--', color='#E53935',
                   label=f'Oracle — labels completas ({sub["oracle"].iloc[0]:.3f})')
        ax.set_xlabel('N slides etiquetados por clase', fontsize=12)
        ax.set_ylabel('Balanced accuracy', fontsize=12)
        ax.set_title(ds_name, fontsize=13)
        ax.set_xticks(N_VALUES)
        ax.legend(fontsize=9, framealpha=0.9)
        ax.set_ylim(0.4, 1.05)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'CORAL condicional few-shot ({args.seeds} semillas)', fontsize=13)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(OUT_DIR / f'fewshot_curve_pca{args.n_pca}.{ext}',
                    dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nListo. Resultados en: {OUT_DIR}')


if __name__ == '__main__':
    main()