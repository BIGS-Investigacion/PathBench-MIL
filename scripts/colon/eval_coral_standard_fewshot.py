#!/usr/bin/env python3
"""
eval_lr_attn_smote.py

LR calibrada con N slides etiquetados del target + Attention-SMOTE para la clase minoritaria.

Attention-SMOTE: genera sintéticos como combinación ponderada por atención sobre k vecinos
más cercanos (softmax de distancias negativas), en lugar de interpolación lineal simple.

Uso:
  python scripts/colon/eval_lr_attn_smote.py [--seeds 20] [--k 5] [--n_synthetic 50]
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
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings('ignore')

ROOT     = Path(__file__).resolve().parents[2]
BAGS_DIR = Path('/home/PARADIS/colon/tfrecords/shared_bags/256_128_none_h_optimus_0')
ANNOT    = ROOT / 'config/annotations/annotations_colon_mss_msi_all.csv'
OUT_DIR  = ROOT / 'results/colon/lr_attn_smote'

DATASET_LABELS = {'cptac_coad': 'CPTAC-COAD', 'tcga_coad': 'TCGA-COAD'}
N_VALUES = [5, 10, 20, 30]


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


def attention_smote(X_pool, n_synthetic, k=5, temperature=1.0, rng=None):
    """
    Genera n_synthetic muestras sintéticas usando atención sobre k vecinos.

    X_pool: conjunto de referencia para k-NN (puede ser Macarena + calib del target)
    Para cada sintético:
      1. Elige un ancla aleatoria de X_pool
      2. Encuentra sus k vecinos más cercanos en X_pool
      3. Calcula pesos α_j = softmax(-d_j / τ)
      4. x_new = Σ α_j * x_j  + ruido pequeño
    """
    if rng is None:
        rng = np.random.default_rng(0)

    n = len(X_pool)
    k_eff = min(k, n - 1)

    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric='euclidean').fit(X_pool)

    synthetic = []
    for _ in range(n_synthetic):
        anchor_idx = rng.integers(0, n)
        anchor = X_pool[anchor_idx:anchor_idx+1]

        distances, indices = nn.kneighbors(anchor)
        # Excluir el propio ancla (índice 0)
        nbr_idx  = indices[0][1:]
        nbr_dist = distances[0][1:]

        # Pesos de atención: más peso a vecinos cercanos
        logits = -nbr_dist / (temperature * (nbr_dist.mean() + 1e-8))
        attn   = np.exp(logits - logits.max())
        attn  /= attn.sum()

        x_new = (attn[:, None] * X_pool[nbr_idx]).sum(axis=0)

        # Ruido pequeño proporcional a la distancia media
        noise_scale = nbr_dist.mean() * 0.05
        x_new += rng.normal(0, noise_scale, size=x_new.shape)

        synthetic.append(x_new)

    return np.stack(synthetic).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_pca',         type=int,   default=128)
    parser.add_argument('--seeds',         type=int,   default=20)
    parser.add_argument('--seed0',         type=int,   default=0)
    parser.add_argument('--min_per_class', type=int,   default=5)
    parser.add_argument('--k',             type=int,   default=5,
                        help='Vecinos para Attention-SMOTE')
    parser.add_argument('--n_synthetic',   type=int,   default=50,
                        help='Sintéticos por clase generados en cada semilla')
    parser.add_argument('--temperature',   type=float, default=1.0)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print('Cargando features...')
    ann = pd.read_csv(ANNOT)
    X_raw, meta = load_features(ann)
    mask_mac = (meta['dataset'] == 'macarena').values

    scaler = StandardScaler().fit(X_raw[mask_mac])
    pca    = PCA(n_components=args.n_pca, random_state=args.seed0).fit(
                 scaler.transform(X_raw[mask_mac]))
    X_pca  = pca.transform(scaler.transform(X_raw))
    print(f'  Varianza explicada PCA: {pca.explained_variance_ratio_.sum():.1%}')

    X_mac = X_pca[mask_mac]
    y_mac = meta.loc[mask_mac, 'category'].values

    # Baseline: LR solo Macarena
    lr_base = LogisticRegression(max_iter=2000, random_state=args.seed0, C=1.0)
    lr_base.fit(X_mac, y_mac)
    baselines = {}
    for ds in ['cptac_coad', 'tcga_coad']:
        mask = (meta['dataset'] == ds).values
        y    = meta.loc[mask, 'category'].values
        baselines[ds] = balanced_accuracy_score(y, lr_base.predict(X_pca[mask]))
    print(f'\n  Baseline (LR solo Macarena):  '
          f'CPTAC={baselines["cptac_coad"]:.3f}  TCGA={baselines["tcga_coad"]:.3f}')

    # Pool de Macarena por clase para Attention-SMOTE
    X_mac_mss = X_mac[y_mac == 0]
    X_mac_msi = X_mac[y_mac == 1]

    print(f'\nCurva LR + Attention-SMOTE ({args.seeds} semillas × {len(N_VALUES)} N)...')
    print(f'  k={args.k}, n_synthetic={args.n_synthetic}, τ={args.temperature}\n')

    records_lr, records_smote = [], []

    for ds in ['cptac_coad', 'tcga_coad']:
        mask_ds = (meta['dataset'] == ds).values
        idx_ds  = np.where(mask_ds)[0]
        y_ds    = meta.loc[mask_ds, 'category'].values
        idx_mss = idx_ds[y_ds == 0]
        idx_msi = idx_ds[y_ds == 1]

        print(f'  {DATASET_LABELS[ds]}  (MSS={len(idx_mss)}, MSI={len(idx_msi)})')

        for N in N_VALUES:
            n_mss = min(max(args.min_per_class, N), len(idx_mss) - 1)
            n_msi = min(max(args.min_per_class, N), len(idx_msi) - 1)

            rng_seeds = np.random.default_rng(args.seed0 + hash(ds) % 1000).integers(
                0, 100000, size=args.seeds)

            accs_lr, accs_smote = [], []
            mss_lr, msi_lr, mss_sm, msi_sm = [], [], [], []

            for seed in rng_seeds:
                rng = np.random.default_rng(seed)
                calib_idx = np.concatenate([
                    rng.choice(idx_mss, n_mss, replace=False),
                    rng.choice(idx_msi, n_msi, replace=False),
                ])
                test_idx = np.setdiff1d(idx_ds, calib_idx)
                y_test   = meta['category'].values[test_idx]

                X_calib = X_pca[calib_idx]
                y_calib = meta['category'].values[calib_idx]

                # ── LR calibrada (sin sintéticos) ─────────────────────────────
                X_tr = np.concatenate([X_mac, X_calib])
                y_tr = np.concatenate([y_mac, y_calib])
                lr = LogisticRegression(max_iter=2000, random_state=int(seed), C=1.0)
                lr.fit(X_tr, y_tr)
                yp = lr.predict(X_pca[test_idx])
                accs_lr.append(balanced_accuracy_score(y_test, yp))
                mss_lr.append((yp[y_test==0] == 0).mean())
                msi_lr.append((yp[y_test==1] == 1).mean())

                # ── LR + Attention-SMOTE ──────────────────────────────────────
                # Pool para sintéticos: Macarena + slides calibración del target
                X_calib_mss = X_calib[y_calib == 0]
                X_calib_msi = X_calib[y_calib == 1]

                pool_mss = np.concatenate([X_mac_mss, X_calib_mss])
                pool_msi = np.concatenate([X_mac_msi, X_calib_msi])

                syn_mss = attention_smote(pool_mss, args.n_synthetic,
                                          k=args.k, temperature=args.temperature, rng=rng)
                syn_msi = attention_smote(pool_msi, args.n_synthetic,
                                          k=args.k, temperature=args.temperature, rng=rng)

                X_tr_sm = np.concatenate([X_mac, X_calib, syn_mss, syn_msi])
                y_tr_sm = np.concatenate([y_mac, y_calib,
                                          np.zeros(len(syn_mss), dtype=int),
                                          np.ones(len(syn_msi),  dtype=int)])
                lr_sm = LogisticRegression(max_iter=2000, random_state=int(seed), C=1.0)
                lr_sm.fit(X_tr_sm, y_tr_sm)
                yp_sm = lr_sm.predict(X_pca[test_idx])
                accs_smote.append(balanced_accuracy_score(y_test, yp_sm))
                mss_sm.append((yp_sm[y_test==0] == 0).mean())
                msi_sm.append((yp_sm[y_test==1] == 1).mean())

            mu_lr,  sd_lr  = np.mean(accs_lr),   np.std(accs_lr)
            mu_sm,  sd_sm  = np.mean(accs_smote), np.std(accs_smote)
            print(f'    N={N:2d}/clase  '
                  f'LR={mu_lr:.3f}±{sd_lr:.3f}  '
                  f'LR+AttnSMOTE={mu_sm:.3f}±{sd_sm:.3f}  '
                  f'(MSS={np.mean(mss_sm):.3f}, MSI={np.mean(msi_sm):.3f})')

            records_lr.append({'dataset': DATASET_LABELS[ds], 'N': N,
                                'mean': mu_lr, 'std': sd_lr, 'baseline': baselines[ds]})
            records_smote.append({'dataset': DATASET_LABELS[ds], 'N': N,
                                  'mean': mu_sm, 'std': sd_sm, 'baseline': baselines[ds],
                                  'acc_mss': np.mean(mss_sm), 'acc_msi': np.mean(msi_sm)})
        print()

    pd.DataFrame(records_lr).to_csv(   OUT_DIR / f'lr_pca{args.n_pca}.csv',    index=False)
    pd.DataFrame(records_smote).to_csv(OUT_DIR / f'smote_pca{args.n_pca}.csv', index=False)

    # Figura comparativa
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, ds_name in zip(axes, ['CPTAC-COAD', 'TCGA-COAD']):
        sub_lr = pd.DataFrame(records_lr)[pd.DataFrame(records_lr)['dataset']==ds_name]
        sub_sm = pd.DataFrame(records_smote)[pd.DataFrame(records_smote)['dataset']==ds_name]

        ax.plot(sub_lr['N'], sub_lr['mean'], 'o--', color='#1565C0', lw=2,
                markersize=6, label='LR calibrada')
        ax.fill_between(sub_lr['N'],
                        sub_lr['mean']-sub_lr['std'], sub_lr['mean']+sub_lr['std'],
                        alpha=0.15, color='#1565C0')
        ax.plot(sub_sm['N'], sub_sm['mean'], 's-', color='#E53935', lw=2.5,
                markersize=7, label='LR + Attention-SMOTE')
        ax.fill_between(sub_sm['N'],
                        sub_sm['mean']-sub_sm['std'], sub_sm['mean']+sub_sm['std'],
                        alpha=0.15, color='#E53935')
        ax.axhline(sub_lr['baseline'].iloc[0], ls=':', color='#757575', lw=1.5,
                   label=f'Sin calibrar ({sub_lr["baseline"].iloc[0]:.3f})')
        ax.set_xlabel('N slides etiquetados por clase', fontsize=12)
        ax.set_ylabel('Balanced accuracy', fontsize=12)
        ax.set_title(ds_name, fontsize=13)
        ax.set_xticks(N_VALUES)
        ax.legend(fontsize=9)
        ax.set_ylim(0.4, 1.05)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'LR calibrada vs LR + Attention-SMOTE ({args.seeds} semillas)', fontsize=13)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(OUT_DIR / f'comparison_pca{args.n_pca}.{ext}', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Listo. Resultados en: {OUT_DIR}')


if __name__ == '__main__':
    main()