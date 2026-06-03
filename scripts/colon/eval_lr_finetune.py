#!/usr/bin/env python3
"""
eval_lr_finetune.py

LR calibrada con N slides etiquetados del dominio target.
Variante Bridge-SMOTE: genera sintéticos que conectan los clusters de
Macarena y el target interpolando entre ambos dominios clase a clase.

Uso:
  python scripts/colon/eval_lr_finetune.py [--seeds 20] [--min_per_class 5]
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
OUT_DIR  = ROOT / 'results/colon/lr_finetune'

DATASET_LABELS = {'cptac_coad': 'CPTAC-COAD', 'tcga_coad': 'TCGA-COAD'}
N_VALUES    = [5, 10, 20, 30]
N_SYNTHETIC = 100   # sintéticos por clase para el puente


def bridge_smote(X_source, X_target, n_synthetic, k=5, temperature=1.0, rng=None):
    """
    Genera n_synthetic muestras que conectan X_source (Macarena) con X_target (calib target).

    Para cada sintético:
      1. Ancla aleatoria de X_target (extremo target)
      2. k vecinos más cercanos en X_source (Macarena)
      3. α_j = softmax(-d_j / τ)
      4. x_new = λ * ancla_target + (1-λ) * Σ α_j * x_j_source,  λ ~ U(0,1)
         → λ=0: cerca de Macarena   λ=1: cerca del target
    """
    if rng is None:
        rng = np.random.default_rng(0)

    k_eff = min(k, len(X_source) - 1)
    nn = NearestNeighbors(n_neighbors=k_eff, metric='euclidean').fit(X_source)

    synthetic = []
    for _ in range(n_synthetic):
        anchor_idx = rng.integers(0, len(X_target))
        anchor     = X_target[anchor_idx:anchor_idx + 1]

        distances, indices = nn.kneighbors(anchor)
        nbr_idx  = indices[0]
        nbr_dist = distances[0]

        logits = -nbr_dist / (temperature * (nbr_dist.mean() + 1e-8))
        attn   = np.exp(logits - logits.max())
        attn  /= attn.sum()
        mac_point = (attn[:, None] * X_source[nbr_idx]).sum(axis=0)

        lam   = rng.random()
        x_new = lam * anchor[0] + (1 - lam) * mac_point
        x_new += rng.normal(0, nbr_dist.mean() * 0.03, size=x_new.shape)
        synthetic.append(x_new)

    return np.stack(synthetic).astype(np.float32)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_pca',         type=int,   default=128)
    parser.add_argument('--seeds',         type=int,   default=20)
    parser.add_argument('--seed0',         type=int,   default=0)
    parser.add_argument('--min_per_class', type=int,   default=5)
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

    print(f'\nCurva ({args.seeds} semillas × {len(N_VALUES)} N)...\n')
    records = []

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

            accs_lr, accs_br = [], []
            rng_seeds = np.random.default_rng(args.seed0 + hash(ds) % 1000).integers(
                0, 100000, size=args.seeds)

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

                # ── LR calibrada ──────────────────────────────────────────────
                X_tr = np.concatenate([X_mac, X_calib])
                y_tr = np.concatenate([y_mac, y_calib])
                lr = LogisticRegression(max_iter=2000, random_state=int(seed), C=1.0)
                lr.fit(X_tr, y_tr)
                accs_lr.append(balanced_accuracy_score(y_test, lr.predict(X_pca[test_idx])))

                # ── LR + Bridge-SMOTE ─────────────────────────────────────────
                syn_mss = bridge_smote(X_mac[y_mac == 0], X_calib[y_calib == 0],
                                       N_SYNTHETIC, rng=rng)
                syn_msi = bridge_smote(X_mac[y_mac == 1], X_calib[y_calib == 1],
                                       N_SYNTHETIC, rng=rng)

                X_tr_br = np.concatenate([X_mac, X_calib, syn_mss, syn_msi])
                y_tr_br = np.concatenate([y_mac, y_calib,
                                          np.zeros(len(syn_mss), dtype=int),
                                          np.ones(len(syn_msi),  dtype=int)])
                lr_br = LogisticRegression(max_iter=2000, random_state=int(seed), C=1.0)
                lr_br.fit(X_tr_br, y_tr_br)
                accs_br.append(balanced_accuracy_score(y_test, lr_br.predict(X_pca[test_idx])))

            mu_lr, sd_lr = np.mean(accs_lr), np.std(accs_lr)
            mu_br, sd_br = np.mean(accs_br), np.std(accs_br)
            print(f'    N={N:2d}/clase  LR={mu_lr:.3f}±{sd_lr:.3f}  '
                  f'LR+Bridge={mu_br:.3f}±{sd_br:.3f}')
            records.append({
                'dataset': DATASET_LABELS[ds], 'N': N,
                'mean_lr': mu_lr, 'std_lr': sd_lr,
                'mean_br': mu_br, 'std_br': sd_br,
                'baseline': baselines[ds],
            })
        print()

    df = pd.DataFrame(records)
    df.to_csv(OUT_DIR / f'results_pca{args.n_pca}.csv', index=False)

    # Figura
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, ds_name in zip(axes, ['CPTAC-COAD', 'TCGA-COAD']):
        sub = df[df['dataset'] == ds_name].sort_values('N')
        if sub.empty:
            continue
        ax.plot(sub['N'], sub['mean_lr'], 'o--', color='#1565C0', lw=2,
                markersize=6, label='LR calibrada')
        ax.fill_between(sub['N'],
                        sub['mean_lr'] - sub['std_lr'],
                        sub['mean_lr'] + sub['std_lr'],
                        alpha=0.15, color='#1565C0')
        ax.plot(sub['N'], sub['mean_br'], 's-', color='#E53935', lw=2.5,
                markersize=7, label='LR + Bridge-SMOTE')
        ax.fill_between(sub['N'],
                        sub['mean_br'] - sub['std_br'],
                        sub['mean_br'] + sub['std_br'],
                        alpha=0.15, color='#E53935')
        ax.axhline(sub['baseline'].iloc[0], ls=':', color='#757575', lw=1.5,
                   label=f'Sin calibrar ({sub["baseline"].iloc[0]:.3f})')
        ax.set_xlabel('N slides etiquetados por clase', fontsize=12)
        ax.set_ylabel('Balanced accuracy', fontsize=12)
        ax.set_title(ds_name, fontsize=13)
        ax.set_xticks(N_VALUES)
        ax.legend(fontsize=9)
        ax.set_ylim(0.4, 1.05)
        ax.grid(True, alpha=0.3)
        for _, r in sub.iterrows():
            ax.annotate(f'{r["mean_br"]:.3f}', (r['N'], r['mean_br']),
                        textcoords='offset points', xytext=(0, 8),
                        ha='center', fontsize=8)

    fig.suptitle(f'LR calibrada vs LR + Bridge-SMOTE ({args.seeds} semillas)', fontsize=13)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(OUT_DIR / f'curve_pca{args.n_pca}.{ext}', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Listo. Resultados en: {OUT_DIR}')


if __name__ == '__main__':
    main()