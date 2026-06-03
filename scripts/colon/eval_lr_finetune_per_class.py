#!/usr/bin/env python3
"""
eval_lr_finetune_per_class.py

LR calibrada con N slides etiquetados del target.
Muestra accuracy por clase (MSS, MSI) + balanced accuracy.

Uso:
  python scripts/colon/eval_lr_finetune_per_class.py [--seeds 20] [--min_per_class 5]
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_pca',         type=int, default=128)
    parser.add_argument('--seeds',         type=int, default=20)
    parser.add_argument('--seed0',         type=int, default=0)
    parser.add_argument('--min_per_class', type=int, default=5)
    args = parser.parse_args()

    ann = pd.read_csv(ANNOT)
    X_raw, meta = load_features(ann)
    mask_mac = (meta['dataset'] == 'macarena').values

    scaler = StandardScaler().fit(X_raw[mask_mac])
    pca    = PCA(n_components=args.n_pca, random_state=args.seed0).fit(
                 scaler.transform(X_raw[mask_mac]))
    X_pca  = pca.transform(scaler.transform(X_raw))
    X_mac  = X_pca[mask_mac]
    y_mac  = meta.loc[mask_mac, 'category'].values

    print(f'\n{"Método":<40} {"MSS":>10} {"MSI":>10} {"bal":>10}')
    print('=' * 74)

    for ds in ['cptac_coad', 'tcga_coad']:
        ds_name  = {'cptac_coad': 'CPTAC-COAD', 'tcga_coad': 'TCGA-COAD'}[ds]
        mask_ds  = (meta['dataset'] == ds).values
        idx_ds   = np.where(mask_ds)[0]
        y_ds     = meta.loc[mask_ds, 'category'].values
        idx_mss  = idx_ds[y_ds == 0]
        idx_msi  = idx_ds[y_ds == 1]

        print(f'\n  {ds_name}  (MSS={len(idx_mss)}, MSI={len(idx_msi)})')
        print('-' * 74)

        # Baseline sin calibrar
        lr0 = LogisticRegression(max_iter=2000, random_state=args.seed0, C=1.0)
        lr0.fit(X_mac, y_mac)
        yp = lr0.predict(X_pca[mask_ds])
        print(f'  {"Sin calibrar":<38} '
              f'{(yp[y_ds==0]==0).mean():>10.3f} '
              f'{(yp[y_ds==1]==1).mean():>10.3f} '
              f'{balanced_accuracy_score(y_ds, yp):>10.3f}')

        # LR calibrada con N slides del target
        for N in N_VALUES:
            n_mss = min(max(args.min_per_class, N), len(idx_mss) - 1)
            n_msi = min(max(args.min_per_class, N), len(idx_msi) - 1)

            rng_seeds = np.random.default_rng(args.seed0 + hash(ds) % 1000).integers(
                0, 100000, size=args.seeds)
            mss_list, msi_list, bal_list = [], [], []

            for seed in rng_seeds:
                rng = np.random.default_rng(seed)
                calib_idx = np.concatenate([
                    rng.choice(idx_mss, n_mss, replace=False),
                    rng.choice(idx_msi, n_msi, replace=False),
                ])
                test_idx = np.setdiff1d(idx_ds, calib_idx)
                X_tr = np.concatenate([X_mac, X_pca[calib_idx]])
                y_tr = np.concatenate([y_mac, meta['category'].values[calib_idx]])
                lr = LogisticRegression(max_iter=2000, random_state=int(seed), C=1.0)
                lr.fit(X_tr, y_tr)
                yp = lr.predict(X_pca[test_idx])
                yt = meta['category'].values[test_idx]
                mss_list.append((yp[yt == 0] == 0).mean())
                msi_list.append((yp[yt == 1] == 1).mean())
                bal_list.append(balanced_accuracy_score(yt, yp))

            print(f'  {"LR calibrada N="+str(N)+"/clase":<38} '
                  f'{np.mean(mss_list):>7.3f}±{np.std(mss_list):.2f} '
                  f'{np.mean(msi_list):>7.3f}±{np.std(msi_list):.2f} '
                  f'{np.mean(bal_list):>7.3f}±{np.std(bal_list):.2f}')

    print()


if __name__ == '__main__':
    main()