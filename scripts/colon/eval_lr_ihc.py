#!/usr/bin/env python3
"""
eval_lr_ihc.py

Evaluación del enfoque slide mean + PCA + LR sobre slides TCGA-COAD con IHC.

Para cada tarea (MSS/MSI, MLH1, MSH2, MSH6, PMS2):
  - LR entrenada en Macarena (anotaciones por tarea)
  - Evaluada en TCGA-COAD IHC slides

Uso:
  python scripts/colon/eval_lr_ihc.py [--n_pca 50]
"""

import warnings
import argparse
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
ANNOT_DIR = ROOT / 'config/annotations'

TASKS = {
    'MSS/MSI': {
        'train': ANNOT_DIR / 'annotations_colon_mss_msi_all.csv',
        'test':  ANNOT_DIR / 'annotations_tcga_coad_mss_msi_ihc.csv',
    },
    'MLH1': {
        'train': ANNOT_DIR / 'annotations_colon_mlh1_all.csv',
        'test':  ANNOT_DIR / 'annotations_tcga_coad_mlh1_ihc.csv',
    },
    'MSH2': {
        'train': ANNOT_DIR / 'annotations_colon_msh2_all.csv',
        'test':  ANNOT_DIR / 'annotations_tcga_coad_msh2_ihc.csv',
    },
    'MSH6': {
        'train': ANNOT_DIR / 'annotations_colon_msh6_all.csv',
        'test':  ANNOT_DIR / 'annotations_tcga_coad_msh6_ihc.csv',
    },
    'PMS2': {
        'train': ANNOT_DIR / 'annotations_colon_pms2_all.csv',
        'test':  ANNOT_DIR / 'annotations_tcga_coad_pms2_ihc.csv',
    },
}

_bag_cache = {}

def slide_mean(slide):
    if slide not in _bag_cache:
        p = BAGS_DIR / f'{slide}.pt'
        if not p.exists():
            _bag_cache[slide] = None
        else:
            bag = torch.load(p, map_location='cpu', weights_only=True).float().numpy()
            _bag_cache[slide] = bag.mean(axis=0).astype(np.float32)
    return _bag_cache[slide]


def load_split(ann_path, dataset_filter=None):
    ann = pd.read_csv(ann_path)
    if dataset_filter:
        ann = ann[ann['dataset'] == dataset_filter]
    vecs, labels = [], []
    for _, row in ann.iterrows():
        feat = slide_mean(row['slide'])
        if feat is None:
            continue
        vecs.append(feat)
        labels.append(int(row['category']))
    return np.stack(vecs).astype(np.float32), np.array(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_pca',  type=int, default=128)
    parser.add_argument('--n_cal',  type=int, default=5)
    parser.add_argument('--seeds',  type=int, default=10)
    parser.add_argument('--seed',   type=int, default=42)
    args = parser.parse_args()

    print(f'PCA={args.n_pca}  |  N_cal={args.n_cal}  |  LR warm-start Macarena → TCGA-COAD IHC\n')
    print(f'{"Tarea":<12} {"IHC N(0)/N(1)":>14}  {"Sin cal. Bal":>13}  {"Reajuste N="+str(args.n_cal)+" Bal":>18}')
    print('=' * 65)

    for task, paths in TASKS.items():
        X_mac, y_mac = load_split(paths['train'], dataset_filter='macarena')
        X_tst, y_tst = load_split(paths['test'])

        sc  = StandardScaler().fit(X_mac)
        pca = PCA(n_components=min(args.n_pca, X_mac.shape[0]-1, X_mac.shape[1]),
                  random_state=args.seed).fit(sc.transform(X_mac))
        X_mac_p = pca.transform(sc.transform(X_mac))
        X_tst_p = pca.transform(sc.transform(X_tst))

        n0_tst, n1_tst = (y_tst == 0).sum(), (y_tst == 1).sum()
        idx_0 = np.where(y_tst == 0)[0]
        idx_1 = np.where(y_tst == 1)[0]

        # Sin calibrar
        lr0 = LogisticRegression(max_iter=2000, random_state=args.seed, C=1.0)
        lr0.fit(X_mac_p, y_mac)
        yp = lr0.predict(X_tst_p)
        bal0 = balanced_accuracy_score(y_tst, yp)

        # Warm-start con N_cal por clase
        n0_c = min(args.n_cal, len(idx_0) - 1)
        n1_c = min(args.n_cal, len(idx_1) - 1)
        rng_master = np.random.default_rng(args.seed)
        seeds = rng_master.integers(0, 100000, size=args.seeds)
        bal_list = []
        for s in seeds:
            rng = np.random.default_rng(s)
            calib = np.concatenate([
                rng.choice(idx_0, n0_c, replace=False),
                rng.choice(idx_1, n1_c, replace=False),
            ])
            test_idx = np.setdiff1d(np.arange(len(y_tst)), calib)
            if len(test_idx) == 0:
                continue
            lr_ws = LogisticRegression(max_iter=2000, warm_start=True,
                                       random_state=int(s), C=1.0)
            lr_ws.fit(X_mac_p, y_mac)
            lr_ws.fit(X_tst_p[calib], y_tst[calib])
            bal_list.append(balanced_accuracy_score(y_tst[test_idx],
                                                    lr_ws.predict(X_tst_p[test_idx])))

        bm, bs = np.nanmean(bal_list), np.nanstd(bal_list)

        a0_0 = (lr0.predict(X_tst_p)[y_tst==0] == 0).mean() if n0_tst > 0 else np.nan
        a1_0 = (lr0.predict(X_tst_p)[y_tst==1] == 1).mean() if n1_tst > 0 else np.nan

        a0_list, a1_list = [], []
        for s in seeds:
            rng = np.random.default_rng(s)
            calib = np.concatenate([
                rng.choice(idx_0, n0_c, replace=False),
                rng.choice(idx_1, n1_c, replace=False),
            ])
            test_idx = np.setdiff1d(np.arange(len(y_tst)), calib)
            if len(test_idx) == 0:
                continue
            lr_ws2 = LogisticRegression(max_iter=2000, warm_start=True,
                                        random_state=int(s), C=1.0)
            lr_ws2.fit(X_mac_p, y_mac)
            lr_ws2.fit(X_tst_p[calib], y_tst[calib])
            yp2 = lr_ws2.predict(X_tst_p[test_idx])
            yt2 = y_tst[test_idx]
            a0_list.append((yp2[yt2==0]==0).mean() if (yt2==0).sum()>0 else np.nan)
            a1_list.append((yp2[yt2==1]==1).mean() if (yt2==1).sum()>0 else np.nan)

        a0m, a0s = np.nanmean(a0_list), np.nanstd(a0_list)
        a1m, a1s = np.nanmean(a1_list), np.nanstd(a1_list)

        print(f'  {task:<10} {str(n0_tst)+"/"+str(n1_tst):>10}')
        print(f'    Sin cal.:    acc(0)={a0_0:.3f}  acc(1)={a1_0:.3f}  bal={bal0:.3f}')
        print(f'    Reajuste N={args.n_cal}: acc(0)={a0m:.3f}±{a0s:.3f}  acc(1)={a1m:.3f}±{a1s:.3f}  bal={bm:.3f}±{bs:.3f}')

    print()


if __name__ == '__main__':
    main()