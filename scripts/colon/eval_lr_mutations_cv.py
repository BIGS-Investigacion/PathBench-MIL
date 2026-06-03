#!/usr/bin/env python3
"""
eval_lr_mutations_cv.py

5-fold CV estratificado en Macarena para cada mutación (MLH1, MSH2, MSH6, PMS2)
usando slide mean + PCA + LR.

Uso:
  python scripts/colon/eval_lr_mutations_cv.py [--n_pca 128] [--n_folds 5]
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
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')

ROOT     = Path(__file__).resolve().parents[2]
BAGS_DIR = Path('/home/PARADIS/colon/tfrecords/shared_bags/256_128_none_h_optimus_0')

MUTATIONS = ['mlh1', 'msh2', 'msh6', 'pms2']


def load_slide_means(ann):
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


def cv_eval(X, y, n_folds, n_pca, seed):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    neg_list, pos_list, bal_list = [], [], []
    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        sc  = StandardScaler().fit(X_tr)
        pca = PCA(n_components=min(n_pca, X_tr.shape[0]-1), random_state=seed).fit(sc.transform(X_tr))
        X_tr_p = pca.transform(sc.transform(X_tr))
        X_te_p = pca.transform(sc.transform(X_te))
        lr = LogisticRegression(max_iter=2000, random_state=seed, C=1.0)
        lr.fit(X_tr_p, y_tr)
        yp = lr.predict(X_te_p)
        neg_list.append((yp[y_te==0] == 0).mean() if (y_te==0).sum() > 0 else np.nan)
        pos_list.append((yp[y_te==1] == 1).mean() if (y_te==1).sum() > 0 else np.nan)
        bal_list.append(balanced_accuracy_score(y_te, yp))
    return np.nanmean(neg_list), np.nanmean(pos_list), np.nanmean(bal_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_pca',   type=int, default=128)
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--seed',    type=int, default=42)
    args = parser.parse_args()

    print(f'\nPCA={args.n_pca}  |  folds={args.n_folds}\n')
    print(f'{"Mutación":<10} {"N(0)/N(1)":>12} {"wt acc":>8} {"mut acc":>8} {"bal":>8}')
    print('=' * 52)

    for mut in MUTATIONS:
        ann_path = ROOT / f'config/annotations/annotations_colon_{mut}_all.csv'
        ann = pd.read_csv(ann_path)
        ann_mac = ann[ann['dataset'] == 'macarena'].reset_index(drop=True)

        X, meta = load_slide_means(ann_mac)
        y = meta['category'].values
        n0, n1 = (y==0).sum(), (y==1).sum()

        neg_acc, pos_acc, bal = cv_eval(X, y, args.n_folds, args.n_pca, args.seed)
        print(f'  {mut.upper():<8} {str(n0)+"/"+str(n1):>12} '
              f'{neg_acc:>8.3f} {pos_acc:>8.3f} {bal:>8.3f}')

    print()


if __name__ == '__main__':
    main()