#!/usr/bin/env python3
"""
eval_final.py

Evaluación final con slide mean + PCA(n_pca) + LR:

  1. 10-fold CV estratificado en Macarena (MSS/MSI + MLH1/MSH2/MSH6/PMS2)
  2. Cross-domain CPTAC y TCGA:
       - Sin calibrar
       - Calibrado con N=5, 10, 20 slides etiquetados del target (10 semillas)

Uso:
  python scripts/colon/eval_final.py [--n_pca 128] [--n_folds 10] [--seeds 10]
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
OUT_DIR  = ROOT / 'results/colon/eval_final'

TASKS = {
    'MSS/MSI': ROOT / 'config/annotations/annotations_colon_mss_msi_all.csv',
    'MLH1':    ROOT / 'config/annotations/annotations_colon_mlh1_all.csv',
    'MSH2':    ROOT / 'config/annotations/annotations_colon_msh2_all.csv',
    'MSH6':    ROOT / 'config/annotations/annotations_colon_msh6_all.csv',
    'PMS2':    ROOT / 'config/annotations/annotations_colon_pms2_all.csv',
}

N_CALIB = [5, 10, 20]


# ── Features ──────────────────────────────────────────────────────────────────

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


def load_features(ann):
    vecs, rows = [], []
    for _, row in ann.iterrows():
        feat = slide_mean(row['slide'])
        if feat is None:
            continue
        vecs.append(feat)
        rows.append({'slide': row['slide'], 'dataset': row['dataset'],
                     'category': int(row['category'])})
    return np.stack(vecs).astype(np.float32), pd.DataFrame(rows)


# ── Evaluación ────────────────────────────────────────────────────────────────

def fit_pca(X_mac, n_pca, seed):
    sc  = StandardScaler().fit(X_mac)
    pca = PCA(n_components=min(n_pca, X_mac.shape[0]-1, X_mac.shape[1]),
              random_state=seed).fit(sc.transform(X_mac))
    return sc, pca


def cv_eval(X, y, n_folds, n_pca, seed):
    """Stratified K-fold CV. Devuelve (acc_0, acc_1, bal)."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    a0, a1, ab = [], [], []
    for tr, te in skf.split(X, y):
        sc, pca = fit_pca(X[tr], n_pca, seed)
        X_tr = pca.transform(sc.transform(X[tr]))
        X_te = pca.transform(sc.transform(X[te]))
        lr = LogisticRegression(max_iter=2000, random_state=seed, C=1.0)
        lr.fit(X_tr, y[tr])
        yp = lr.predict(X_te)
        yt = y[te]
        a0.append((yp[yt==0]==0).mean() if (yt==0).sum() > 0 else np.nan)
        a1.append((yp[yt==1]==1).mean() if (yt==1).sum() > 0 else np.nan)
        ab.append(balanced_accuracy_score(yt, yp))
    return np.nanmean(a0), np.nanmean(a1), np.nanmean(ab)


def cross_domain_eval(X_mac, y_mac, X_tgt, y_tgt, n_pca, seed,
                      calib_idx=None):
    """LR entrenada en Macarena (+ calibración opcional), evaluada en test."""
    if calib_idx is not None and len(calib_idx) > 0:
        X_tr = np.concatenate([X_mac, X_tgt[calib_idx]])
        y_tr = np.concatenate([y_mac, y_tgt[calib_idx]])
        test_idx = np.setdiff1d(np.arange(len(y_tgt)), calib_idx)
    else:
        X_tr, y_tr = X_mac, y_mac
        test_idx = np.arange(len(y_tgt))

    if len(test_idx) == 0:
        return np.nan, np.nan, np.nan

    sc, pca = fit_pca(X_mac, n_pca, seed)
    X_tr_p  = pca.transform(sc.transform(X_tr))
    X_te_p  = pca.transform(sc.transform(X_tgt[test_idx]))
    y_te    = y_tgt[test_idx]

    lr = LogisticRegression(max_iter=2000, random_state=seed, C=1.0)
    lr.fit(X_tr_p, y_tr)
    yp = lr.predict(X_te_p)

    a0 = (yp[y_te==0]==0).mean() if (y_te==0).sum() > 0 else np.nan
    a1 = (yp[y_te==1]==1).mean() if (y_te==1).sum() > 0 else np.nan
    return a0, a1, balanced_accuracy_score(y_te, yp)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_task(task_name, ann_path, args):
    ann = pd.read_csv(ann_path)
    X_raw, meta = load_features(ann)

    mask_mac = (meta['dataset'] == 'macarena').values
    X_mac = X_raw[mask_mac]
    y_mac = meta.loc[mask_mac, 'category'].values
    n0_mac, n1_mac = (y_mac==0).sum(), (y_mac==1).sum()

    records = []

    # ── 1. CV en Macarena ───────────────────────────────────────────────────
    print(f'\n{"─"*70}')
    print(f'  {task_name}  —  Macarena: {n0_mac} wt / {n1_mac} mut')
    print(f'{"─"*70}')
    print(f'  {"Evaluación":<35} {"acc(0)":>8} {"acc(1)":>8} {"bal":>8}')
    print(f'  {"─"*59}')

    a0, a1, bal = cv_eval(X_mac, y_mac, args.n_folds, args.n_pca, args.seed)
    label = f'Macarena {args.n_folds}-fold CV'
    print(f'  {label:<35} {a0:>8.3f} {a1:>8.3f} {bal:>8.3f}')
    records.append({'task': task_name, 'eval': label, 'dataset': 'macarena',
                    'N': None, 'acc0': a0, 'acc0_std': np.nan,
                    'acc1': a1, 'acc1_std': np.nan,
                    'bal': bal, 'bal_std': np.nan})

    # ── 2. Cross-domain ─────────────────────────────────────────────────────
    for ds in ['cptac_coad', 'tcga_coad']:
        mask_ds = (meta['dataset'] == ds).values
        if mask_ds.sum() == 0:
            continue
        X_tgt = X_raw[mask_ds]
        y_tgt = meta.loc[mask_ds, 'category'].values
        idx_0 = np.where(y_tgt == 0)[0]
        idx_1 = np.where(y_tgt == 1)[0]
        ds_label = {'cptac_coad': 'CPTAC-COAD', 'tcga_coad': 'TCGA-COAD'}[ds]

        print(f'\n  {ds_label}  ({len(idx_0)} wt / {len(idx_1)} mut)')

        # Sin calibrar
        a0, a1, bal = cross_domain_eval(X_mac, y_mac, X_tgt, y_tgt,
                                        args.n_pca, args.seed)
        print(f'  {"Sin calibrar":<35} {a0:>8.3f} {a1:>8.3f} {bal:>8.3f}')
        records.append({'task': task_name, 'eval': 'Sin calibrar',
                        'dataset': ds_label, 'N': 0,
                        'acc0': a0, 'acc0_std': np.nan,
                        'acc1': a1, 'acc1_std': np.nan,
                        'bal': bal, 'bal_std': np.nan})

        # Con calibración N por clase
        rng_master = np.random.default_rng(args.seed + hash(ds) % 1000)
        seeds = rng_master.integers(0, 100000, size=args.seeds)

        for N in N_CALIB:
            n0_c = min(N, len(idx_0) - 1)
            n1_c = min(N, len(idx_1) - 1)
            if n0_c < 1 or n1_c < 1:
                continue

            a0_list, a1_list, bal_list = [], [], []
            for seed in seeds:
                rng = np.random.default_rng(seed)
                calib = np.concatenate([
                    rng.choice(idx_0, n0_c, replace=False),
                    rng.choice(idx_1, n1_c, replace=False),
                ])
                a0_s, a1_s, bal_s = cross_domain_eval(
                    X_mac, y_mac, X_tgt, y_tgt,
                    args.n_pca, int(seed), calib_idx=calib)
                a0_list.append(a0_s)
                a1_list.append(a1_s)
                bal_list.append(bal_s)

            a0m, a0s = np.nanmean(a0_list), np.nanstd(a0_list)
            a1m, a1s = np.nanmean(a1_list), np.nanstd(a1_list)
            bm,  bs  = np.nanmean(bal_list), np.nanstd(bal_list)
            label = f'Calibrado N={N}/clase'
            print(f'  {label:<35} '
                  f'{a0m:>5.3f}±{a0s:.2f} '
                  f'{a1m:>5.3f}±{a1s:.2f} '
                  f'{bm:>5.3f}±{bs:.2f}')
            records.append({'task': task_name, 'eval': label,
                            'dataset': ds_label, 'N': N,
                            'acc0': a0m, 'acc0_std': a0s,
                            'acc1': a1m, 'acc1_std': a1s,
                            'bal': bm, 'bal_std': bs})

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_pca',   type=int, default=128)
    parser.add_argument('--n_folds', type=int, default=10)
    parser.add_argument('--seeds',   type=int, default=10)
    parser.add_argument('--seed',    type=int, default=42)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f'PCA={args.n_pca}  |  CV folds={args.n_folds}  |  calib seeds={args.seeds}')

    all_records = []
    for task_name, ann_path in TASKS.items():
        records = run_task(task_name, ann_path, args)
        all_records.extend(records)

    print('\n')
    out_csv = OUT_DIR / f'results_pca{args.n_pca}_cv{args.n_folds}.csv'
    pd.DataFrame(all_records).to_csv(out_csv, index=False)
    print(f'Resultados en: {out_csv}')


if __name__ == '__main__':
    main()