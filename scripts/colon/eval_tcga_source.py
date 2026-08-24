#!/usr/bin/env python3
"""
eval_tcga_source.py

Problema invertido: TCGA-COAD como dominio fuente (train), MSS/MSI.
Representación: slide mean (media de todos los patches) + PCA + LR.

  1. K-fold CV estratificado por paciente dentro de TCGA.
  2. Entrenar con TODO TCGA y validar de forma independiente en Macarena y CPTAC
     (sin calibrar, y con reajuste warm-start N=5/10/20 por clase como referencia).

Uso:
  python scripts/colon/eval_tcga_source.py [--n_pca 128] [--n_folds 10] [--seeds 10]
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
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings('ignore')

ROOT     = Path(__file__).resolve().parents[2]
BAGS_DIR = Path('/home/PARADIS/datos/features/shared_bags/256_128_none_h_optimus_0')
ANNOT    = ROOT / 'config/annotations/annotations_colon_mss_msi_all.csv'
OUT_DIR  = ROOT / 'results/colon/tcga_source'

N_CALIB = [5, 10, 20]

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
        rows.append({'slide': row['slide'], 'patient': row['patient'],
                     'dataset': row['dataset'], 'category': int(row['category'])})
    return np.stack(vecs).astype(np.float32), pd.DataFrame(rows)


def fit_pca(X, n_pca, seed):
    sc  = StandardScaler().fit(X)
    pca = PCA(n_components=min(n_pca, X.shape[0]-1, X.shape[1]),
              random_state=seed).fit(sc.transform(X))
    return sc, pca


def cv_eval(X, y, groups, n_folds, n_pca, seed):
    """K-fold CV estratificado por clase + agrupado por paciente."""
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    a0, a1, ab = [], [], []
    for tr, te in sgkf.split(X, y, groups=groups):
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


def cross_domain(X_src, y_src, X_tgt, y_tgt, n_pca, seed, calib_idx=None):
    """LR entrenada en TCGA; opcional reajuste warm-start con N slides del target."""
    sc, pca = fit_pca(X_src, n_pca, seed)
    X_src_p = pca.transform(sc.transform(X_src))
    X_tgt_p = pca.transform(sc.transform(X_tgt))

    lr = LogisticRegression(max_iter=2000, warm_start=True, random_state=seed, C=1.0)
    lr.fit(X_src_p, y_src)

    if calib_idx is not None and len(calib_idx) > 0:
        lr.fit(X_tgt_p[calib_idx], y_tgt[calib_idx])
        test_idx = np.setdiff1d(np.arange(len(y_tgt)), calib_idx)
    else:
        test_idx = np.arange(len(y_tgt))

    if len(test_idx) == 0:
        return np.nan, np.nan, np.nan
    yp   = lr.predict(X_tgt_p[test_idx])
    y_te = y_tgt[test_idx]
    a0 = (yp[y_te==0]==0).mean() if (y_te==0).sum() > 0 else np.nan
    a1 = (yp[y_te==1]==1).mean() if (y_te==1).sum() > 0 else np.nan
    return a0, a1, balanced_accuracy_score(y_te, yp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_pca',   type=int, default=128)
    parser.add_argument('--n_folds', type=int, default=10)
    parser.add_argument('--seeds',   type=int, default=10)
    parser.add_argument('--seed',    type=int, default=42)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f'FUENTE = TCGA-COAD  |  PCA={args.n_pca}  |  CV folds={args.n_folds}  |  seeds={args.seeds}\n')

    ann = pd.read_csv(ANNOT)
    X_raw, meta = load_features(ann)

    mask_src = (meta['dataset'] == 'tcga_coad').values
    X_src = X_raw[mask_src]
    y_src = meta.loc[mask_src, 'category'].values
    grp   = meta.loc[mask_src, 'patient'].values
    n0, n1 = (y_src==0).sum(), (y_src==1).sum()

    records = []

    # ── 1. CV en TCGA ────────────────────────────────────────────────────────
    print(f'{"="*64}')
    print(f'1) {args.n_folds}-fold CV en TCGA-COAD  (MSS={n0}, MSI={n1})')
    print(f'{"="*64}')
    a0, a1, bal = cv_eval(X_src, y_src, grp, args.n_folds, args.n_pca, args.seed)
    print(f'  {"":18} {"MSS":>8} {"MSI":>8} {"bal":>8}')
    print(f'  {"TCGA "+str(args.n_folds)+"-fold CV":<18} {a0:>8.3f} {a1:>8.3f} {bal:>8.3f}')
    records.append({'eval': f'TCGA {args.n_folds}-fold CV', 'dataset': 'tcga_coad',
                    'N': None, 'acc0': a0, 'acc1': a1, 'bal': bal, 'bal_std': np.nan})

    # ── 2. Train TODO TCGA → validar en Macarena y CPTAC ─────────────────────
    print(f'\n{"="*64}')
    print(f'2) Train TODO TCGA-COAD → validación independiente')
    print(f'{"="*64}')

    for ds in ['macarena', 'cptac_coad']:
        mask = (meta['dataset'] == ds).values
        X_tgt = X_raw[mask]
        y_tgt = meta.loc[mask, 'category'].values
        idx_0 = np.where(y_tgt == 0)[0]
        idx_1 = np.where(y_tgt == 1)[0]
        label = {'macarena': 'Macarena', 'cptac_coad': 'CPTAC-COAD'}[ds]
        print(f'\n  {label}  (MSS={len(idx_0)}, MSI={len(idx_1)})')
        print(f'  {"":22} {"MSS":>8} {"MSI":>8} {"bal":>8}')

        # Sin calibrar
        a0, a1, bal = cross_domain(X_src, y_src, X_tgt, y_tgt, args.n_pca, args.seed)
        print(f'  {"Sin calibrar":<22} {a0:>8.3f} {a1:>8.3f} {bal:>8.3f}')
        records.append({'eval': 'Sin calibrar', 'dataset': label, 'N': 0,
                        'acc0': a0, 'acc1': a1, 'bal': bal, 'bal_std': np.nan})

        # Reajuste warm-start (referencia)
        rng_master = np.random.default_rng(args.seed + hash(ds) % 1000)
        seeds = rng_master.integers(0, 100000, size=args.seeds)
        for N in N_CALIB:
            n0_c = min(N, len(idx_0) - 1)
            n1_c = min(N, len(idx_1) - 1)
            if n0_c < 1 or n1_c < 1:
                continue
            a0l, a1l, bl = [], [], []
            for s in seeds:
                rng = np.random.default_rng(s)
                calib = np.concatenate([rng.choice(idx_0, n0_c, replace=False),
                                        rng.choice(idx_1, n1_c, replace=False)])
                r0, r1, rb = cross_domain(X_src, y_src, X_tgt, y_tgt,
                                          args.n_pca, int(s), calib_idx=calib)
                a0l.append(r0); a1l.append(r1); bl.append(rb)
            print(f'  {"Reajuste N="+str(N):<22} '
                  f'{np.nanmean(a0l):>8.3f} {np.nanmean(a1l):>8.3f} '
                  f'{np.nanmean(bl):>7.3f}±{np.nanstd(bl):.2f}')
            records.append({'eval': f'Reajuste N={N}', 'dataset': label, 'N': N,
                            'acc0': np.nanmean(a0l), 'acc1': np.nanmean(a1l),
                            'bal': np.nanmean(bl), 'bal_std': np.nanstd(bl)})

    out = OUT_DIR / f'results_pca{args.n_pca}.csv'
    pd.DataFrame(records).to_csv(out, index=False)
    print(f'\nGuardado en: {out}')


if __name__ == '__main__':
    main()
