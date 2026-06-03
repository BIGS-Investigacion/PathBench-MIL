#!/usr/bin/env python3
"""
eval_lr_representations.py

Compara dos representaciones de slides con PCA + LR:
  1. Media de TODOS los patches (slide mean)
  2. Media de los K patches con mayor atención DSMIL (top-K attention mean)

Para cada representación:
  - K-fold CV estratificado en Macarena (accuracy por clase + balanced accuracy)
  - LR entrenada en Macarena, evaluada en CPTAC y TCGA

Uso:
  python scripts/colon/eval_lr_representations.py [--k_attn 8] [--n_folds 5]
"""

import warnings
import json
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')

ROOT      = Path(__file__).resolve().parents[2]
BAGS_DIR  = Path('/home/PARADIS/colon/tfrecords/shared_bags/256_128_none_h_optimus_0')
MODEL_DIR = ROOT / 'experiments/colon_mss_msi_benchmark/mil/00000-256_128_none_h_optimus_0_dsmil_CrossEntropyLoss_ReLU_Adam_1'
ANNOT     = ROOT / 'config/annotations/annotations_colon_mss_msi_all.csv'
OUT_DIR   = ROOT / 'results/colon/lr_representations'

DATASET_LABELS = {'macarena': 'Macarena', 'cptac_coad': 'CPTAC-COAD', 'tcga_coad': 'TCGA-COAD'}


def load_dsmil(device):
    spec = importlib.util.spec_from_file_location(
        'aggregators', ROOT / 'pathbench/models/aggregators.py')
    agg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agg)
    with open(MODEL_DIR / 'mil_params.json') as f:
        mp = json.load(f)
    params = mp['params']
    model = getattr(agg, params['model'])(
        n_feats=mp['input_shape'], n_out=mp['output_shape'],
        z_dim=params.get('z_dim', 256), dropout_p=params.get('dropout_p', 0.1),
        activation_function=params.get('activation_function', 'ReLU'),
        encoder_layers=params.get('encoder_layers', 1),
    )
    ckpt  = sorted(MODEL_DIR.glob('checkpoints/**/best-epoch*.ckpt'))[-1]
    state = torch.load(ckpt, map_location='cpu')
    sd    = {k.removeprefix('model.'): v
             for k, v in state['state_dict'].items() if k.startswith('model.')}
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()


def get_representations(model, slide, device, k_attn):
    """Devuelve (slide_mean, topk_attn_mean) para un slide."""
    p = BAGS_DIR / f'{slide}.pt'
    if not p.exists():
        return None, None
    bag = torch.load(p, map_location='cpu', weights_only=True).float()
    slide_mean = bag.mean(dim=0).numpy().astype(np.float32)

    with torch.no_grad():
        x           = bag.unsqueeze(0).to(device)
        inst_feat   = model.instance_encoder(x.view(-1, x.size(-1)))
        inst_feat   = inst_feat.view(1, bag.shape[0], -1)
        inst_scores = model.instance_classifier(inst_feat).view(1, bag.shape[0])
        _, max_idx  = inst_scores.max(dim=1)
        critical    = inst_feat[0, max_idx[0]]
        attn_w      = F.softmax(
            model.attention(inst_feat - critical.unsqueeze(0).unsqueeze(0)), dim=1
        ).squeeze()
        topk_idx    = attn_w.topk(min(k_attn, bag.shape[0])).indices
    topk_mean = bag[topk_idx.cpu()].mean(dim=0).numpy().astype(np.float32)

    return slide_mean, topk_mean


def cv_eval(X, y, n_folds, n_pca, seed):
    """K-fold CV estratificado con PCA + LR. Devuelve (mss_acc, msi_acc, bal_acc)."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    mss_list, msi_list, bal_list = [], [], []
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
        mss_list.append((yp[y_te==0] == 0).mean() if (y_te==0).sum() > 0 else np.nan)
        msi_list.append((yp[y_te==1] == 1).mean() if (y_te==1).sum() > 0 else np.nan)
        bal_list.append(balanced_accuracy_score(y_te, yp))
    return np.nanmean(mss_list), np.nanmean(msi_list), np.nanmean(bal_list)


def cross_domain_eval(X_mac, y_mac, X_test, y_test, n_pca, seed):
    """LR entrenada en Macarena, evaluada en dominio test."""
    sc  = StandardScaler().fit(X_mac)
    pca = PCA(n_components=n_pca, random_state=seed).fit(sc.transform(X_mac))
    X_mac_p  = pca.transform(sc.transform(X_mac))
    X_test_p = pca.transform(sc.transform(X_test))
    lr = LogisticRegression(max_iter=2000, random_state=seed, C=1.0)
    lr.fit(X_mac_p, y_mac)
    yp = lr.predict(X_test_p)
    acc_mss = (yp[y_test==0] == 0).mean() if (y_test==0).sum() > 0 else np.nan
    acc_msi = (yp[y_test==1] == 1).mean() if (y_test==1).sum() > 0 else np.nan
    return acc_mss, acc_msi, balanced_accuracy_score(y_test, yp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k_attn',  type=int, default=8,
                        help='Top-K patches por atención')
    parser.add_argument('--n_pca',   type=int, default=128)
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--seed',    type=int, default=42)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}  |  top-K={args.k_attn}  |  PCA={args.n_pca}  |  folds={args.n_folds}\n')

    ann = pd.read_csv(ANNOT)
    model = load_dsmil(device)

    print('Extrayendo representaciones...')
    means, topks, metas = [], [], []
    for _, row in ann.iterrows():
        sm, tk = get_representations(model, row['slide'], device, args.k_attn)
        if sm is None:
            continue
        means.append(sm)
        topks.append(tk)
        metas.append({'slide': row['slide'], 'dataset': row['dataset'],
                      'category': int(row['category'])})

    meta   = pd.DataFrame(metas)
    X_mean = np.stack(means).astype(np.float32)
    X_topk = np.stack(topks).astype(np.float32)
    print(f'  Slides: {len(meta)}\n')

    mask_mac = (meta['dataset'] == 'macarena').values

    print(f'{"Representación":<30} {"Evaluación":<22} {"MSS":>8} {"MSI":>8} {"bal":>8}')
    print('=' * 80)

    rows = []
    for rep_name, X_all in [('Slide mean (todos los patches)', X_mean),
                             (f'Top-{args.k_attn} atención (DSMIL)', X_topk)]:
        print(f'\n  {rep_name}')
        print('-' * 80)

        X_mac = X_all[mask_mac]
        y_mac = meta.loc[mask_mac, 'category'].values

        mss, msi, bal = cv_eval(X_mac, y_mac, args.n_folds, args.n_pca, args.seed)
        print(f'  {"Macarena "+str(args.n_folds)+"-fold CV":<52} '
              f'{mss:>8.3f} {msi:>8.3f} {bal:>8.3f}')
        rows.append({'rep': rep_name, 'eval': f'Macarena {args.n_folds}-fold CV',
                     'mss': mss, 'msi': msi, 'bal': bal})

        for ds in ['cptac_coad', 'tcga_coad']:
            mask_ds = (meta['dataset'] == ds).values
            X_test  = X_all[mask_ds]
            y_test  = meta.loc[mask_ds, 'category'].values
            mss, msi, bal = cross_domain_eval(X_mac, y_mac, X_test, y_test,
                                              args.n_pca, args.seed)
            ds_name = DATASET_LABELS[ds]
            print(f'  {ds_name+" (test)":<52} '
                  f'{mss:>8.3f} {msi:>8.3f} {bal:>8.3f}')
            rows.append({'rep': rep_name, 'eval': ds_name,
                         'mss': mss, 'msi': msi, 'bal': bal})

    print()
    out_csv = OUT_DIR / f'results_k{args.k_attn}_pca{args.n_pca}.csv'
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f'Resultados en: {out_csv}')


if __name__ == '__main__':
    main()