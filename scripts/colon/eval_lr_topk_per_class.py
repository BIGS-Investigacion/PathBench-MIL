#!/usr/bin/env python3
"""
eval_lr_topk_per_class.py

Igual que eval_lr_finetune_per_class.py pero usando la media de los K patches
con mayor atención DSMIL como representación del slide (en lugar de slide mean).

LR calibrada con N slides etiquetados del target.
Muestra accuracy por clase (MSS, MSI) + balanced accuracy.

Uso:
  python scripts/colon/eval_lr_topk_per_class.py [--k 8] [--seeds 20]
"""

import argparse
import warnings
import json
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

warnings.filterwarnings('ignore')

ROOT      = Path(__file__).resolve().parents[2]
BAGS_DIR  = Path('/home/PARADIS/colon/tfrecords/shared_bags/256_128_none_h_optimus_0')
MODEL_DIR = ROOT / 'experiments/colon_mss_msi_benchmark/mil/00000-256_128_none_h_optimus_0_dsmil_CrossEntropyLoss_ReLU_Adam_1'
ANNOT     = ROOT / 'config/annotations/annotations_colon_mss_msi_all.csv'

N_VALUES = [5, 10, 20, 30]


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


def topk_attn_mean(model, slide, device, k):
    p = BAGS_DIR / f'{slide}.pt'
    if not p.exists():
        return None
    bag = torch.load(p, map_location='cpu', weights_only=True).float()
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
        topk_idx    = attn_w.topk(min(k, bag.shape[0])).indices
    return bag[topk_idx.cpu()].mean(dim=0).numpy().astype(np.float32)


def load_features(ann, model, device, k):
    vecs, rows = [], []
    for i, (_, row) in enumerate(ann.iterrows()):
        feat = topk_attn_mean(model, row['slide'], device, k)
        if feat is None:
            continue
        vecs.append(feat)
        rows.append({'slide': row['slide'], 'dataset': row['dataset'],
                     'category': int(row['category'])})
        if (i + 1) % 100 == 0:
            print(f'  {i+1}/{len(ann)} slides procesados...', flush=True)
    return np.stack(vecs).astype(np.float32), pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k',             type=int, default=8)
    parser.add_argument('--n_pca',         type=int, default=128)
    parser.add_argument('--seeds',         type=int, default=20)
    parser.add_argument('--seed0',         type=int, default=0)
    parser.add_argument('--min_per_class', type=int, default=5)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}  |  top-K={args.k}  |  PCA={args.n_pca}\n')

    ann   = pd.read_csv(ANNOT)
    model = load_dsmil(device)

    print('Extrayendo representaciones top-K atención...')
    X_raw, meta = load_features(ann, model, device, args.k)
    print(f'  Total slides: {len(meta)}\n')

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