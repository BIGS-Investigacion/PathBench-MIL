#!/usr/bin/env python3
"""
eval_lr_dsmil_embed.py

Extrae el embedding de atención de DSMIL (conv_out, 270 dims) para cada slide
y entrena una LR sobre los slides de Macarena. Evalúa en CPTAC y TCGA.

Sin etiquetas del target — completamente honesto.

Uso:
  python scripts/colon/eval_lr_dsmil_embed.py
"""

import warnings
import json
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

warnings.filterwarnings('ignore')

ROOT      = Path(__file__).resolve().parents[2]
BAGS_DIR  = Path('/home/PARADIS/colon/tfrecords/shared_bags/256_128_none_h_optimus_0')
MODEL_DIR = ROOT / 'experiments/colon_mss_msi_benchmark/mil/00000-256_128_none_h_optimus_0_dsmil_CrossEntropyLoss_ReLU_Adam_1'
ANNOT     = ROOT / 'config/annotations/annotations_colon_mss_msi_all.csv'
OUT_DIR   = ROOT / 'results/colon/lr_dsmil_embed'

DATASET_LABELS = {'macarena': 'Macarena', 'cptac_coad': 'CPTAC-COAD', 'tcga_coad': 'TCGA-COAD'}


def load_model(device):
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


def extract_embed(model, slide, device):
    """Extrae el conv_out (270 dims) de DSMIL para un slide."""
    p = BAGS_DIR / f'{slide}.pt'
    if not p.exists():
        return None
    bag = torch.load(p, map_location='cpu', weights_only=True).float()
    with torch.no_grad():
        x = bag.unsqueeze(0).to(device)
        # Replicar forward de dsmil hasta conv_out
        batch_size, num_instances, _ = x.size()
        inst_feat = model.instance_encoder(x.view(-1, x.size(-1)))
        inst_feat = inst_feat.view(batch_size, num_instances, -1)
        inst_scores = model.instance_classifier(inst_feat).view(batch_size, num_instances)
        _, max_idx = inst_scores.max(dim=1)
        critical = inst_feat[torch.arange(batch_size), max_idx]
        attn = torch.nn.functional.softmax(
            model.attention(inst_feat - critical.unsqueeze(1)), dim=1)
        bag_emb = (attn * inst_feat).sum(dim=1).unsqueeze(-1)
        conv_out = model.conv1d(bag_emb).squeeze(-1)
    return conv_out.squeeze(0).cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')

    ann = pd.read_csv(ANNOT)
    model = load_model(device)

    # 1. Extraer embeddings para todos los slides
    print('Extrayendo embeddings DSMIL...')
    embeds, metas = {}, []
    for _, row in ann.iterrows():
        emb = extract_embed(model, row['slide'], device)
        if emb is None:
            continue
        embeds[row['slide']] = emb
        metas.append({'slide': row['slide'], 'dataset': row['dataset'],
                      'category': int(row['category'])})
    meta = pd.DataFrame(metas)
    print(f'  Slides procesados: {len(meta)}')

    X_all = np.stack([embeds[s] for s in meta['slide']]).astype(np.float32)

    # 2. LR entrenada solo en Macarena
    mask_mac = (meta['dataset'] == 'macarena').values
    X_mac = X_all[mask_mac]
    y_mac = meta.loc[mask_mac, 'category'].values
    print(f'  Macarena: {len(X_mac)} slides  '
          f'(MSS={( y_mac==0).sum()}, MSI={(y_mac==1).sum()})')

    lr = LogisticRegression(max_iter=2000, random_state=args.seed,
                            C=1.0, class_weight='balanced')
    lr.fit(X_mac, y_mac)
    print('\nLR entrenada sobre embeddings de Macarena.\n')

    # 3. Evaluar en CPTAC y TCGA
    rows = []
    print(f'  {"Dataset":<12}  {"MSS acc":>8}  {"MSI acc":>8}  {"bal-acc":>8}  {"n":>5}')
    print('  ' + '-'*52)
    for ds in ['macarena', 'cptac_coad', 'tcga_coad']:
        mask_ds = (meta['dataset'] == ds).values
        X_ds = X_all[mask_ds]
        y_ds = meta.loc[mask_ds, 'category'].values
        y_pred = lr.predict(X_ds)
        acc_mss = (y_pred[y_ds==0] == 0).mean() if (y_ds==0).sum() > 0 else float('nan')
        acc_msi = (y_pred[y_ds==1] == 1).mean() if (y_ds==1).sum() > 0 else float('nan')
        bal     = balanced_accuracy_score(y_ds, y_pred)
        print(f'  {DATASET_LABELS[ds]:<12}  {acc_mss:>8.3f}  {acc_msi:>8.3f}  {bal:>8.3f}  {len(y_ds):>5}')
        rows.append({'dataset': DATASET_LABELS[ds], 'acc_mss': acc_mss,
                     'acc_msi': acc_msi, 'bal_acc': bal})

    pd.DataFrame(rows).to_csv(OUT_DIR / 'results.csv', index=False)
    print(f'\nResultados en: {OUT_DIR}')


if __name__ == '__main__':
    main()