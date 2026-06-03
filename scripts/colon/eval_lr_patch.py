#!/usr/bin/env python3
"""
eval_lr_patch.py

LR entrenada a nivel de patch con todos los patches de Macarena.
Etiqueta de slide asignada a cada patch (supervisión débil).
Inferencia: media de probabilidades sobre todos los patches del slide test.

Sin etiquetas del target — completamente honesto.

Uso:
  python scripts/colon/eval_lr_patch.py [--n_pca 128] [--max_patches_per_slide 500]
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
OUT_DIR  = ROOT / 'results/colon/lr_patch'

DATASET_LABELS = {'macarena': 'Macarena', 'cptac_coad': 'CPTAC-COAD', 'tcga_coad': 'TCGA-COAD'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_pca',               type=int,   default=128)
    parser.add_argument('--max_patches_per_slide', type=int, default=500,
                        help='Máx. patches por slide para entrenar LR (0=todos)')
    parser.add_argument('--seed',                type=int,   default=42)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    ann = pd.read_csv(ANNOT)

    # 1. Ajustar PCA sobre slide means de Macarena (consistente con el resto)
    print('Ajustando PCA sobre slide means de Macarena...')
    mac_rows = ann[ann['dataset'] == 'macarena']
    means = []
    for _, row in mac_rows.iterrows():
        p = BAGS_DIR / f'{row["slide"]}.pt'
        if not p.exists(): continue
        feat = torch.load(p, map_location='cpu', weights_only=True).float().numpy()
        means.append(feat.mean(axis=0))
    means = np.stack(means).astype(np.float32)
    scaler = StandardScaler().fit(means)
    pca    = PCA(n_components=args.n_pca, random_state=args.seed).fit(scaler.transform(means))
    print(f'  Varianza explicada: {pca.explained_variance_ratio_.sum():.1%}')

    # 2. Cargar todos los patches de Macarena en espacio PCA
    print('\nCargando patches de Macarena...')
    X_train, y_train = [], []
    for _, row in mac_rows.iterrows():
        p = BAGS_DIR / f'{row["slide"]}.pt'
        if not p.exists(): continue
        feat = torch.load(p, map_location='cpu', weights_only=True).float().numpy()
        if args.max_patches_per_slide > 0 and feat.shape[0] > args.max_patches_per_slide:
            idx = rng.choice(feat.shape[0], args.max_patches_per_slide, replace=False)
            feat = feat[idx]
        feat_pca = pca.transform(scaler.transform(feat))
        X_train.append(feat_pca)
        y_train.extend([int(row['category'])] * len(feat_pca))

    X_train = np.concatenate(X_train).astype(np.float32)
    y_train = np.array(y_train)
    n_mss = (y_train == 0).sum()
    n_msi = (y_train == 1).sum()
    print(f'  Patches totales: {len(X_train):,}  (MSS={n_mss:,}, MSI={n_msi:,})')

    # 3. Entrenar LR
    print('\nEntrenando LR...')
    lr = LogisticRegression(max_iter=1000, random_state=args.seed, C=1.0,
                            class_weight='balanced', solver='saga', n_jobs=-1)
    lr.fit(X_train, y_train)
    print('  LR entrenada.')

    # 4. Inferencia slide a slide sobre CPTAC y TCGA
    print('\nInferencia por slide:')
    print(f'  {"Dataset":<12}  {"Slide":<45}  {"pred":>4}  {"true":>4}  {"p(MSI)":>6}')
    print('  ' + '-'*75)

    results = []
    for ds in ['cptac_coad', 'tcga_coad']:
        ds_rows = ann[ann['dataset'] == ds]
        preds, labels, probs = [], [], []
        for _, row in ds_rows.iterrows():
            p = BAGS_DIR / f'{row["slide"]}.pt'
            if not p.exists(): continue
            feat = torch.load(p, map_location='cpu', weights_only=True).float().numpy()
            feat_pca = pca.transform(scaler.transform(feat))
            # Agregación: media de probabilidades → predicción de slide
            prob_msi = lr.predict_proba(feat_pca)[:, 1].mean()
            pred = int(prob_msi >= 0.5)
            preds.append(pred)
            labels.append(int(row['category']))
            probs.append(prob_msi)

        preds  = np.array(preds)
        labels = np.array(labels)
        probs  = np.array(probs)

        mss_mask = labels == 0
        msi_mask = labels == 1
        acc_mss = (preds[mss_mask] == 0).mean() if mss_mask.sum() > 0 else float('nan')
        acc_msi = (preds[msi_mask] == 1).mean() if msi_mask.sum() > 0 else float('nan')
        bal     = balanced_accuracy_score(labels, preds)

        print(f'\n  {DATASET_LABELS[ds]}')
        print(f'    MSS acc = {acc_mss:.3f}  ({mss_mask.sum()} slides)')
        print(f'    MSI acc = {acc_msi:.3f}  ({msi_mask.sum()} slides)')
        print(f'    Bal-acc = {bal:.3f}')

        results.append({'dataset': DATASET_LABELS[ds],
                        'acc_mss': acc_mss, 'acc_msi': acc_msi, 'bal_acc': bal,
                        'n_mss': mss_mask.sum(), 'n_msi': msi_mask.sum()})

    pd.DataFrame(results).to_csv(
        OUT_DIR / f'results_pca{args.n_pca}_mp{args.max_patches_per_slide}.csv', index=False)
    print(f'\nResultados en: {OUT_DIR}')


if __name__ == '__main__':
    main()