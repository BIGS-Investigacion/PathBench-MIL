#!/usr/bin/env python3
"""
select_closest_to_centroid.py

A partir de los top-8 patches por slide (top8_patches_{dataset}.csv + centroids_{dataset}.csv):
  1. Carga las features H-Optimus-0 de cada patch
  2. Calcula distancia coseno al centroide de su clase
  3. Selecciona los 100 más cercanos por clase
  4. Copia las imágenes a results/colon_mss_msi/closest100_{dataset}/{MSS,MSI}/
  5. Elimina top_patches/{dataset}/

Uso:
  python scripts/colon/select_closest_to_centroid.py [--top_n 100] --dataset tcga_coad
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT      = Path(__file__).resolve().parents[2]
BAGS_DIR  = Path('/home/PARADIS/colon/tfrecords/shared_bags/256_128_none_h_optimus_0')
OUT_DIR   = ROOT / 'results/colon'
PATCH_DIR = OUT_DIR / 'top_patches'


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_n', type=int, default=100)
    parser.add_argument('--dataset', type=str, default='macarena',
                        choices=['macarena', 'tcga_coad', 'cptac_coad'])
    args = parser.parse_args()

    patch_dir = PATCH_DIR / args.dataset
    close_dir = OUT_DIR / f'closest100_{args.dataset}'

    # Cargar centroides
    df_cent = pd.read_csv(OUT_DIR / f'centroids_{args.dataset}.csv')
    feat_cols = [c for c in df_cent.columns if c.startswith('feat_')]
    centroids = {}
    for _, row in df_cent.iterrows():
        centroids[row['class']] = row[feat_cols].values.astype(np.float32)
    print(f'Centroides cargados: {list(centroids.keys())}')

    # Cargar top8_patches
    df_top = pd.read_csv(OUT_DIR / f'top8_patches_{args.dataset}.csv')
    print(f'Total patches: {len(df_top)}')

    records = []
    slides_loaded = {}

    for i, row in df_top.iterrows():
        slide    = row['slide']
        cls_name = row['class']
        rank     = int(row['rank'])
        centroid = centroids[cls_name]

        if slide not in slides_loaded:
            bag_path = BAGS_DIR / f'{slide}.pt'
            if not bag_path.exists():
                continue
            slides_loaded[slide] = torch.load(bag_path, map_location='cpu',
                                              weights_only=True).numpy()
        bag = slides_loaded[slide]

        idx_path = BAGS_DIR / f'{slide}.index.npz'
        if not idx_path.exists():
            continue
        coords = np.load(idx_path)['arr_0']
        x, y = int(row['x']), int(row['y'])
        matches = np.where((coords[:, 0] == x) & (coords[:, 1] == y))[0]
        if len(matches) == 0:
            continue
        patch_idx = matches[0]
        feat = bag[patch_idx].astype(np.float32)

        dist = cosine_distance(feat, centroid)
        img_path = patch_dir / cls_name / slide / f'rank{rank:02d}_att{row["attention"]:.4f}.png'

        records.append({
            'slide': slide, 'class': cls_name, 'rank': rank,
            'x': x, 'y': y, 'attention': row['attention'],
            'cosine_dist': dist, 'img_path': str(img_path),
        })

        if (i + 1) % 200 == 0:
            print(f'  [{i+1}/{len(df_top)}]')

    df_rec = pd.DataFrame(records)
    df_rec.to_csv(OUT_DIR / f'top8_with_distances_{args.dataset}.csv', index=False)
    print(f'\nDistancias calculadas: {len(df_rec)} patches')

    for cls_name in ['MSS', 'MSI']:
        sub = df_rec[df_rec['class'] == cls_name].nsmallest(args.top_n, 'cosine_dist')
        out_cls = close_dir / cls_name
        out_cls.mkdir(parents=True, exist_ok=True)

        copied = 0
        for rank_new, (_, r) in enumerate(sub.iterrows(), 1):
            src = Path(r['img_path'])
            if src.exists():
                dst = out_cls / f'{rank_new:03d}_{r["slide"]}_d{r["cosine_dist"]:.4f}.png'
                shutil.copy(src, dst)
                copied += 1

        print(f'{cls_name}: {copied}/{args.top_n} imágenes copiadas → {out_cls}')
        print(f'  dist range: {sub["cosine_dist"].min():.4f} – {sub["cosine_dist"].max():.4f}')
        sub.to_csv(OUT_DIR / f'closest{args.top_n}_{cls_name.lower()}_{args.dataset}.csv', index=False)

    # Eliminar top_patches/{dataset}
    if patch_dir.exists():
        shutil.rmtree(patch_dir)
        print(f'\nEliminado: {patch_dir}')


if __name__ == '__main__':
    main()