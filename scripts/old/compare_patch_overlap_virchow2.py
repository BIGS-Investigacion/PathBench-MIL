#!/usr/bin/env python3
"""
compare_patch_overlap_virchow2.py

Compares pathologist baseline patches (PNG images) against the new
centroid-based selections by computing Virchow2 features directly
from the PNG images and comparing against pre-computed bag features.

Pipeline:
  1. Load all PNG patches from patches_clam_baseline (baseline)
  2. Compute Virchow2 features for each PNG
  3. Load Virchow2 features for centroid_patches_clam_att patches from .pt bags (via pidx)
  4. For each baseline patch, find nearest neighbour in the new set (Euclidean dist)
  5. Report overlap statistics (distance < threshold = same patch)

Uso:
  python scripts/compare_patch_overlap_virchow2.py [--threshold 1.0]
                                                    [--batch_size 32]
                                                    [--tasks er erbb2 pr pam50]
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms

# ── Rutas ─────────────────────────────────────────────────────────────────────

BASELINE_DIR   = Path('/shared/home/jorgarcia/PathBench-MIL/results/patches_pathologists/patches_clam_baseline')
NEW_DIR        = Path('/shared/home/jorgarcia/PathBench-MIL/results/centroid_patches_clam_att')
BAGS_DIR       = Path('/home/PARADIS/mama/tfrecords/shared_bags/256_128_none_virchow2')
OUTPUT_DIR     = Path('/shared/home/jorgarcia/PathBench-MIL/results/patch_overlap_virchow2')

TASKS = ['er', 'erbb2', 'pr', 'pam50']

# Regex para extraer slide y pidx del nombre de los ficheros nuevos
# formato: {dataset}_{slide}_pidx{pidx}_x{x}_y{y}.jpg
NEW_RE = re.compile(r'^(?P<dataset>[^_]+)_(?P<slide>.+)_pidx(?P<pidx>\d+)_x(?P<x>\d+)_y(?P<y>\d+)\.jpg$')


# ── Virchow2 ──────────────────────────────────────────────────────────────────

def load_virchow2(device: torch.device):
    model = timm.create_model(
        'hf-hub:paige-ai/Virchow2',
        pretrained=True,
        mlp_layer=timm.layers.SwiGLUPacked,
        act_layer=nn.SiLU,
    )
    model = model.eval().to(device)
    cfg = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**cfg, is_training=False)
    return model, transform


@torch.inference_mode()
def compute_features(model, transform, image_paths: list[Path],
                     batch_size: int, device: torch.device) -> np.ndarray:
    feats = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        imgs = []
        for p in batch_paths:
            img = Image.open(p).convert('RGB')
            imgs.append(transform(img))
        batch = torch.stack(imgs).to(device)
        out = model.forward_features(batch)
        # Virchow2: out shape (B, N+1, 1280) where index 0 = CLS, 1: = patch tokens
        # Bags were generated as CLS + mean(patch_tokens) → 2560 dims
        cls_token   = out[:, 0, :]           # (B, 1280)
        patch_tokens = out[:, 1:, :]          # (B, N, 1280)
        patch_mean   = patch_tokens.mean(dim=1)  # (B, 1280)
        out = torch.cat([cls_token, patch_mean], dim=1)  # (B, 2560)
        feats.append(out.cpu().numpy())
        if (i // batch_size) % 5 == 0:
            print(f'    Procesados {min(i + batch_size, len(image_paths))}/{len(image_paths)}')
    return np.vstack(feats)


# ── Carga de features de los patches nuevos desde bags ────────────────────────

def load_new_features(task: str) -> tuple[np.ndarray, list[dict]]:
    """
    Carga las features pre-computadas de los patches en centroid_patches_clam_att
    usando el pidx del nombre del fichero para indexar en el .pt bag.
    """
    task_dir = NEW_DIR / task
    records  = []
    embs     = []

    for jpg in sorted(task_dir.rglob('*.jpg')):
        m = NEW_RE.match(jpg.name)
        if not m:
            continue
        dataset = m.group('dataset')
        slide   = m.group('slide')
        pidx    = int(m.group('pidx'))
        x       = int(m.group('x'))
        y       = int(m.group('y'))
        cls_id  = jpg.parent.name

        bag_path = BAGS_DIR / f'{slide}.pt'
        if not bag_path.exists():
            continue

        bag = torch.load(bag_path, map_location='cpu', weights_only=True).numpy()
        if pidx >= len(bag):
            continue

        records.append({
            'task': task, 'dataset': dataset, 'slide': slide,
            'pidx': pidx, 'x': x, 'y': y, 'cls_id': cls_id,
            'path': str(jpg),
        })
        embs.append(bag[pidx])

    if not embs:
        return np.empty((0, 0)), []
    return np.stack(embs), records


# ── Análisis de solapamiento ──────────────────────────────────────────────────

def nn_distances(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Euclidean distance from each query to nearest gallery neighbour."""
    dists = []
    chunk = 128
    for i in range(0, len(query), chunk):
        q   = query[i:i + chunk]                              # (C, D)
        d   = np.linalg.norm(q[:, None] - gallery[None], axis=2)  # (C, G)
        dists.append(d.min(axis=1))
    return np.concatenate(dists)


def analyse_overlap(baseline_embs: np.ndarray, new_embs: np.ndarray,
                    baseline_meta: list[dict], new_meta: list[dict],
                    threshold: float) -> pd.DataFrame:
    dists = nn_distances(baseline_embs, new_embs)

    rows = []
    for i, (rec, d) in enumerate(zip(baseline_meta, dists)):
        rows.append({**rec, 'nn_dist': float(d), 'match': d < threshold})

    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Overlap between pathologist patches and centroid selections via Virchow2.')
    parser.add_argument('--threshold',  type=float, default=1.0,
                        help='Euclidean distance threshold for "same patch" (default: 1.0)')
    parser.add_argument('--batch_size', type=int,   default=32)
    parser.add_argument('--tasks',      nargs='+',  default=TASKS, choices=TASKS)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')
    print(f'Umbral : {args.threshold}\n')

    print('Cargando Virchow2...')
    model, transform = load_virchow2(device)
    print('Virchow2 cargado.\n')

    summary_rows = []

    for task in args.tasks:
        print(f'=== {task.upper()} ===')

        # ── Patches baseline (PNG) ──────────────────────────────────────────
        task_base = BASELINE_DIR / task
        base_pngs = sorted(task_base.rglob('*.png'))
        print(f'  Baseline PNGs : {len(base_pngs)}')
        if not base_pngs:
            print('  [WARN] Sin PNGs baseline.\n')
            continue

        base_meta = [{'task': task, 'path': str(p),
                      'dataset': p.parts[-3],   # …/task/dataset/class/file.png
                      'cls_id':  p.parts[-2]}
                     for p in base_pngs]

        print('  Computando features baseline con Virchow2...')
        base_embs = compute_features(model, transform, base_pngs, args.batch_size, device)
        print(f'  Features baseline : {base_embs.shape}')

        # ── Patches nuevos (desde bags) ─────────────────────────────────────
        print('  Cargando features nuevos desde bags...')
        new_embs, new_meta = load_new_features(task)
        print(f'  Features nuevos   : {new_embs.shape}')

        if new_embs.shape[0] == 0:
            print('  [WARN] Sin patches nuevos para esta tarea.\n')
            continue

        # ── Solapamiento ────────────────────────────────────────────────────
        print('  Calculando distancias NN...')
        df = analyse_overlap(base_embs, new_embs, base_meta, new_meta, args.threshold)

        n_match  = df['match'].sum()
        p_match  = 100 * n_match / len(df)
        d_min    = df['nn_dist'].min()
        d_med    = df['nn_dist'].median()
        d_max    = df['nn_dist'].max()

        print(f'  Solapamiento : {n_match}/{len(df)} ({p_match:.1f}%) dist<{args.threshold}')
        print(f'  Dist NN      : min={d_min:.4f}  med={d_med:.4f}  max={d_max:.4f}')

        # Guardar CSV detallado
        out_csv = OUTPUT_DIR / f'overlap_{task}.csv'
        df[['task', 'dataset', 'cls_id', 'path', 'nn_dist', 'match']].to_csv(out_csv, index=False)
        print(f'  Guardado: {out_csv}')

        summary_rows.append({
            'task': task, 'n_baseline': len(df), 'n_new': len(new_meta),
            'n_match': int(n_match), 'pct_match': round(p_match, 2),
            'dist_min': round(d_min, 4), 'dist_med': round(d_med, 4),
            'dist_max': round(d_max, 4),
        })
        print()

    if summary_rows:
        out_sum = OUTPUT_DIR / 'overlap_summary.csv'
        pd.DataFrame(summary_rows).to_csv(out_sum, index=False)
        print(f'Resumen guardado: {out_sum}')
        print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == '__main__':
    main()