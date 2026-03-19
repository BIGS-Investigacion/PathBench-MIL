#!/usr/bin/env python3
"""
patch_intersection_clam.py

Para cada tarea y cada WSI, carga las puntuaciones de atención de CLAM-MB
con normalización 'none', selecciona la columna correspondiente a la clase
ground-truth del slide y elige el top-K patches.

A diferencia de patch_intersection.py (que promedia los 3 modelos), este
script usa exclusivamente CLAM-MB con atención específica por clase.

Salida por tarea:
  <output_dir>/<task>/top8_patches.csv
    task, slide, rank, patch_idx, x, y, att_clam_mil_mb

Uso:
  python scripts/patch_intersection_clam.py [--top_k 8] [--tasks er erbb2 pr pam50]
                                            [--output_dir results/patch_intersection_clam]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ── Configuración de rutas ────────────────────────────────────────────────────

EXPERIMENTS_DIR = Path('/shared/home/jorgarcia/PathBench-MIL/experiments')
BAGS_DIR        = Path('/home/PARADIS/mama/tfrecords/shared_bags/256_128_none_virchow2')

TASKS          = ['er', 'erbb2', 'pr', 'pam50']
MIL_KEY        = 'clam_mil_mb'
TASK_N_CLASSES = {'er': 2, 'erbb2': 2, 'pr': 2, 'pam50': 5}

ANNOT_DIR  = Path('/shared/home/jorgarcia/PathBench-MIL/config/annotations')
ANNOT_FILE = {
    'er':    ANNOT_DIR / 'annotations_brca_er.csv',
    'erbb2': ANNOT_DIR / 'annotations_brca_erbb2.csv',
    'pr':    ANNOT_DIR / 'annotations_brca_pr.csv',
    'pam50': ANNOT_DIR / 'annotations_brca_pam50.csv',
}


# ── Funciones auxiliares ──────────────────────────────────────────────────────

def find_eval_subdir(mil_eval_dir: Path, norm_prefix: str = '00001') -> Path | None:
    candidates = sorted(mil_eval_dir.glob(f'{norm_prefix}-*none*'))
    return candidates[0] if candidates else None


def find_model_subdir(none_eval_dir: Path) -> Path | None:
    candidates = sorted(none_eval_dir.glob('00000-*'))
    return candidates[0] if candidates else None


def load_attention_class(att_path: Path, n_patches: int, n_classes: int,
                         cls_id: int) -> np.ndarray:
    """
    Carga atención de CLAM-MB y devuelve vector (n_patches,) para cls_id.
    CLAM-MB guarda (n_patches * n_classes,) → reshape y selección de columna.
    """
    arr = np.load(att_path)['arr_0']
    if arr.shape[0] == n_patches * n_classes and n_classes > 1:
        return arr.reshape(n_patches, n_classes)[:, cls_id]
    elif arr.shape[0] == n_patches:
        return arr  # modelo binario con un solo vector
    else:
        raise ValueError(
            f'Shape inesperado: {arr.shape}, esperado ({n_patches},) '
            f'o ({n_patches * n_classes},)'
        )


def minmax_norm(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def load_slide_info(task: str) -> dict:
    """Devuelve {slide: {'dataset': str, 'category': int}}."""
    annot = pd.read_csv(ANNOT_FILE[task])[['slide', 'dataset', 'category']]
    return annot.set_index('slide')[['dataset', 'category']].to_dict('index')


# ── Selección de patches ──────────────────────────────────────────────────────

def process_task(task: str, top_k: int, output_dir: Path,
                 slide_info: dict) -> None:
    n_classes = TASK_N_CLASSES[task]
    task_out  = output_dir / task
    task_out.mkdir(parents=True, exist_ok=True)

    exp_dir   = EXPERIMENTS_DIR / f'brca_virchow2_test_{task}_{MIL_KEY}'
    none_eval = find_eval_subdir(exp_dir / 'mil_eval')
    if none_eval is None:
        print(f'  [ERROR] No eval none para {task}/{MIL_KEY}')
        return
    model_dir = find_model_subdir(none_eval)
    if model_dir is None:
        print(f'  [ERROR] No model subdir para {task}/{MIL_KEY}')
        return
    att_dir = model_dir / 'attention'

    slides = sorted(p.name.replace('_att.npz', '') for p in att_dir.glob('*_att.npz'))
    print(f'  Slides disponibles: {len(slides)}')

    rows = []
    for slide in slides:
        if slide not in slide_info:
            continue
        cls_id = int(slide_info[slide]['category'])

        idx_file = BAGS_DIR / f'{slide}.index.npz'
        if not idx_file.exists():
            print(f'  [WARN] Sin index para {slide}')
            continue
        coords    = np.load(idx_file)['arr_0']
        n_patches = len(coords)

        att_path = att_dir / f'{slide}_att.npz'
        if not att_path.exists():
            continue
        try:
            att      = load_attention_class(att_path, n_patches, n_classes, cls_id)
            att_norm = minmax_norm(att)
        except ValueError as e:
            print(f'  [WARN] {slide}: {e}')
            continue

        k       = min(top_k, n_patches)
        top_idx = np.argsort(att_norm)[-k:][::-1]

        for rank, idx in enumerate(top_idx, start=1):
            rows.append({
                'task':            task,
                'slide':           slide,
                'rank':            rank,
                'patch_idx':       int(idx),
                'x':               int(coords[idx, 0]),
                'y':               int(coords[idx, 1]),
                'att_clam_mil_mb': round(float(att_norm[idx]), 6),
            })

    if rows:
        df  = pd.DataFrame(rows)
        out = task_out / f'top{top_k}_patches.csv'
        df.to_csv(out, index=False)
        print(f'  Guardado: {out}  ({len(df)} filas, {df["slide"].nunique()} slides)')
    else:
        print(f'  Sin datos para {task}.')


# ── Centroides ────────────────────────────────────────────────────────────────

def _get_train_att_dir(task: str) -> Path | None:
    exp_dir = EXPERIMENTS_DIR / f'brca_virchow2_test_{task}_{MIL_KEY}'
    runs    = sorted((exp_dir / 'mil').glob('00001-*none*'))
    return (runs[0] / 'attention') if runs else None


def get_top_patches_tcga(task: str, top_k: int, n_classes: int,
                         slide_info: dict) -> pd.DataFrame:
    att_dir = _get_train_att_dir(task)
    if att_dir is None:
        print(f'  [WARN] No se encontró directorio de atención TCGA para {task}')
        return pd.DataFrame()

    tcga_slides = sorted(
        p.name.replace('_att.npz', '')
        for p in att_dir.glob('TCGA-*_att.npz')
    )
    print(f'  TCGA slides con atención en disco: {len(tcga_slides)}')

    rows = []
    for slide in tcga_slides:
        if slide not in slide_info:
            continue
        cls_id = int(slide_info[slide]['category'])

        idx_file = BAGS_DIR / f'{slide}.index.npz'
        if not idx_file.exists():
            continue
        n_patches = len(np.load(idx_file)['arr_0'])

        att_path = att_dir / f'{slide}_att.npz'
        if not att_path.exists():
            continue
        try:
            att      = load_attention_class(att_path, n_patches, n_classes, cls_id)
            att_norm = minmax_norm(att)
        except Exception as e:
            print(f'  [WARN] {slide}: {e}')
            continue

        k       = min(top_k, n_patches)
        top_idx = np.argsort(att_norm)[-k:][::-1]
        for rank, idx in enumerate(top_idx, start=1):
            rows.append({'slide': slide, 'rank': rank, 'patch_idx': int(idx)})

    return pd.DataFrame(rows)


def compute_centroids(task: str, top_k: int, output_dir: Path,
                      slide_info: dict) -> None:
    import torch

    task_out    = output_dir / task
    patches_csv = task_out / f'top{top_k}_patches.csv'
    if not patches_csv.exists():
        print(f'  [WARN] No existe {patches_csv}. Ejecuta process_task primero.')
        return

    df_cptac  = pd.read_csv(patches_csv)
    n_classes = TASK_N_CLASSES[task]

    df_tcga = get_top_patches_tcga(task, top_k, n_classes, slide_info)
    print(f'  TCGA slides con atención: {df_tcga["slide"].nunique() if not df_tcga.empty else 0}')

    df_cptac = df_cptac[['slide', 'patch_idx']].copy()
    df_cptac['dataset'] = 'cptac'
    if not df_tcga.empty:
        df_tcga = df_tcga[['slide', 'patch_idx']].copy()
        df_tcga['dataset'] = 'tcga'
        df_all = pd.concat([df_cptac, df_tcga], ignore_index=True)
    else:
        df_all = df_cptac

    df_all['category'] = df_all['slide'].map(
        lambda s: slide_info[s]['category'] if s in slide_info else None)
    df_all = df_all.dropna(subset=['category'])
    df_all['category'] = df_all['category'].astype(int)

    embeddings: dict[tuple, list[np.ndarray]] = {}

    n_loaded = 0
    for slide, grp in df_all.groupby('slide'):
        bag_path = BAGS_DIR / f'{slide}.pt'
        if not bag_path.exists():
            continue
        bag    = torch.load(bag_path, map_location='cpu', weights_only=True).numpy()
        ds     = grp['dataset'].iloc[0]
        cls_id = grp['category'].iloc[0]
        for idx in grp['patch_idx'].values:
            if idx < len(bag):
                embeddings.setdefault((cls_id, ds), []).append(bag[idx])
        n_loaded += 1

    print(f'  Slides con bag cargado: {n_loaded}')

    classes = sorted({k[0] for k in embeddings})
    rows = []
    for cls_id in classes:
        key_tcga  = (cls_id, 'tcga')
        key_cptac = (cls_id, 'cptac')
        if key_tcga not in embeddings or key_cptac not in embeddings:
            print(f'  [WARN] Clase {cls_id}: faltan embeddings '
                  f'(tcga={key_tcga in embeddings}, cptac={key_cptac in embeddings})')
            continue

        mu_tcga  = np.stack(embeddings[key_tcga]).mean(axis=0)
        mu_cptac = np.stack(embeddings[key_cptac]).mean(axis=0)
        dist     = float(1.0 - np.dot(mu_tcga, mu_cptac) /
                         (np.linalg.norm(mu_tcga) * np.linalg.norm(mu_cptac)))

        rows.append({
            'task':          task,
            'class':         cls_id,
            'n_tcga':        len(embeddings[key_tcga]),
            'n_cptac':       len(embeddings[key_cptac]),
            'centroid_dist': round(dist, 6),
        })
        print(f'  Clase {cls_id}: n_tcga={len(embeddings[key_tcga])}, '
              f'n_cptac={len(embeddings[key_cptac])}, d={dist:.4f}')

    if rows:
        out = task_out / f'centroid_distances_top{top_k}.csv'
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f'  Guardado: {out}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Top-K patches por atención de clase (CLAM-MB, none norm).')
    parser.add_argument('--top_k', type=int, default=8,
                        help='Nº de patches top a seleccionar por WSI (default: 8)')
    parser.add_argument('--tasks', nargs='+', default=TASKS,
                        choices=TASKS, help='Tareas a procesar')
    parser.add_argument('--output_dir', type=Path,
                        default=Path('/home/jorgarcia/PathBench-MIL/results/patch_intersection_clam'),
                        help='Directorio de salida')
    args = parser.parse_args()

    print(f'Modelo: {MIL_KEY} (atención específica por clase ground-truth)')
    print(f'Top-K : {args.top_k}')
    print(f'Tareas: {args.tasks}')
    print(f'Salida: {args.output_dir}\n')

    for task in args.tasks:
        print(f'=== {task.upper()} ===')
        slide_info = load_slide_info(task)
        process_task(task, args.top_k, args.output_dir, slide_info)
        print(f'  -- Centroides --')
        compute_centroids(task, args.top_k, args.output_dir, slide_info)
        print()


if __name__ == '__main__':
    main()