#!/usr/bin/env python3
"""
patch_intersection_all.py

Calcula d_c (distancia coseno entre centroides TCGA/CPTAC en espacio Virchow2)
usando los top-K patches por atención de los 3 modelos Optuna:
  - CLAM-MB: atención específica por clase ground-truth (columna cls_id)
  - DSMIL:   atención genérica (un único vector)
  - TransMIL: atención genérica (un único vector)

d_c final = media de las 3 distancias individuales.

Salida por tarea en <output_dir>/<task>/:
  centroid_distances_all.csv
    task, class, d_clam, d_dsmil, d_transmil, d_mean

Uso:
  python scripts/patch_intersection_all.py [--top_k 8] [--tasks er erbb2 pr pam50]
                                           [--output_dir results/patch_intersection_all]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ── Configuración ─────────────────────────────────────────────────────────────

EXPERIMENTS_DIR = Path('/shared/home/jorgarcia/PathBench-MIL/experiments')
BAGS_DIR        = Path('/home/PARADIS/mama/tfrecords/shared_bags/256_128_none_virchow2')
ANNOT_DIR       = Path('/shared/home/jorgarcia/PathBench-MIL/config/annotations')

TASKS          = ['er', 'erbb2', 'pr', 'pam50']
MILS           = ['clam_mil_mb', 'dsmil', 'transmil']
TASK_N_CLASSES = {'er': 2, 'erbb2': 2, 'pr': 2, 'pam50': 5}

ANNOT_FILE = {
    'er':    ANNOT_DIR / 'annotations_brca_er.csv',
    'erbb2': ANNOT_DIR / 'annotations_brca_erbb2.csv',
    'pr':    ANNOT_DIR / 'annotations_brca_pr.csv',
    'pam50': ANNOT_DIR / 'annotations_brca_pam50.csv',
}


# ── Utilidades ────────────────────────────────────────────────────────────────

def load_slide_info(task: str) -> dict:
    annot = pd.read_csv(ANNOT_FILE[task])[['slide', 'dataset', 'category']]
    return annot.set_index('slide')[['dataset', 'category']].to_dict('index')


def minmax_norm(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def load_attention(att_path: Path, n_patches: int, n_classes: int,
                   cls_id: int, mil: str) -> np.ndarray:
    """
    Carga atención y devuelve vector (n_patches,).
    - CLAM-MB: reshape (n_patches, n_classes) y selecciona columna cls_id.
    - DSMIL / TransMIL: vector único (n_patches,).
    """
    arr = np.load(att_path)['arr_0']

    if mil == 'clam_mil_mb':
        if arr.shape[0] == n_patches * n_classes and n_classes > 1:
            return arr.reshape(n_patches, n_classes)[:, cls_id]
        elif arr.shape[0] == n_patches:
            return arr
        else:
            raise ValueError(f'CLAM-MB shape inesperado: {arr.shape}')
    else:
        # DSMIL / TransMIL: un solo vector
        if arr.shape[0] == n_patches:
            return arr
        else:
            raise ValueError(f'{mil} shape inesperado: {arr.shape}')


def find_none_eval_dir(task: str, mil: str) -> Path | None:
    """Directorio de evaluación con norm=none para CPTAC (test)."""
    exp_dir  = EXPERIMENTS_DIR / f'brca_virchow2_test_{task}_{mil}'
    mil_eval = exp_dir / 'mil_eval'
    candidates = sorted(mil_eval.glob('00001-*none*'))
    if not candidates:
        return None
    model_dirs = sorted(candidates[0].glob('00000-*'))
    return model_dirs[0] if model_dirs else None


def find_train_att_dir(task: str, mil: str) -> Path | None:
    """Directorio de atención TCGA (train/val)."""
    exp_dir = EXPERIMENTS_DIR / f'brca_virchow2_test_{task}_{mil}'
    runs    = sorted((exp_dir / 'mil').glob('00001-*none*'))
    return (runs[0] / 'attention') if runs else None


# ── Selección de top-K patches ────────────────────────────────────────────────

def get_top_patches(att_dir: Path, mil: str, task: str, top_k: int,
                    slide_info: dict, dataset_filter: str | None = None
                    ) -> dict[tuple[int, str], list[int]]:
    """
    Devuelve {(cls_id, slide): [patch_idx, ...]} con los top-K patches
    por atención para cada slide en att_dir.

    dataset_filter: 'tcga' | 'cptac' | None (todos)
    """
    n_classes = TASK_N_CLASSES[task]
    result: dict[tuple[int, str], list[int]] = {}

    att_files = sorted(att_dir.glob('*_att.npz'))
    for att_path in att_files:
        slide = att_path.name.replace('_att.npz', '')
        if slide not in slide_info:
            continue
        info = slide_info[slide]
        if dataset_filter and info['dataset'] != dataset_filter:
            continue

        cls_id    = int(info['category'])
        idx_file  = BAGS_DIR / f'{slide}.index.npz'
        if not idx_file.exists():
            continue
        n_patches = len(np.load(idx_file)['arr_0'])

        try:
            att      = load_attention(att_path, n_patches, n_classes, cls_id, mil)
            att_norm = minmax_norm(att)
        except Exception as e:
            print(f'    [WARN] {slide}: {e}')
            continue

        k       = min(top_k, n_patches)
        top_idx = np.argsort(att_norm)[-k:][::-1].tolist()
        result[(cls_id, slide)] = top_idx

    return result


# ── Cálculo de centroides y distancia coseno ──────────────────────────────────

def centroid_distance(top_patches_tcga: dict, top_patches_cptac: dict,
                      n_classes: int) -> dict[int, float]:
    """
    Para cada clase, apila los embeddings de todos los patches seleccionados
    (TCGA y CPTAC por separado), calcula el centroide y devuelve la distancia
    coseno entre ambos centroides.
    """
    embs: dict[tuple[int, str], list[np.ndarray]] = {}

    for (cls_id, slide), idxs in top_patches_tcga.items():
        bag_path = BAGS_DIR / f'{slide}.pt'
        if not bag_path.exists():
            continue
        bag = torch.load(bag_path, map_location='cpu', weights_only=True).numpy()
        for idx in idxs:
            if idx < len(bag):
                embs.setdefault((cls_id, 'tcga'), []).append(bag[idx])

    for (cls_id, slide), idxs in top_patches_cptac.items():
        bag_path = BAGS_DIR / f'{slide}.pt'
        if not bag_path.exists():
            continue
        bag = torch.load(bag_path, map_location='cpu', weights_only=True).numpy()
        for idx in idxs:
            if idx < len(bag):
                embs.setdefault((cls_id, 'cptac'), []).append(bag[idx])

    distances: dict[int, float] = {}
    for cls_id in range(n_classes):
        key_t = (cls_id, 'tcga')
        key_c = (cls_id, 'cptac')
        if key_t not in embs or key_c not in embs:
            print(f'    [WARN] Clase {cls_id}: faltan embeddings '
                  f'(tcga={key_t in embs}, cptac={key_c in embs})')
            continue
        mu_t = np.stack(embs[key_t]).mean(axis=0)
        mu_c = np.stack(embs[key_c]).mean(axis=0)
        d    = float(1.0 - np.dot(mu_t, mu_c) /
                     (np.linalg.norm(mu_t) * np.linalg.norm(mu_c)))
        distances[cls_id] = d
        print(f'    Clase {cls_id}: n_tcga={len(embs[key_t])}, '
              f'n_cptac={len(embs[key_c])}, d={d:.4f}')

    return distances


# ── Procesamiento por tarea ───────────────────────────────────────────────────

def process_task(task: str, top_k: int, output_dir: Path,
                 slide_info: dict) -> None:
    n_classes = TASK_N_CLASSES[task]
    task_out  = output_dir / task
    task_out.mkdir(parents=True, exist_ok=True)

    all_dists: dict[str, dict[int, float]] = {}

    for mil in MILS:
        print(f'  [{mil}]')
        none_dir  = find_none_eval_dir(task, mil)
        train_dir = find_train_att_dir(task, mil)

        if none_dir is None:
            print(f'    [ERROR] No eval none para {task}/{mil}')
            continue
        if train_dir is None:
            print(f'    [ERROR] No train att para {task}/{mil}')
            continue

        cptac_att_dir = none_dir / 'attention'
        tcga_att_dir  = train_dir

        top_cptac = get_top_patches(cptac_att_dir, mil, task, top_k,
                                    slide_info, dataset_filter='cptac')
        top_tcga  = get_top_patches(tcga_att_dir,  mil, task, top_k,
                                    slide_info, dataset_filter='tcga')

        print(f'    CPTAC slides: {len({s for (_, s) in top_cptac})}  '
              f'TCGA slides: {len({s for (_, s) in top_tcga})}')

        dists = centroid_distance(top_tcga, top_cptac, n_classes)
        all_dists[mil] = dists

    # ── Construir CSV con las 3 distancias + media ─────────────────────────
    rows = []
    for cls_id in range(n_classes):
        d_clam    = all_dists.get('clam_mil_mb', {}).get(cls_id, float('nan'))
        d_dsmil   = all_dists.get('dsmil',       {}).get(cls_id, float('nan'))
        d_trans   = all_dists.get('transmil',     {}).get(cls_id, float('nan'))
        vals      = [v for v in [d_clam, d_dsmil, d_trans] if not np.isnan(v)]
        d_mean    = float(np.mean(vals)) if vals else float('nan')
        rows.append({
            'task':      task,
            'class':     cls_id,
            'd_clam':    round(d_clam,  6),
            'd_dsmil':   round(d_dsmil, 6),
            'd_transmil': round(d_trans, 6),
            'd_mean':    round(d_mean,  6),
        })

    if rows:
        out = task_out / f'centroid_distances_all.csv'
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f'  Guardado: {out}')
        print(pd.DataFrame(rows).to_string(index=False))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='d_c como media de los 3 modelos Optuna.')
    parser.add_argument('--top_k', type=int, default=8)
    parser.add_argument('--tasks', nargs='+', default=TASKS, choices=TASKS)
    parser.add_argument('--output_dir', type=Path,
                        default=Path('/shared/home/jorgarcia/PathBench-MIL/'
                                     'results/patch_intersection_all'))
    args = parser.parse_args()

    print(f'Top-K : {args.top_k}')
    print(f'Tareas: {args.tasks}')
    print(f'Salida: {args.output_dir}\n')

    for task in args.tasks:
        print(f'=== {task.upper()} ===')
        slide_info = load_slide_info(task)
        process_task(task, args.top_k, args.output_dir, slide_info)
        print()


if __name__ == '__main__':
    main()