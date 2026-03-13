#!/usr/bin/env python3
"""
patch_intersection.py

Para cada tarea y cada WSI, carga las puntuaciones de atención de los 3 modelos
MIL con normalización 'none' (TransMIL, DSMIL, CLAM-MB), las normaliza a [0,1],
calcula la atención media por patch y selecciona el top-8.

Salida por tarea:
  <output_dir>/<task>/top8_patches.csv
    slide, rank, patch_idx, x, y, mean_att, att_transmil, att_dsmil, att_clam_mil_mb

Uso:
  python scripts/patch_intersection.py [--top_k 8] [--tasks er erbb2 pr pam50]
                                       [--output_dir results/patch_intersection]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ── Configuración de rutas ────────────────────────────────────────────────────

EXPERIMENTS_DIR = Path('/shared/home/jorgarcia/PathBench-MIL/experiments')
BAGS_DIR        = Path('/home/PARADIS/mama/tfrecords/shared_bags/256_128_none_virchow2')

TASKS    = ['er', 'erbb2', 'pr', 'pam50']
MIL_KEYS = ['transmil', 'dsmil', 'clam_mil_mb']

TASK_N_CLASSES = {'er': 2, 'erbb2': 2, 'pr': 2, 'pam50': 5}


# ── Funciones auxiliares ──────────────────────────────────────────────────────

def find_eval_subdir(mil_eval_dir: Path, norm_prefix: str = '00001') -> Path | None:
    candidates = sorted(mil_eval_dir.glob(f'{norm_prefix}-*none*'))
    return candidates[0] if candidates else None


def find_model_subdir(none_eval_dir: Path) -> Path | None:
    candidates = sorted(none_eval_dir.glob('00000-*'))
    return candidates[0] if candidates else None


def load_attention(att_path: Path, n_patches: int, n_classes: int) -> np.ndarray:
    """
    Carga atención y devuelve array (n_patches,).
    CLAM-MB guarda (n_patches * n_classes,) → se reshapea y promedia por clase.
    """
    arr = np.load(att_path)['arr_0']
    if arr.shape[0] == n_patches * n_classes and n_classes > 1:
        arr = arr.reshape(n_patches, n_classes).mean(axis=1)
    elif arr.shape[0] != n_patches:
        raise ValueError(
            f'Shape inesperado: {arr.shape}, esperado ({n_patches},) '
            f'o ({n_patches * n_classes},)'
        )
    return arr


def minmax_norm(arr: np.ndarray) -> np.ndarray:
    """Normaliza a [0, 1]. Si todos los valores son iguales devuelve ceros."""
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


# ── Lógica principal ──────────────────────────────────────────────────────────

def process_task(task: str, top_k: int, output_dir: Path) -> None:
    n_classes = TASK_N_CLASSES[task]
    task_out  = output_dir / task
    task_out.mkdir(parents=True, exist_ok=True)

    # Localizar directorios de atención por modelo
    att_dirs: dict[str, Path] = {}
    for mil in MIL_KEYS:
        exp_dir      = EXPERIMENTS_DIR / f'brca_virchow2_test_{task}_{mil}'
        none_eval    = find_eval_subdir(exp_dir / 'mil_eval')
        if none_eval is None:
            print(f'  [WARN] No eval none para {task}/{mil}')
            continue
        model_dir = find_model_subdir(none_eval)
        if model_dir is None:
            print(f'  [WARN] No model subdir para {task}/{mil}')
            continue
        att_dirs[mil] = model_dir / 'attention'

    if len(att_dirs) < 3:
        print(f'  [ERROR] Faltan directorios de atención para {task}. Saltando.')
        return

    # Slides presentes en los 3 modelos
    slides_per_model = {
        mil: {p.name.replace('_att.npz', '') for p in adir.glob('*_att.npz')}
        for mil, adir in att_dirs.items()
    }
    common_slides = sorted(set.intersection(*slides_per_model.values()))
    print(f'  Slides comunes (3 modelos): {len(common_slides)}')

    rows = []

    for slide in common_slides:
        idx_file = BAGS_DIR / f'{slide}.index.npz'
        if not idx_file.exists():
            print(f'  [WARN] Sin index para {slide}')
            continue
        coords    = np.load(idx_file)['arr_0']   # (n_patches, 2)
        n_patches = len(coords)

        # Cargar y normalizar atención de cada modelo
        att_norm: dict[str, np.ndarray] = {}
        skip = False
        for mil, adir in att_dirs.items():
            att_path = adir / f'{slide}_att.npz'
            if not att_path.exists():
                skip = True
                break
            try:
                att = load_attention(att_path, n_patches, n_classes)
            except ValueError as e:
                print(f'  [WARN] {slide}/{mil}: {e}')
                skip = True
                break
            att_norm[mil] = minmax_norm(att)

        if skip:
            continue

        # Atención media entre los 3 modelos
        mean_att = np.stack(list(att_norm.values())).mean(axis=0)  # (n_patches,)

        # Top-K por atención media
        k        = min(top_k, n_patches)
        top_idx  = np.argsort(mean_att)[-k:][::-1]   # orden descendente

        for rank, idx in enumerate(top_idx, start=1):
            rows.append({
                'task':            task,
                'slide':           slide,
                'rank':            rank,
                'patch_idx':       int(idx),
                'x':               int(coords[idx, 0]),
                'y':               int(coords[idx, 1]),
                'mean_att':        round(float(mean_att[idx]), 6),
                'att_transmil':    round(float(att_norm['transmil'][idx]), 6),
                'att_dsmil':       round(float(att_norm['dsmil'][idx]), 6),
                'att_clam_mil_mb': round(float(att_norm['clam_mil_mb'][idx]), 6),
            })

    if rows:
        df = pd.DataFrame(rows)
        out = task_out / f'top{top_k}_patches.csv'
        df.to_csv(out, index=False)
        print(f'  Guardado: {out}  ({len(df)} filas, {len(common_slides)} slides)')
    else:
        print(f'  Sin datos para {task}.')


ANNOT_DIR  = Path('/shared/home/jorgarcia/PathBench-MIL/config/annotations')
ANNOT_FILE = {
    'er':    ANNOT_DIR / 'annotations_brca_er.csv',
    'erbb2': ANNOT_DIR / 'annotations_brca_erbb2.csv',
    'pr':    ANNOT_DIR / 'annotations_brca_pr.csv',
    'pam50': ANNOT_DIR / 'annotations_brca_pam50.csv',
}


# ── Centroides en el espacio de embeddings ────────────────────────────────────

def _get_train_att_dirs(task: str) -> dict[str, Path]:
    """Devuelve {mil: att_dir} de los runs de entrenamiento (none norm)."""
    att_dirs = {}
    for mil in MIL_KEYS:
        exp_dir = EXPERIMENTS_DIR / f'brca_virchow2_test_{task}_{mil}'
        runs    = sorted((exp_dir / 'mil').glob('00001-*none*'))
        if runs:
            att_dirs[mil] = runs[0] / 'attention'
    return att_dirs


def get_top_patches_tcga(task: str, top_k: int, n_classes: int,
                         slide_info: dict) -> pd.DataFrame:
    """
    Selecciona top-K patches por atención media para TODAS las WSIs TCGA,
    leyendo los ficheros _att.npz generados previamente desde disco.
    """
    att_dirs = _get_train_att_dirs(task)
    if not att_dirs:
        print(f'  [WARN] No se encontraron directorios de atención para {task}')
        return pd.DataFrame()

    # Unión de slides TCGA disponibles en al menos un modelo
    all_tcga: set[str] = set()
    for adir in att_dirs.values():
        all_tcga |= {p.name.replace('_att.npz', '')
                     for p in adir.glob('TCGA-*_att.npz')}
    tcga_slides = sorted(all_tcga)
    print(f'  TCGA slides con atención en disco: {len(tcga_slides)}')

    rows = []
    for slide in tcga_slides:
        idx_file = BAGS_DIR / f'{slide}.index.npz'
        if not idx_file.exists():
            continue
        n_patches = len(np.load(idx_file)['arr_0'])

        att_norm: dict[str, np.ndarray] = {}
        for mil, adir in att_dirs.items():
            att_path = adir / f'{slide}_att.npz'
            if not att_path.exists():
                continue
            try:
                arr = load_attention(att_path, n_patches, n_classes)
                att_norm[mil] = minmax_norm(arr)
            except Exception as e:
                print(f'  [WARN] {slide}/{mil}: {e}')

        if not att_norm:
            continue

        mean_att = np.stack(list(att_norm.values())).mean(axis=0)
        k        = min(top_k, n_patches)
        top_idx  = np.argsort(mean_att)[-k:][::-1]

        for rank, idx in enumerate(top_idx, start=1):
            rows.append({'slide': slide, 'rank': rank, 'patch_idx': int(idx)})

    return pd.DataFrame(rows)


def compute_centroids(task: str, top_k: int, output_dir: Path) -> None:
    import torch

    task_out    = output_dir / task
    patches_csv = task_out / f'top{top_k}_patches.csv'
    if not patches_csv.exists():
        print(f'  [WARN] No existe {patches_csv}. Ejecuta process_task primero.')
        return

    df_cptac = pd.read_csv(patches_csv)   # slides CPTAC

    # Anotaciones: slide → (category, dataset)
    annot      = pd.read_csv(ANNOT_FILE[task])[['slide', 'dataset', 'category']]
    slide_info = annot.set_index('slide')[['dataset', 'category']].to_dict('index')

    # Top-K patches para TCGA (desde carpeta de entrenamiento)
    n_classes  = TASK_N_CLASSES[task]
    df_tcga    = get_top_patches_tcga(task, top_k, n_classes, slide_info)
    print(f'  TCGA slides con atención: {df_tcga["slide"].nunique() if not df_tcga.empty else 0}')

    # Unir ambos DataFrames con columna dataset
    df_cptac = df_cptac[['slide', 'patch_idx']].copy()
    df_cptac['dataset'] = 'cptac'
    if not df_tcga.empty:
        df_tcga = df_tcga[['slide', 'patch_idx']].copy()
        df_tcga['dataset'] = 'tcga'
        df_all = pd.concat([df_cptac, df_tcga], ignore_index=True)
    else:
        df_all = df_cptac

    # Añadir etiqueta de clase desde anotaciones
    df_all['category'] = df_all['slide'].map(
        lambda s: slide_info[s]['category'] if s in slide_info else None)
    df_all = df_all.dropna(subset=['category'])
    df_all['category'] = df_all['category'].astype(int)

    # Acumular embeddings por (clase, dataset)
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

    # Calcular centroides y distancias
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
        dist     = float(1.0 - np.dot(mu_tcga, mu_cptac) / (np.linalg.norm(mu_tcga) * np.linalg.norm(mu_cptac)))

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


def main():
    parser = argparse.ArgumentParser(
        description='Top-K patches por atención media entre 3 modelos MIL (none).')
    parser.add_argument('--top_k', type=int, default=8,
                        help='Nº de patches top a seleccionar por WSI (default: 8)')
    parser.add_argument('--tasks', nargs='+', default=TASKS,
                        choices=TASKS, help='Tareas a procesar')
    parser.add_argument('--output_dir', type=Path,
                        default=Path('/home/jorgarcia/PathBench-MIL/results/patch_intersection'),
                        help='Directorio de salida')
    args = parser.parse_args()

    print(f'Top-K : {args.top_k}')
    print(f'Tareas: {args.tasks}')
    print(f'Salida: {args.output_dir}\n')

    for task in args.tasks:
        print(f'=== {task.upper()} ===')
        process_task(task, args.top_k, args.output_dir)
        print(f'  -- Centroides --')
        compute_centroids(task, args.top_k, args.output_dir)
        print()


if __name__ == '__main__':
    main()