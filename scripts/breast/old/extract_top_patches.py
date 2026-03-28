#!/usr/bin/env python3
"""
extract_top_patches.py

Para cada tarea y clase molecular, selecciona los N patches más cercanos
al centroide de la clase (distancia coseno) tanto para CPTAC como para TCGA,
y los extrae en formato JPG desde los WSIs originales.

Salida:
  <output_dir>/<task>/centroid_patches/<dataset>/<class_id>/<slide>_pidx<i>_x<x>_y<y>.jpg

Uso:
  python scripts/extract_top_patches.py [--tasks er erbb2 pr pam50]
                                        [--top_k 8]
                                        [--n_closest 25]
                                        [--output_dir results/patch_intersection]
                                        [--tile_px 256]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

# ── Configuración ─────────────────────────────────────────────────────────────

EXPERIMENTS_DIR = Path('/shared/home/jorgarcia/PathBench-MIL/experiments')
BAGS_DIR = Path('/home/PARADIS/mama/tfrecords/shared_bags/256_128_none_virchow2')
WSI_DIRS = [
    Path('/home/PARADIS/mama/wsi/CPTAC-BRCA'),
    Path('/home/PARADIS/mama/wsi/TCGA-BRCA'),
]
ANNOT_DIR = Path('/shared/home/jorgarcia/PathBench-MIL/config/annotations')
ANNOT_FILE = {
    'er':    ANNOT_DIR / 'annotations_brca_er.csv',
    'erbb2': ANNOT_DIR / 'annotations_brca_erbb2.csv',
    'pr':    ANNOT_DIR / 'annotations_brca_pr.csv',
    'pam50': ANNOT_DIR / 'annotations_brca_pam50.csv',
}
MIL_KEYS       = ['transmil', 'dsmil', 'clam_mil_mb']
TASK_N_CLASSES = {'er': 2, 'erbb2': 2, 'pr': 2, 'pam50': 5}

TASKS   = ['er', 'erbb2', 'pr', 'pam50']
TILE_PX  = 256
TILE_UM  = 128   # µm covered per tile (matches tile_um=128 in training config)


# ── Utilidades ────────────────────────────────────────────────────────────────

def find_wsi(slide: str) -> Path | None:
    for wsi_dir in WSI_DIRS:
        matches = list(wsi_dir.glob(f'{slide}.*'))
        if matches:
            return matches[0]
    return None


def cosine_dist_vec(embs: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Distancia coseno de cada fila de embs al vector mu."""
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    mu_n  = mu / (np.linalg.norm(mu) + 1e-12)
    sims  = (embs / (norms + 1e-12)) @ mu_n
    return 1.0 - sims


def load_attention_file(att_path: Path, n_patches: int, n_classes: int) -> np.ndarray:
    arr = np.load(att_path)['arr_0']
    if arr.shape[0] == n_patches * n_classes and n_classes > 1:
        arr = arr.reshape(n_patches, n_classes).mean(axis=1)
    return arr


def minmax_norm(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return np.zeros_like(arr) if hi == lo else (arr - lo) / (hi - lo)


# ── Carga de patches top-K por atención ──────────────────────────────────────

def get_cptac_patch_records(task: str, top_k: int,
                             slide_to_cls: dict) -> list[dict]:
    """Lee top8_patches.csv y devuelve registros con embeddings para slides CPTAC."""
    patches_csv = (Path('/shared/home/jorgarcia/PathBench-MIL/results/patch_intersection')
                   / task / f'top{top_k}_patches.csv')
    if not patches_csv.exists():
        print(f'  [WARN] No existe {patches_csv}')
        return []

    df = pd.read_csv(patches_csv)
    df['category'] = df['slide'].map(slide_to_cls)
    df = df.dropna(subset=['category'])
    df['category'] = df['category'].astype(int)

    records = []
    for slide, grp in df.groupby('slide'):
        bag_path = BAGS_DIR / f'{slide}.pt'
        if not bag_path.exists():
            continue
        bag = torch.load(bag_path, map_location='cpu', weights_only=True).numpy()
        cls_id = grp['category'].iloc[0]
        for _, row in grp.iterrows():
            idx = int(row['patch_idx'])
            if idx < len(bag):
                records.append({'slide': slide, 'patch_idx': idx,
                                'x': int(row['x']), 'y': int(row['y']),
                                'category': cls_id, 'emb': bag[idx]})
    return records


def get_tcga_patch_records(task: str, top_k: int, slide_to_cls: dict) -> list[dict]:
    """Calcula top-K patches TCGA por atención media y devuelve registros con embeddings."""
    n_classes = TASK_N_CLASSES[task]

    # Directorios de atención de entrenamiento
    att_dirs = {}
    for mil in MIL_KEYS:
        exp_dir = EXPERIMENTS_DIR / f'brca_virchow2_test_{task}_{mil}'
        runs    = sorted((exp_dir / 'mil').glob('00001-*none*'))
        if runs:
            att_dirs[mil] = runs[0] / 'attention'

    if not att_dirs:
        print(f'  [WARN] Sin directorios de atención para {task}')
        return []

    all_tcga = set()
    for adir in att_dirs.values():
        all_tcga |= {p.name.replace('_att.npz', '')
                     for p in adir.glob('TCGA-*_att.npz')}

    records = []
    for slide in sorted(all_tcga):
        cls_id = slide_to_cls.get(slide)
        if cls_id is None:
            continue
        idx_file = BAGS_DIR / f'{slide}.index.npz'
        bag_path = BAGS_DIR / f'{slide}.pt'
        if not idx_file.exists() or not bag_path.exists():
            continue

        n_patches = len(np.load(idx_file)['arr_0'])

        att_norm_list = []
        for mil, adir in att_dirs.items():
            att_path = adir / f'{slide}_att.npz'
            if not att_path.exists():
                continue
            try:
                arr = load_attention_file(att_path, n_patches, n_classes)
                att_norm_list.append(minmax_norm(arr))
            except Exception:
                pass

        if not att_norm_list:
            continue

        mean_att = np.stack(att_norm_list).mean(axis=0)
        k        = min(top_k, n_patches)
        top_idx  = np.argsort(mean_att)[-k:]

        bag = torch.load(bag_path, map_location='cpu', weights_only=True).numpy()
        coords = np.load(idx_file)['arr_0']

        for idx in top_idx:
            if idx < len(bag):
                records.append({'slide': slide, 'patch_idx': int(idx),
                                'x': int(coords[idx, 0]), 'y': int(coords[idx, 1]),
                                'category': int(cls_id), 'emb': bag[idx]})
    return records


# ── Extracción de patches JPG ─────────────────────────────────────────────────

def extract_closest_to_centroid(records: list[dict], dataset: str, task: str,
                                 n_closest: int, output_dir: Path,
                                 tile_px: int) -> None:
    import cucim

    classes = sorted(set(r['category'] for r in records))

    for cls_id in classes:
        cls_recs = [r for r in records if r['category'] == cls_id]
        embs = np.stack([r['emb'] for r in cls_recs])
        mu   = embs.mean(axis=0)
        dists = cosine_dist_vec(embs, mu)

        k = min(n_closest, len(dists))
        closest_idx = np.argsort(dists)[:k]

        out_dir = output_dir / task / 'centroid_patches' / dataset / str(cls_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        open_wsis: dict[str, object] = {}
        slide_read_px: dict[str, int] = {}   # level-0 px to read per slide
        n_saved = 0

        for ci in closest_idx:
            rec   = cls_recs[ci]
            slide, x, y, pidx = rec['slide'], rec['x'], rec['y'], rec['patch_idx']
            out_path = out_dir / f'{slide}_pidx{pidx}_x{x}_y{y}.jpg'

            if out_path.exists():
                n_saved += 1
                continue

            if slide not in open_wsis:
                wsi_path = find_wsi(slide)
                if wsi_path is None:
                    open_wsis[slide] = None
                    continue
                try:
                    wsi = cucim.CuImage(str(wsi_path))
                    open_wsis[slide] = wsi
                    # Compute how many level-0 pixels cover TILE_UM µm
                    try:
                        mpp = float(wsi.metadata['aperio']['MPP'])
                    except (KeyError, TypeError, ValueError):
                        mpp = TILE_UM / tile_px   # fallback: assume exact match
                    slide_read_px[slide] = max(tile_px, round(TILE_UM / mpp))
                except Exception as e:
                    print(f'  [WARN] {slide}: {e}')
                    open_wsis[slide] = None
                    continue

            wsi = open_wsis[slide]
            if wsi is None:
                continue

            read_px = slide_read_px.get(slide, tile_px)
            try:
                region = wsi.read_region(location=(x, y), level=0,
                                          size=(read_px, read_px))
                arr = np.asarray(region)
                if arr.ndim == 3 and arr.shape[2] == 4:
                    arr = arr[:, :, :3]
                img = Image.fromarray(arr)
                if read_px != tile_px:
                    img = img.resize((tile_px, tile_px), Image.LANCZOS)
                img.save(out_path, quality=95)
                n_saved += 1
            except Exception as e:
                print(f'  [WARN] {slide} ({x},{y}): {e}')

        for wsi in open_wsis.values():
            if wsi is not None:
                wsi.close()

        print(f'  [{dataset}] Clase {cls_id}: {n_saved}/{k} → {out_dir}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks',      nargs='+', default=TASKS, choices=TASKS)
    parser.add_argument('--top_k',      type=int, default=8)
    parser.add_argument('--n_closest',  type=int, default=25)
    parser.add_argument('--tile_px',    type=int, default=TILE_PX)
    parser.add_argument('--output_dir', type=Path,
                        default=Path('/shared/home/jorgarcia/PathBench-MIL/results/patch_intersection'))
    args = parser.parse_args()

    for task in args.tasks:
        print(f'=== {task.upper()} ===')

        annot      = pd.read_csv(ANNOT_FILE[task])[['slide', 'dataset', 'category']]
        slide_info = annot.set_index('slide').to_dict('index')
        slide_to_cls = {s: info['category'] for s, info in slide_info.items()}

        print('  Cargando CPTAC...')
        cptac_recs = get_cptac_patch_records(task, args.top_k, slide_to_cls)
        print(f'  {len(cptac_recs)} patches CPTAC cargados')
        extract_closest_to_centroid(cptac_recs, 'cptac', task, args.n_closest,
                                    args.output_dir, args.tile_px)

        print('  Cargando TCGA...')
        tcga_recs = get_tcga_patch_records(task, args.top_k, slide_to_cls)
        print(f'  {len(tcga_recs)} patches TCGA cargados')
        extract_closest_to_centroid(tcga_recs, 'tcga', task, args.n_closest,
                                    args.output_dir, args.tile_px)
        print()


if __name__ == '__main__':
    main()