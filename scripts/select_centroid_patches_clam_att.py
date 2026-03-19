#!/usr/bin/env python3
"""
select_centroid_patches_clam_att.py

Variante de select_centroid_patches_clam.py que usa atención clase-específica
de CLAM-MB para reducir el pool inicial a top-K patches por WSI antes de
calcular el centroide de clase.

Pipeline por tarea, cohort y clase:
  1. Para cada WSI de la clase: selecciona top-K patches por atención
     clase-específica de CLAM-MB (columna ground-truth).
  2. Carga embeddings Virchow2 de esos top-K patches desde los bags.
  3. Calcula el centroide de clase sobre el pool top-K de todos los WSIs.
  4. Selecciona los N patches más cercanos al centroide (distancia coseno).
  5. Extrae y guarda los patches como JPG.

TCGA y CPTAC se procesan por separado.

Salida:
  <output_dir>/<task>/<dataset>/<class_id>/<dataset>_<slide>_pidx<i>_x<x>_y<y>.jpg

Uso:
  python scripts/select_centroid_patches_clam_att.py [--top_k 8] [--n_closest 25]
                                                     [--tasks er erbb2 pr pam50]
                                                     [--output_dir results/centroid_patches_clam_att]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

# ── Configuración ─────────────────────────────────────────────────────────────

EXPERIMENTS_DIR = Path('/shared/home/jorgarcia/PathBench-MIL/experiments')
BAGS_DIR        = Path('/home/PARADIS/mama/tfrecords/shared_bags/256_128_none_virchow2')
WSI_DIRS        = [
    Path('/home/PARADIS/mama/wsi/CPTAC-BRCA'),
    Path('/home/PARADIS/mama/wsi/TCGA-BRCA'),
]
ANNOT_DIR  = Path('/shared/home/jorgarcia/PathBench-MIL/config/annotations')
ANNOT_FILE = {
    'er':    ANNOT_DIR / 'annotations_brca_er.csv',
    'erbb2': ANNOT_DIR / 'annotations_brca_erbb2.csv',
    'pr':    ANNOT_DIR / 'annotations_brca_pr.csv',
    'pam50': ANNOT_DIR / 'annotations_brca_pam50.csv',
}

MIL_KEY        = 'clam_mil_mb'
TASKS          = ['er', 'erbb2', 'pr', 'pam50']
TASK_N_CLASSES = {'er': 2, 'erbb2': 2, 'pr': 2, 'pam50': 5}
TILE_PX        = 256
TILE_UM        = 128


# ── Utilidades ────────────────────────────────────────────────────────────────

def find_wsi(slide: str) -> Path | None:
    for wsi_dir in WSI_DIRS:
        matches = list(wsi_dir.glob(f'{slide}.*'))
        if matches:
            return matches[0]
    return None


def minmax_norm(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return np.zeros_like(arr) if hi == lo else (arr - lo) / (hi - lo)


def load_attention_class(att_path: Path, n_patches: int, n_classes: int,
                         cls_id: int) -> np.ndarray:
    arr = np.load(att_path)['arr_0']
    if arr.shape[0] == n_patches * n_classes and n_classes > 1:
        return arr.reshape(n_patches, n_classes)[:, cls_id]
    elif arr.shape[0] == n_patches:
        return arr
    raise ValueError(f'Shape inesperado: {arr.shape}')


def cosine_dist_vec(embs: np.ndarray, mu: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    mu_n  = mu / (np.linalg.norm(mu) + 1e-12)
    sims  = (embs / (norms + 1e-12)) @ mu_n
    return 1.0 - sims


# ── Directorios de atención ───────────────────────────────────────────────────

def get_cptac_att_dir(task: str) -> Path | None:
    exp_dir = EXPERIMENTS_DIR / f'brca_virchow2_test_{task}_{MIL_KEY}'
    candidates = sorted((exp_dir / 'mil_eval').glob('00001-*none*'))
    if not candidates:
        return None
    model_dirs = sorted(candidates[0].glob('00000-*'))
    return (model_dirs[0] / 'attention') if model_dirs else None


def get_tcga_att_dir(task: str) -> Path | None:
    exp_dir = EXPERIMENTS_DIR / f'brca_virchow2_test_{task}_{MIL_KEY}'
    runs    = sorted((exp_dir / 'mil').glob('00001-*none*'))
    return (runs[0] / 'attention') if runs else None


# ── Carga top-K patches por atención ─────────────────────────────────────────

def load_topk_patches(slides: list[str], slide_to_cls: dict, att_dir: Path,
                      dataset: str, top_k: int, n_classes: int) -> list[dict]:
    """
    Para cada slide, selecciona top-K patches por atención clase-específica
    y carga sus embeddings Virchow2.
    """
    records = []
    for slide in sorted(slides):
        cls_id = slide_to_cls.get(slide)
        if cls_id is None:
            continue

        att_path = att_dir / f'{slide}_att.npz'
        idx_file = BAGS_DIR / f'{slide}.index.npz'
        bag_path = BAGS_DIR / f'{slide}.pt'
        if not att_path.exists() or not idx_file.exists() or not bag_path.exists():
            continue

        coords    = np.load(idx_file)['arr_0']
        n_patches = len(coords)

        try:
            att      = load_attention_class(att_path, n_patches, n_classes, int(cls_id))
            att_norm = minmax_norm(att)
        except Exception as e:
            print(f'  [WARN] {slide}: {e}')
            continue

        bag     = torch.load(bag_path, map_location='cpu', weights_only=True).numpy()
        top_idx = np.argsort(att_norm)[-min(top_k, n_patches):]

        for idx in top_idx:
            if idx < len(bag):
                records.append({
                    'slide':     slide,
                    'patch_idx': int(idx),
                    'x':         int(coords[idx, 0]),
                    'y':         int(coords[idx, 1]),
                    'category':  int(cls_id),
                    'dataset':   dataset,
                    'emb':       bag[idx],
                })
    return records


# ── Selección y extracción ────────────────────────────────────────────────────

def select_and_extract(records: list[dict], task: str, dataset: str,
                       n_closest: int, output_dir: Path, tile_px: int) -> None:
    import cucim

    classes = sorted(set(r['category'] for r in records))

    for cls_id in classes:
        cls_recs = [r for r in records if r['category'] == cls_id]
        if not cls_recs:
            continue

        embs  = np.stack([r['emb'] for r in cls_recs])
        mu    = embs.mean(axis=0)
        dists = cosine_dist_vec(embs, mu)

        k           = min(n_closest, len(dists))
        closest_idx = np.argsort(dists)[:k]

        out_dir = output_dir / task / dataset / str(cls_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        open_wsis:     dict[str, object] = {}
        slide_read_px: dict[str, int]    = {}
        n_saved = 0

        for ci in closest_idx:
            rec   = cls_recs[ci]
            slide = rec['slide']
            x, y, pidx = rec['x'], rec['y'], rec['patch_idx']
            out_path = out_dir / f'{dataset}_{slide}_pidx{pidx}_x{x}_y{y}.jpg'

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
                    try:
                        mpp = float(wsi.metadata['aperio']['MPP'])
                    except (KeyError, TypeError, ValueError):
                        mpp = TILE_UM / tile_px
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

        print(f'  [{dataset}] Clase {cls_id}: pool={len(cls_recs)}, '
              f'{n_saved}/{k} guardados → {out_dir}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Top-K por atención CLAM-MB → centroide de clase → 25 más cercanos. '
                    'TCGA y CPTAC por separado.')
    parser.add_argument('--top_k',      type=int,  default=8,
                        help='Top-K patches por atención por WSI (default: 8)')
    parser.add_argument('--n_closest',  type=int,  default=25,
                        help='Patches a seleccionar por clase y cohort (default: 25)')
    parser.add_argument('--tasks',      nargs='+', default=TASKS, choices=TASKS)
    parser.add_argument('--tile_px',    type=int,  default=TILE_PX)
    parser.add_argument('--output_dir', type=Path,
                        default=Path('/shared/home/jorgarcia/PathBench-MIL/results/centroid_patches_clam_att'))
    args = parser.parse_args()

    print(f'Top-K atención : {args.top_k} patches/WSI')
    print(f'N closest      : {args.n_closest} patches/clase/cohort')
    print(f'Salida         : {args.output_dir}\n')

    for task in args.tasks:
        print(f'=== {task.upper()} ===')
        annot        = pd.read_csv(ANNOT_FILE[task])[['slide', 'dataset', 'category']]
        slide_to_cls = dict(zip(annot['slide'], annot['category']))
        n_classes    = TASK_N_CLASSES[task]

        tcga_slides  = annot[annot['dataset'] == 'tcga']['slide'].tolist()
        cptac_slides = annot[annot['dataset'] == 'cptac']['slide'].tolist()

        tcga_att_dir  = get_tcga_att_dir(task)
        cptac_att_dir = get_cptac_att_dir(task)

        if tcga_att_dir is None:
            print(f'  [WARN] Sin directorio de atención TCGA para {task}')
        else:
            print(f'  Cargando top-{args.top_k} patches TCGA ({len(tcga_slides)} slides)...')
            tcga_recs = load_topk_patches(tcga_slides, slide_to_cls, tcga_att_dir,
                                          'tcga', args.top_k, n_classes)
            print(f'  {len(tcga_recs):,} patches TCGA en pool')
            select_and_extract(tcga_recs, task, 'tcga', args.n_closest,
                               args.output_dir, args.tile_px)
            del tcga_recs

        if cptac_att_dir is None:
            print(f'  [WARN] Sin directorio de atención CPTAC para {task}')
        else:
            print(f'  Cargando top-{args.top_k} patches CPTAC ({len(cptac_slides)} slides)...')
            cptac_recs = load_topk_patches(cptac_slides, slide_to_cls, cptac_att_dir,
                                           'cptac', args.top_k, n_classes)
            print(f'  {len(cptac_recs):,} patches CPTAC en pool')
            select_and_extract(cptac_recs, task, 'cptac', args.n_closest,
                               args.output_dir, args.tile_px)
            del cptac_recs

        print()


if __name__ == '__main__':
    main()