#!/usr/bin/env python3
"""
select_centroid_patches_clam.py

Para cada tarea, cohort (TCGA / CPTAC) y clase molecular:
  1. Carga TODOS los patches de cada slide desde los bags Virchow2 pre-computados
     (sin filtrado por atención).
  2. Calcula el centroide de clase (media de embeddings de todos los patches).
  3. Selecciona los N patches más cercanos al centroide (distancia coseno).
  4. Extrae y guarda los patches como JPG.

Salida:
  <output_dir>/<task>/<dataset>/<class_id>/<dataset>_<slide>_pidx<i>_x<x>_y<y>.jpg

Uso:
  python scripts/select_centroid_patches_clam.py [--n_closest 25]
                                                 [--tasks er erbb2 pr pam50]
                                                 [--output_dir results/centroid_patches_clam]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

# ── Configuración ─────────────────────────────────────────────────────────────

BAGS_DIR   = Path('/home/PARADIS/mama/tfrecords/shared_bags/256_128_none_virchow2')
WSI_DIRS   = [
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

TASKS   = ['er', 'erbb2', 'pr', 'pam50']
TILE_PX = 256
TILE_UM = 128


# ── Utilidades ────────────────────────────────────────────────────────────────

def find_wsi(slide: str) -> Path | None:
    for wsi_dir in WSI_DIRS:
        matches = list(wsi_dir.glob(f'{slide}.*'))
        if matches:
            return matches[0]
    return None


def cosine_dist_vec(embs: np.ndarray, mu: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    mu_n  = mu / (np.linalg.norm(mu) + 1e-12)
    sims  = (embs / (norms + 1e-12)) @ mu_n
    return 1.0 - sims


# ── Carga de patches ──────────────────────────────────────────────────────────

def load_all_patches(slides: list[str], slide_to_cls: dict,
                     dataset: str) -> list[dict]:
    """
    Para cada slide, carga TODOS los patches del bag Virchow2.
    Devuelve lista de registros {slide, patch_idx, x, y, category, dataset, emb}.
    """
    records = []
    for slide in sorted(slides):
        cls_id = slide_to_cls.get(slide)
        if cls_id is None:
            continue
        idx_file = BAGS_DIR / f'{slide}.index.npz'
        bag_path = BAGS_DIR / f'{slide}.pt'
        if not idx_file.exists() or not bag_path.exists():
            continue

        coords = np.load(idx_file)['arr_0']          # (n_patches, 2)
        bag    = torch.load(bag_path, map_location='cpu',
                            weights_only=True).numpy()  # (n_patches, dim)

        n = min(len(coords), len(bag))
        for idx in range(n):
            records.append({
                'slide':     slide,
                'patch_idx': idx,
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

        print(f'  [{dataset}] Clase {cls_id}: {len(cls_recs)} patches en pool, '
              f'{n_saved}/{k} guardados → {out_dir}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Selecciona los N patches más cercanos al centroide de clase '
                    '(todos los patches del bag, TCGA y CPTAC por separado).')
    parser.add_argument('--n_closest',  type=int,  default=25,
                        help='Patches a seleccionar por clase y cohort (default: 25)')
    parser.add_argument('--tasks',      nargs='+', default=TASKS, choices=TASKS)
    parser.add_argument('--tile_px',    type=int,  default=TILE_PX)
    parser.add_argument('--output_dir', type=Path,
                        default=Path('/shared/home/jorgarcia/PathBench-MIL/results/centroid_patches_clam'))
    args = parser.parse_args()

    print(f'N closest: {args.n_closest} patches/clase/cohort')
    print(f'Salida   : {args.output_dir}\n')

    for task in args.tasks:
        print(f'=== {task.upper()} ===')
        annot        = pd.read_csv(ANNOT_FILE[task])[['slide', 'dataset', 'category']]
        slide_to_cls = dict(zip(annot['slide'], annot['category']))

        tcga_slides  = annot[annot['dataset'] == 'tcga']['slide'].tolist()
        cptac_slides = annot[annot['dataset'] == 'cptac']['slide'].tolist()

        print(f'  TCGA slides: {len(tcga_slides)}, CPTAC slides: {len(cptac_slides)}')

        print('  Cargando patches TCGA...')
        tcga_recs = load_all_patches(tcga_slides, slide_to_cls, 'tcga')
        print(f'  {len(tcga_recs):,} patches TCGA cargados')
        select_and_extract(tcga_recs, task, 'tcga', args.n_closest,
                           args.output_dir, args.tile_px)
        del tcga_recs

        print('  Cargando patches CPTAC...')
        cptac_recs = load_all_patches(cptac_slides, slide_to_cls, 'cptac')
        print(f'  {len(cptac_recs):,} patches CPTAC cargados')
        select_and_extract(cptac_recs, task, 'cptac', args.n_closest,
                           args.output_dir, args.tile_px)
        del cptac_recs

        print()


if __name__ == '__main__':
    main()