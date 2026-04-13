#!/usr/bin/env python3
"""
extract_attention_patches.py

Para cada slide de macarena:
  1. Carga el modelo MSS/MSI (dsmil/none)
  2. Calcula atención por patch
  3. Extrae los top-K patches más atentos de la WSI
  4. Guarda las imágenes en results/colon_mss_msi/top_patches/{class}/{slide}/

Además:
  - Guarda top8_patches.csv con coordenadas y scores de atención
  - Calcula centroides por clase (media de features de top-K patches)
  - Guarda centroids.csv

Uso:
  python scripts/colon/extract_attention_patches.py [--top_k 8] [--force]
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tifffile
import zarr
from PIL import Image

# ── Rutas ─────────────────────────────────────────────────────────────────────

ROOT         = Path(__file__).resolve().parents[2]
MODEL_DIR    = ROOT / 'experiments/colon_mss_msi_benchmark/mil/00000-256_128_none_h_optimus_0_dsmil_CrossEntropyLoss_ReLU_Adam_1'
BAGS_DIR     = Path('/home/PARADIS/colon/tfrecords/shared_bags/256_128_none_h_optimus_0')

WSI_DIRS = {
    'macarena': Path('/home/PARADIS/colon/wsi/macarena/todos'),
    'tcga_coad': Path('/home/PARADIS/colon/wsi/TCGA-COAD'),
    'cptac_coad': Path('/home/PARADIS/colon/wsi/CPTAC-COAD_v1/COAD'),
}
WSI_EXT = {
    'macarena': '.tiff',
    'tcga_coad': '.svs',
    'cptac_coad': '.svs',
}
ANNOT_FILES = {
    'macarena':   ROOT / 'config/annotations/annotations_colon_mss_msi.csv',
    'tcga_coad':  ROOT / 'config/annotations/annotations_tcga_coad_mss_msi.csv',
    'cptac_coad': ROOT / 'config/annotations/annotations_cptac_coad_mss_msi.csv',
}

OUT_DIR      = ROOT / 'results/colon'
PATCH_DIR    = OUT_DIR / 'top_patches'

TILE_PX      = 512   # 40x: same physical area (128µm) as H-Optimus-0 input but at full resolution
CONTEXT_PX   = 1536  # 3×TILE_PX: 3×3 grid centered on the patch
CLASS_NAMES  = {0: 'MSS', 1: 'MSI'}

_SF_MIL = {
    'transmil': str(ROOT / 'slideflow_fork/slideflow/mil/models/transmil.py'),
}


# ── Carga de modelo ───────────────────────────────────────────────────────────

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _get_model_cls(model_name: str):
    if model_name.lower() in _SF_MIL:
        mod = _load_module(f'sf_{model_name}', _SF_MIL[model_name.lower()])
        return getattr(mod, 'TransMIL')
    agg = _load_module('aggregators', str(ROOT / 'pathbench/models/aggregators.py'))
    return getattr(agg, model_name)


def load_model(model_dir: Path, device: torch.device):
    with open(model_dir / 'mil_params.json') as f:
        mp = json.load(f)
    params = mp['params']
    cls    = _get_model_cls(params['model'])
    model  = cls(
        n_feats=mp['input_shape'],
        n_out=mp['output_shape'],
        z_dim=params.get('z_dim', 256),
        dropout_p=params.get('dropout_p', 0.1),
        activation_function=params.get('activation_function', 'ReLU'),
        encoder_layers=params.get('encoder_layers', 1),
    )
    ckpts = sorted(model_dir.glob('checkpoints/**/best-epoch*.ckpt'))
    if not ckpts:
        raise FileNotFoundError(f'Sin checkpoint en {model_dir}')
    ckpt  = torch.load(ckpts[-1], map_location='cpu')
    state = {k.removeprefix('model.'): v
             for k, v in ckpt['state_dict'].items() if k.startswith('model.')}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), mp['output_shape']


# ── Cómputo de atención ───────────────────────────────────────────────────────

def compute_attention(model, features: torch.Tensor, n_classes: int) -> np.ndarray:
    with torch.no_grad():
        inp = features.unsqueeze(0)
        try:
            att = model.calculate_attention(inp)
        except TypeError:
            att = model.calculate_attention(inp, apply_softmax=False)
    att = torch.squeeze(att)
    if att.dim() == 2:
        att = att.mean(dim=-1)
    arr = att.cpu().float().numpy()
    n_patches = features.shape[0]
    if arr.ndim == 1 and arr.shape[0] == n_patches * n_classes and n_classes > 1:
        arr = arr.reshape(n_patches, n_classes).mean(axis=1)
    return arr.astype(np.float32)


# ── Extracción de patch de WSI ────────────────────────────────────────────────

def extract_patch(wsi_path: Path, x: int, y: int, tile_px: int = 512,
                  context_px: int = 1536) -> Image.Image | None:
    """
    Extrae una región context_px×context_px centrada en el patch (x, y).
    El patch seleccionado (tile_px×tile_px) queda en el centro.
    Si la región sale fuera de la WSI, devuelve None.
    """
    offset = (context_px - tile_px) // 2  # = tile_px para 3×3
    x0 = x - offset
    y0 = y - offset
    try:
        tif = tifffile.TiffFile(str(wsi_path))
        store = tif.aszarr(level=0)
        z = zarr.open(store, mode='r')
        h, w = z.shape[0], z.shape[1]
        if x0 < 0 or y0 < 0 or x0 + context_px > w or y0 + context_px > h:
            return None
        region = z[y0:y0 + context_px, x0:x0 + context_px]
        if region.shape[0] < context_px or region.shape[1] < context_px:
            return None
        return Image.fromarray(region.astype(np.uint8))
    except Exception as e:
        print(f'    [WARN] No se pudo extraer contexto ({x},{y}) de {wsi_path.name}: {e}')
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_k', type=int, default=8)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--dataset', type=str, default='macarena',
                        choices=['macarena', 'tcga_coad', 'cptac_coad'])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Dispositivo: {device}  Dataset: {args.dataset}')

    wsi_dir = WSI_DIRS[args.dataset]
    wsi_ext = WSI_EXT[args.dataset]

    # Anotaciones
    annot = pd.read_csv(ANNOT_FILES[args.dataset])
    slide_label = dict(zip(annot['slide'], annot['category']))
    print(f'Slides {args.dataset}: {len(slide_label)}  (MSI={sum(v==1 for v in slide_label.values())}, MSS={sum(v==0 for v in slide_label.values())})')

    # Cargar modelo
    model, n_out = load_model(MODEL_DIR, device)
    print(f'Modelo cargado: {MODEL_DIR.name}')

    # Crear dirs de salida
    patch_dir = PATCH_DIR / args.dataset
    for cls_name in CLASS_NAMES.values():
        (patch_dir / cls_name).mkdir(parents=True, exist_ok=True)

    records = []
    centroid_feats = {0: [], 1: []}

    slides = sorted(slide_label.keys())
    for i, slide in enumerate(slides, 1):
        cls_id   = int(slide_label[slide])
        cls_name = CLASS_NAMES[cls_id]

        bag_path = BAGS_DIR / f'{slide}.pt'
        idx_path = BAGS_DIR / f'{slide}.index.npz'
        wsi_path = wsi_dir / f'{slide}{wsi_ext}'

        if not bag_path.exists():
            print(f'  [SKIP] {slide}: sin bag')
            continue
        if not idx_path.exists():
            print(f'  [SKIP] {slide}: sin index')
            continue
        if not wsi_path.exists():
            print(f'  [SKIP] {slide}: sin WSI')
            continue

        # Features y coordenadas
        features = torch.load(bag_path, map_location=device,
                              weights_only=True).float()
        coords   = np.load(idx_path)['arr_0']  # (n_patches, 2) — (x, y)

        # Atención
        att = compute_attention(model, features, n_out)

        # Top-K
        top_idx = np.argsort(att)[::-1][:args.top_k].copy()

        # Centroide: acumular features de top-K
        top_feats = features[top_idx].cpu().numpy()
        centroid_feats[cls_id].append(top_feats)

        # Extraer imágenes
        slide_patch_dir = patch_dir / cls_name / slide
        if not args.force and slide_patch_dir.exists():
            pass  # ya procesado, igualmente guardamos records
        else:
            slide_patch_dir.mkdir(parents=True, exist_ok=True)
            for rank, pidx in enumerate(top_idx):
                x, y = int(coords[pidx, 0]), int(coords[pidx, 1])
                img = extract_patch(wsi_path, x, y, TILE_PX, CONTEXT_PX)
                if img is not None:
                    img.save(slide_patch_dir / f'rank{rank+1:02d}_att{att[pidx]:.4f}.png')

        # Registrar
        for rank, pidx in enumerate(top_idx):
            x, y = int(coords[pidx, 0]), int(coords[pidx, 1])
            records.append({
                'slide': slide, 'class': cls_name, 'class_id': cls_id,
                'rank': rank + 1, 'x': x, 'y': y,
                'attention': float(att[pidx]),
            })

        if i % 20 == 0 or i == len(slides):
            print(f'  [{i}/{len(slides)}] {slide} ({cls_name})')

    # Guardar top8_patches.csv
    df_rec = pd.DataFrame(records)
    df_rec.to_csv(OUT_DIR / f'top8_patches_{args.dataset}.csv', index=False)
    print(f'\nGuardado: top8_patches.csv  ({len(df_rec)} filas)')


    # Calcular y guardar centroides
    centroids = []
    for cls_id, cls_name in CLASS_NAMES.items():
        if centroid_feats[cls_id]:
            all_feats = np.vstack(centroid_feats[cls_id])  # (N*top_k, 1536)
            centroid  = all_feats.mean(axis=0)             # (1536,)
            row = {'class': cls_name, 'class_id': cls_id}
            for j, v in enumerate(centroid):
                row[f'feat_{j}'] = float(v)
            centroids.append(row)
            print(f'Centroide {cls_name}: {len(all_feats)} patches de {len(centroid_feats[cls_id])} slides')

    df_cent = pd.DataFrame(centroids)
    df_cent.to_csv(OUT_DIR / f'centroids_{args.dataset}.csv', index=False)
    print(f'Guardado: centroids.csv')

    # Distancia coseno entre centroides
    if len(centroids) == 2:
        c0 = np.array([centroids[0][f'feat_{j}'] for j in range(1536)])
        c1 = np.array([centroids[1][f'feat_{j}'] for j in range(1536)])
        cos_sim = np.dot(c0, c1) / (np.linalg.norm(c0) * np.linalg.norm(c1))
        print(f'\nDistancia coseno MSS-MSI: {1 - cos_sim:.4f}  (similitud: {cos_sim:.4f})')

if __name__ == '__main__':
    main()