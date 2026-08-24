#!/usr/bin/env python3
"""
02_select_patches.py
--------------------
Paso 2 del bundle: dado el DSMIL óptimo y los bags de features ya extraídos,
reproduce la selección de patches que se mostraron a los patólogos.

Procedimiento (idéntico al del estudio, pero SIN etiquetas de entrada):
  1. Para cada slide:
       - se calcula la ATENCIÓN del DSMIL por patch,
       - se toman los top-K patches más atendidos,
       - se guarda la imagen de cada uno (región 3x3 centrada en el patch),
       - se calcula la distancia COSENO de su feature a los centroides
         MSS y MSI de referencia (Macarena, incluidos en el bundle).
  2. De todos los top-K patches de todos los slides:
       - los 100 más cercanos al centroide MSS  -> carpeta MSS/
       - los 100 más cercanos al centroide MSI  -> carpeta MSI/
  3. Se escribe un manifest CSV con slide, coordenadas, atención y distancias.

Uso:
  python 02_select_patches.py \
      --bags_dir  <WORK_DIR>/bags/256_128_none_h_optimus_0 \
      --wsi_dir   <carpeta_wsis> \
      --out_dir   <carpeta_salida> \
      [--top_k 8] [--top_n 100]

El modelo y los centroides se cargan de las carpetas model/ y reference/ del bundle.
"""
import os, json, argparse, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
import openslide

BUNDLE = Path(__file__).resolve().parents[1]
# Repo PathBench (para importar la definición del modelo DSMIL). Configurable.
REPO = Path(os.environ.get('PATHBENCH_REPO', BUNDLE.parent))

TILE_PX    = 512    # tamaño del patch a resolución nativa (256px@20x == 512px@40x)
CONTEXT_PX = 1536   # 3x -> rejilla 3x3 centrada en el patch (contexto para el patólogo)
CLASS_NAMES = {0: 'MSS', 1: 'MSI'}


# --------------------------------------------------------------------------- #
#  Carga del modelo DSMIL óptimo (checkpoint + hiperparámetros del bundle)
# --------------------------------------------------------------------------- #
def load_model(device):
    spec = importlib.util.spec_from_file_location(
        'aggregators', REPO / 'pathbench/models/aggregators.py')
    agg = importlib.util.module_from_spec(spec); spec.loader.exec_module(agg)
    mp = json.load(open(BUNDLE / 'model/mil_params.json')); pr = mp['params']
    model = getattr(agg, pr['model'])(
        n_feats=mp['input_shape'], n_out=mp['output_shape'],
        z_dim=pr.get('z_dim', 256), dropout_p=pr.get('dropout_p', 0.1),
        activation_function=pr.get('activation_function', 'ReLU'),
        encoder_layers=pr.get('encoder_layers', 1))
    st = torch.load(BUNDLE / 'model/best_dsmil.ckpt', map_location='cpu')
    sd = {k.removeprefix('model.'): v for k, v in st['state_dict'].items()
          if k.startswith('model.')}
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval(), mp['output_shape']


# --------------------------------------------------------------------------- #
#  Atención del DSMIL por patch (mismo cálculo que en el estudio)
# --------------------------------------------------------------------------- #
def attention(model, feats, n_out, device):
    with torch.no_grad():
        x = feats.unsqueeze(0).to(device)                       # [1, N, 1536]
        inst = model.instance_encoder(x.view(-1, x.size(-1))).view(1, feats.shape[0], -1)
        scores = model.instance_classifier(inst).view(1, feats.shape[0])
        _, mx = scores.max(dim=1)
        critical = inst[0, mx[0]]
        att = F.softmax(model.attention(inst - critical.unsqueeze(0).unsqueeze(0)), dim=1)
    return att.squeeze().cpu().numpy().astype(np.float32)         # [N]


# --------------------------------------------------------------------------- #
#  Distancia coseno y extracción del patch de la WSI
# --------------------------------------------------------------------------- #
def cosdist(a, b):
    return 1.0 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def extract_patch(slide, x, y):
    """Región CONTEXT_PX x CONTEXT_PX centrada en (x,y) a nivel 0 (nativo)."""
    off = (CONTEXT_PX - TILE_PX) // 2
    x0, y0 = x - off, y - off
    w, h = slide.dimensions
    if x0 < 0 or y0 < 0 or x0 + CONTEXT_PX > w or y0 + CONTEXT_PX > h:
        return None
    return slide.read_region((x0, y0), 0, (CONTEXT_PX, CONTEXT_PX)).convert('RGB')


def find_wsi(wsi_dir, slide_id):
    for ext in ('.svs', '.tiff', '.tif', '.ndpi', '.mrxs'):
        p = wsi_dir / f'{slide_id}{ext}'
        if p.exists():
            return p
    return None


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bags_dir', required=True, help='.../256_128_none_h_optimus_0')
    ap.add_argument('--wsi_dir',  required=True)
    ap.add_argument('--out_dir',  required=True)
    ap.add_argument('--top_k', type=int, default=8,   help='patches por slide (atención)')
    ap.add_argument('--top_n', type=int, default=100, help='patches finales por clase')
    args = ap.parse_args()

    bags = Path(args.bags_dir); wsi_dir = Path(args.wsi_dir); out = Path(args.out_dir)
    (out / 'MSS').mkdir(parents=True, exist_ok=True)
    (out / 'MSI').mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, n_out = load_model(device)
    print(f'Dispositivo: {device} | modelo DSMIL cargado')

    # Centroides de referencia (Macarena) incluidos en el bundle
    cent = pd.read_csv(BUNDLE / 'reference/centroids_macarena.csv')
    fcols = [c for c in cent.columns if c.startswith('feat_')]
    C = {int(r['class_id']): r[fcols].values.astype(np.float32)
         for _, r in cent.iterrows()}   # {0: cMSS, 1: cMSI}

    # 1) Recorrer slides -> top-K atención -> imagen + distancias a los centroides
    records = []
    slides = sorted(p.stem for p in bags.glob('*.pt'))
    print(f'Slides a procesar: {len(slides)}')
    tmp_imgs = {}   # (slide, x, y) -> PIL Image (para guardar solo los elegidos)
    for i, sid in enumerate(slides, 1):
        idx_p = bags / f'{sid}.index.npz'
        if not idx_p.exists():
            print(f'  [SKIP] {sid}: sin index'); continue
        feats = torch.load(bags / f'{sid}.pt', map_location='cpu', weights_only=True).float()
        coords = np.load(idx_p)['arr_0']                      # [N, 2] -> (x, y)
        att = attention(model, feats, n_out, device)
        top = np.argsort(att)[::-1][:args.top_k]

        wsi_path = find_wsi(wsi_dir, sid)
        slide_obj = openslide.OpenSlide(str(wsi_path)) if wsi_path else None
        for pidx in top:
            x, y = int(coords[pidx, 0]), int(coords[pidx, 1])
            f = feats[pidx].numpy()
            img = extract_patch(slide_obj, x, y) if slide_obj else None
            if img is not None:
                tmp_imgs[(sid, x, y)] = img
            records.append({'slide': sid, 'x': x, 'y': y,
                            'attention': float(att[pidx]),
                            'dist_MSS': cosdist(f, C[0]),
                            'dist_MSI': cosdist(f, C[1]),
                            'has_img': img is not None})
        if slide_obj: slide_obj.close()
        if i % 20 == 0 or i == len(slides):
            print(f'  [{i}/{len(slides)}] {sid}')

    df = pd.DataFrame(records)
    df.to_csv(out / 'all_topk_patches.csv', index=False)

    # 2) 100 más cercanos a cada centroide -> guardar imágenes
    for cls_id, cls in CLASS_NAMES.items():
        col = f'dist_{cls}'
        sel = df[df['has_img']].nsmallest(args.top_n, col).reset_index(drop=True)
        for rank, r in sel.iterrows():
            img = tmp_imgs.get((r['slide'], int(r['x']), int(r['y'])))
            if img is None: continue
            img.save(out / cls / f'{rank+1:03d}_{r["slide"]}_d{r[col]:.4f}.png')
        sel.to_csv(out / f'closest{args.top_n}_{cls}.csv', index=False)
        print(f'{cls}: {len(sel)} patches guardados en {out/cls}  '
              f'(dist {sel[col].min():.4f}–{sel[col].max():.4f})')

    print(f'\nListo. Salida en: {out}')


if __name__ == '__main__':
    main()
