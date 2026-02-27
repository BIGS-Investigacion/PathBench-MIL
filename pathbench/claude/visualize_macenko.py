"""
Visualiza el efecto de la normalización Macenko sobre tiles de TCGA y CPTAC.
Lee tiles directamente de los tfrecords y aplica el normalizador de SlideFlow.

Uso:
    source /shared/home/PARADIS/PathBench-MIL/pathbench_env/bin/activate
    python3 claude/visualize_macenko.py
"""

import sys
import io
import os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/shared/home/PARADIS/PathBench-MIL/slideflow_fork')
import slideflow as sf
from slideflow.norm import StainNormalizer

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
TCGA_TFR = (
    '/home/PARADIS/mama/tfrecords/TCGA-BRCA/256px_128um/'
    'TCGA-3C-AAAU-01A-01-TS1.2F52DD63-7476-4E85-B7C6-E06092DB6CC1.tfrecords'
)
CPTAC_TFR = (
    '/home/PARADIS/mama/tfrecords/CPTAC-BRCA/256px_128um/'
    '01BR001-0684a407-f446-486d-9160-b483cb.tfrecords'
)
OUT_DIR = 'claude'
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Leer tiles
# ---------------------------------------------------------------------------
def read_tile(tfr_path: str, tile_idx: int = 10) -> np.ndarray:
    """Lee un tile de un tfrecord y lo devuelve como array numpy (H, W, 3)."""
    tfr = sf.TFRecord(tfr_path)
    locs = tfr.locations
    _, jpeg_bytes = tfr.get_record_by_xy(*locs[tile_idx])
    img = Image.open(io.BytesIO(jpeg_bytes)).convert('RGB')
    return np.array(img)

print("Leyendo tiles...")
tcga_tile = read_tile(TCGA_TFR, tile_idx=10)
cptac_tile = read_tile(CPTAC_TFR, tile_idx=10)
print(f"  TCGA tile shape: {tcga_tile.shape}")
print(f"  CPTAC tile shape: {cptac_tile.shape}")

# ---------------------------------------------------------------------------
# Aplicar Macenko
# ---------------------------------------------------------------------------
print("Aplicando normalización Macenko (preset v3, referencia TCGA)...")
normalizer = StainNormalizer('macenko')
tcga_norm = normalizer.transform(tcga_tile)
cptac_norm = normalizer.transform(cptac_tile)
print("  Hecho.")

# ---------------------------------------------------------------------------
# Figura comparativa
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

titles = [
    ('TCGA — original', tcga_tile),
    ('TCGA — Macenko', tcga_norm),
    ('CPTAC — original', cptac_tile),
    ('CPTAC — Macenko', cptac_norm),
]

for ax, (title, img) in zip(axes.flat, titles):
    ax.imshow(img)
    ax.set_title(title, fontsize=13)
    ax.axis('off')

fig.suptitle('Normalización Macenko (preset v3 — referencia TCGA)', fontsize=14, y=1.01)
plt.tight_layout()

out_path = os.path.join(OUT_DIR, 'macenko_tiles.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nImagen guardada en: {out_path}")

# ---------------------------------------------------------------------------
# Figura con múltiples tiles por dataset (ver variabilidad)
# ---------------------------------------------------------------------------
N = 6
fig2, axes2 = plt.subplots(4, N, figsize=(N * 3, 12))

for col, idx in enumerate([5, 15, 30, 50, 80, 120]):
    try:
        t = read_tile(TCGA_TFR, idx)
        c = read_tile(CPTAC_TFR, idx)
    except Exception:
        continue
    axes2[0, col].imshow(t);  axes2[0, col].axis('off')
    axes2[1, col].imshow(normalizer.transform(t)); axes2[1, col].axis('off')
    axes2[2, col].imshow(c);  axes2[2, col].axis('off')
    axes2[3, col].imshow(normalizer.transform(c)); axes2[3, col].axis('off')

for ax, label in zip(axes2[:, 0], ['TCGA orig', 'TCGA Macenko', 'CPTAC orig', 'CPTAC Macenko']):
    ax.set_ylabel(label, fontsize=11, rotation=0, labelpad=80, va='center')

fig2.suptitle('Varios tiles — TCGA vs CPTAC — antes y después de Macenko', fontsize=13)
plt.tight_layout()

out_path2 = os.path.join(OUT_DIR, 'macenko_tiles_grid.png')
plt.savefig(out_path2, dpi=130, bbox_inches='tight')
print(f"Grid guardado en: {out_path2}")
