"""
Genera captions con CONCH para imágenes de citología.
Recorre data/Carcinoma/ y data/Negativas/ y guarda los resultados en
results/cytology/captions.csv
"""
from pathlib import Path
import torch
import pandas as pd
from PIL import Image

ROOT     = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / 'data'
OUT_DIR  = ROOT / 'results/cytology'
OUT_DIR.mkdir(parents=True, exist_ok=True)

import sys
sys.path.insert(0, str(ROOT))

CLASSES = ['Carcinoma', 'Negativas']

from pathbench.models.feature_extractors import ConchCaptioner

print('Cargando CONCH...')
captioner = ConchCaptioner(seq_len=60, generation_type='top_k')
print('Modelo listo.\n')

records = []
for cls in CLASSES:
    cls_dir = DATA_DIR / cls
    images  = sorted(cls_dir.glob('*.png')) + sorted(cls_dir.glob('*.jpg'))
    print(f'[{cls}] {len(images)} imágenes')
    for img_path in images:
        img = Image.open(img_path).convert('RGB')
        caption = captioner.caption(img)
        print(f'  {img_path.name}: {caption}')
        records.append({
            'class':    cls,
            'filename': img_path.name,
            'caption':  caption,
        })

df = pd.DataFrame(records)
out_path = OUT_DIR / 'captions.csv'
df.to_csv(out_path, index=False)
print(f'\nGuardado: {out_path}')