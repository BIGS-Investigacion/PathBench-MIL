"""
Genera un Excel con 3 pestañas (macarena, tcga_coad, cptac_coad).
Cada pestaña lista los nombres de las imágenes seleccionadas (closest100),
separadas por clase (MSS / MSI).
"""
from pathlib import Path
import pandas as pd

ROOT     = Path(__file__).resolve().parents[2]
OUT_DIR  = ROOT / 'results/colon'
DATASETS = ['macarena', 'tcga_coad', 'cptac_coad']

with pd.ExcelWriter(OUT_DIR / 'patches_selection.xlsx', engine='openpyxl') as writer:
    for dataset in DATASETS:
        base = OUT_DIR / f'closest100_{dataset}'
        rows = []
        for cls in ['MSS', 'MSI']:
            cls_dir = base / cls
            if not cls_dir.exists():
                print(f'[WARN] No existe: {cls_dir}')
                continue
            for img in sorted(cls_dir.iterdir()):
                rows.append({'filename': img.name, 'class': cls})

        df = pd.DataFrame(rows, columns=['filename', 'class'])
        df.to_excel(writer, sheet_name=dataset, index=False)
        print(f'{dataset}: {len(df)} imágenes')

print(f'\nGuardado: {OUT_DIR / "patches_selection.xlsx"}')