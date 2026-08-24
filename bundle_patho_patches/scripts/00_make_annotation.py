#!/usr/bin/env python3
"""
00_make_annotation.py
----------------------
PathBench necesita un CSV de anotación que liste los slides a procesar. Como en
este bundle NO tenemos etiquetas (la clase se decide después por cercanía a los
centroides de Macarena), generamos un CSV con una categoría "dummy" = 0.

Escanea la carpeta de WSIs y escribe:  slide,patient,dataset,category

Uso:
  python 00_make_annotation.py <carpeta_wsis> <salida_csv>
"""
import sys
from pathlib import Path
import pandas as pd

# Extensiones de WSI admitidas (añade las que necesites)
EXTS = {'.svs', '.tiff', '.tif', '.ndpi', '.mrxs'}

def main():
    wsi_dir = Path(sys.argv[1])
    out_csv = Path(sys.argv[2])
    rows = []
    for f in sorted(wsi_dir.iterdir()):
        if f.suffix.lower() in EXTS:
            slide = f.stem                       # nombre sin extensión = id del slide
            rows.append({'slide': slide, 'patient': slide,
                         'dataset': 'input', 'category': 0})
    if not rows:
        sys.exit(f'ERROR: no se encontraron WSIs en {wsi_dir} (extensiones {EXTS})')
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f'Anotación escrita: {out_csv}  ({len(df)} slides)')

if __name__ == '__main__':
    main()
