#!/usr/bin/env python3
"""Genera thumbnails de WSIs CPTAC y TCGA para comparar diagnóstico(FFPE) vs congelado."""
import openslide, glob, os
from pathlib import Path
out=Path('results/colon/wsi_thumbs'); out.mkdir(parents=True,exist_ok=True)
def thumbs(pattern,prefix,n=6):
    for i,f in enumerate(sorted(glob.glob(pattern))[:n]):
        try:
            s=openslide.OpenSlide(f)
            s.get_thumbnail((512,512)).save(out/f'{prefix}_{i:02d}.png')
            print("ok",prefix,i,os.path.basename(f))
        except Exception as e:
            print("err",f,e)
thumbs('/home/PARADIS/datos/wsis/cptac_coad/*.svs','cptac',6)
thumbs('/home/PARADIS/datos/wsis/tcga_coad/*.svs','tcga',6)
