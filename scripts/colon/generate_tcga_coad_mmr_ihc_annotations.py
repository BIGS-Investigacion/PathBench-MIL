#!/usr/bin/env python3
"""
generate_tcga_coad_mmr_ihc_annotations.py

Genera ficheros de anotaciones para TCGA-COAD usando datos IHC de MMR proteins
(MLH1, MSH2, MSH6, PMS2) desde config/annotations/tcga_coad_mmr_ihc_gdc.csv.

  category 0 = Expressed (proteína presente, wildtype)
  category 1 = Lost      (pérdida de expresión, deficiente)

Fuente slides: annotations_colon_mss_msi_all.csv (subconjunto tcga_coad).

Output: config/annotations/annotations_tcga_coad_{protein}_ihc.csv
        [slide, patient, dataset, category]
"""

from pathlib import Path
import pandas as pd

ROOT        = Path(__file__).resolve().parents[2]
ANNOT_DIR   = ROOT / 'config' / 'annotations'
IHC_FILE    = ANNOT_DIR / 'tcga_coad_mmr_ihc_gdc.csv'
SLIDES_FILE = ANNOT_DIR / 'annotations_colon_mss_msi_all.csv'
PROTEINS    = ['MLH1', 'MSH2', 'MSH6', 'PMS2']

# Slides TCGA-COAD disponibles
slides_df = pd.read_csv(SLIDES_FILE)
slides_df = slides_df[slides_df['dataset'] == 'tcga_coad'][['slide', 'patient']].copy()

# IHC labels
ihc = pd.read_csv(IHC_FILE)
ihc = ihc[ihc['case_id'] != 'CDE_ID:2003301'].rename(columns={'case_id': 'patient'})

print(f'Slides TCGA-COAD disponibles: {len(slides_df)}')
print(f'Pacientes con IHC (alguna proteína): {ihc.dropna(subset=PROTEINS, how="all").shape[0]}\n')

for protein in PROTEINS:
    ihc_sub = ihc[['patient', protein]].dropna(subset=[protein]).copy()
    ihc_sub['category'] = ihc_sub[protein].map({'Expressed': 0, 'Lost': 1})
    ihc_sub = ihc_sub.dropna(subset=['category'])
    ihc_sub['category'] = ihc_sub['category'].astype(int)

    merged = slides_df.merge(ihc_sub[['patient', 'category']], on='patient', how='inner')
    merged['dataset'] = 'tcga_coad'
    merged = merged[['slide', 'patient', 'dataset', 'category']]

    n0 = (merged['category'] == 0).sum()
    n1 = (merged['category'] == 1).sum()
    print(f'{protein}: {len(merged)} slides ({n0} Expressed / {n1} Lost)  '
          f'— {merged["patient"].nunique()} pacientes únicos')

    out = ANNOT_DIR / f'annotations_tcga_coad_{protein.lower()}_ihc.csv'
    merged.to_csv(out, index=False)
    print(f'  → {out.name}\n')

# MSS/MSI derivado de IHC: any Lost → dMMR/MSI (cat 1), all Expressed → pMMR/MSS (cat 0)
ihc_any = ihc[PROTEINS].copy()
ihc['n_lost']      = (ihc_any == 'Lost').sum(axis=1)
ihc['n_expressed'] = (ihc_any == 'Expressed').sum(axis=1)
ihc['n_assessed']  = ihc['n_lost'] + ihc['n_expressed']

# Solo pacientes con al menos una proteína evaluada
ihc_msi = ihc[ihc['n_assessed'] > 0].copy()
ihc_msi['category'] = (ihc_msi['n_lost'] > 0).astype(int)

merged_msi = slides_df.merge(ihc_msi[['patient', 'category']], on='patient', how='inner')
merged_msi['dataset'] = 'tcga_coad'
merged_msi = merged_msi[['slide', 'patient', 'dataset', 'category']]

n0 = (merged_msi['category'] == 0).sum()
n1 = (merged_msi['category'] == 1).sum()
print(f'MSS/MSI (IHC): {len(merged_msi)} slides ({n0} pMMR/MSS / {n1} dMMR/MSI)  '
      f'— {merged_msi["patient"].nunique()} pacientes únicos')

out_msi = ANNOT_DIR / 'annotations_tcga_coad_mss_msi_ihc.csv'
merged_msi.to_csv(out_msi, index=False)
print(f'  → {out_msi.name}')