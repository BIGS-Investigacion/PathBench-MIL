#!/usr/bin/env python3
"""
pathologist_agreement.py

Calcula el agreement inter-observador entre dos patólogos (LMG, MPP) sobre las
características morfológicas binarias anotadas en los mismos patches.

Para cada característica y conjunto reporta:
  - % de acuerdo (observed agreement)
  - Cohen's kappa

Uso:
  python scripts/colon/pathologist_agreement.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score

ROOT  = Path(__file__).resolve().parents[2]
F_LMG = ROOT / 'LMG.xlsx'
F_MPP = ROOT / 'MPP.xlsx'

FEATURES = [
    'Arquitectura glandular', 'Diferenciación tumoral', 'Serración glandular',
    'Mucina', 'TILs intraepiteliales', 'Inflamación peritunoral',
    'Necrosis intraluminal',
]
SHEETS = ['macarena', 'tcga_coad', 'cptac_coad']


def kappa_label(k):
    """Interpretación de Landis & Koch."""
    if np.isnan(k):           return '—'
    if k < 0.00:              return 'Poor'
    if k < 0.20:              return 'Slight'
    if k < 0.40:              return 'Fair'
    if k < 0.60:              return 'Moderate'
    if k < 0.80:              return 'Substantial'
    return 'Almost perfect'


def agreement(a, b):
    a = np.asarray(a); b = np.asarray(b)
    obs = (a == b).mean()
    try:
        k = cohen_kappa_score(a, b)
    except ValueError:
        k = np.nan
    return obs, k


def merge_raters(sheet):
    lmg = pd.read_excel(F_LMG, sheet_name=sheet)
    mpp = pd.read_excel(F_MPP, sheet_name=sheet)
    merged = lmg.merge(mpp, on='filename', suffixes=('_lmg', '_mpp'))
    return merged


def main():
    print('Agreement inter-observador  (LMG vs MPP)')
    print('=' * 78)

    all_rows = []
    pooled = {feat: ([], []) for feat in FEATURES}

    for sheet in SHEETS:
        m = merge_raters(sheet)
        print(f'\n{sheet}  (n={len(m)} patches)')
        print(f'  {"Característica":<26} {"% acuerdo":>10} {"kappa":>8}  {"interpretación"}')
        print('  ' + '-' * 70)

        for feat in FEATURES:
            a = m[f'{feat}_lmg']
            b = m[f'{feat}_mpp']
            obs, k = agreement(a, b)
            pooled[feat][0].extend(a.tolist())
            pooled[feat][1].extend(b.tolist())
            print(f'  {feat:<26} {obs*100:>9.1f}% {k:>8.3f}  {kappa_label(k)}')
            all_rows.append({'dataset': sheet, 'feature': feat,
                             'pct_agreement': obs, 'kappa': k})

    # Pooled (todos los conjuntos juntos)
    print(f'\nGLOBAL  (todos los conjuntos)')
    print(f'  {"Característica":<26} {"% acuerdo":>10} {"kappa":>8}  {"interpretación"}')
    print('  ' + '-' * 70)
    for feat in FEATURES:
        a, b = pooled[feat]
        obs, k = agreement(a, b)
        print(f'  {feat:<26} {obs*100:>9.1f}% {k:>8.3f}  {kappa_label(k)}')
        all_rows.append({'dataset': 'GLOBAL', 'feature': feat,
                         'pct_agreement': obs, 'kappa': k})

    # Agreement en la clase MSS/MSI
    print(f'\nClase MSS/MSI')
    for sheet in SHEETS:
        m = merge_raters(sheet)
        obs, k = agreement(m['class_lmg'], m['class_mpp'])
        print(f'  {sheet:<14} % acuerdo={obs*100:>6.1f}%  kappa={k:>6.3f}  ({kappa_label(k)})')

    out = ROOT / 'results/colon/pathologist_agreement.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out, index=False)
    print(f'\nGuardado en: {out}')


if __name__ == '__main__':
    main()