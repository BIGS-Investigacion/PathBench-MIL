#!/usr/bin/env python3
"""
feature_significance.py

Para cada fichero de anotación (LMG, MPP, FUSION_formato_LMG) testea la asociación
de cada característica morfológica binaria con el estado MSI/MSS mediante el test
exacto de Fisher (contingencia 2x2: feature 0/1  x  MSS/MSI).

Reporta p-valor, odds ratio y significancia, y compara los tres ficheros para
detectar discrepancias según cuál se use.

Uso:
  python scripts/colon/feature_significance.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

ROOT   = Path(__file__).resolve().parents[2]
FILES  = {
    'LMG':    ROOT / 'LMG.xlsx',
    'MPP':    ROOT / 'MPP.xlsx',
    'FUSION': ROOT / 'FUSION_formato_LMG.xlsx',
}
SHEETS = ['macarena', 'tcga_coad', 'cptac_coad']
FEATURES = [
    'Arquitectura glandular', 'Diferenciación tumoral', 'Serración glandular',
    'Mucina', 'TILs intraepiteliales', 'Inflamación peritunoral',
    'Necrosis intraluminal',
]


def load_sheet(path, sheet):
    df = pd.read_excel(path, sheet_name=sheet)
    df['y'] = (df['class'] == 'MSI').astype(int)
    return df


def fisher_feature(df, feat):
    """Contingencia 2x2: feature (0/1) x MSI (0/1). Devuelve (OR, p)."""
    a = ((df[feat] == 1) & (df['y'] == 1)).sum()  # feat+ MSI
    b = ((df[feat] == 1) & (df['y'] == 0)).sum()  # feat+ MSS
    c = ((df[feat] == 0) & (df['y'] == 1)).sum()  # feat- MSI
    d = ((df[feat] == 0) & (df['y'] == 0)).sum()  # feat- MSS
    table = [[a, b], [c, d]]
    try:
        odds, p = fisher_exact(table, alternative='two-sided')
    except ValueError:
        odds, p = np.nan, np.nan
    return odds, p, (a, b, c, d)


def stars(p):
    if np.isnan(p):  return ''
    if p < 0.001:    return '***'
    if p < 0.01:     return '**'
    if p < 0.05:     return '*'
    return 'ns'


def analyze(df):
    """Fisher por caracteristica + correccion FDR-BH. Devuelve DataFrame."""
    rows, pvals = [], []
    for feat in FEATURES:
        odds, p, counts = fisher_feature(df, feat)
        rows.append({'feature': feat, 'odds_ratio': odds, 'p': p, 'counts': counts})
        pvals.append(p)
    rej, p_adj, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
    for r, pa, rj in zip(rows, p_adj, rej):
        r['p_adj'] = pa
        r['sig_adj'] = rj
    return pd.DataFrame(rows)


def main():
    # results[cohort][file] = DataFrame
    results = {sh: {} for sh in SHEETS}
    for sheet in SHEETS:
        for name, path in FILES.items():
            df = load_sheet(path, sheet)
            results[sheet][name] = analyze(df)

    for sheet in SHEETS:
        n = len(load_sheet(FILES['LMG'], sheet))
        print(f'\n{"#"*74}\n#  {sheet.upper()}  (n={n} patches)\n{"#"*74}')
        for name in FILES:
            res = results[sheet][name]
            print(f'\n  --- {name} ---')
            print(f'  {"Caracteristica":<26} {"OR":>8} {"p":>10} {"p_adj":>10}  sig')
            print('  ' + '-'*62)
            for _, r in res.iterrows():
                print(f'  {r["feature"]:<26} {r["odds_ratio"]:>8.2f} '
                      f'{r["p"]:>10.4f} {r["p_adj"]:>10.4f}  {stars(r["p_adj"])}')

        print(f'\n  significativa (p_adj<0.05)?  {sheet}')
        print(f'  {"Caracteristica":<26} {"LMG":>6} {"MPP":>6} {"FUSION":>7}   discrepancia')
        print('  ' + '-'*60)
        for feat in FEATURES:
            flags = {name: results[sheet][name][
                results[sheet][name]['feature']==feat].iloc[0]['sig_adj']
                for name in FILES}
            disc = '  <-- DISCREPA' if len(set(flags.values())) > 1 else ''
            mark = lambda b: 'SI' if b else 'no'
            print(f'  {feat:<26} {mark(flags["LMG"]):>6} {mark(flags["MPP"]):>6} '
                  f'{mark(flags["FUSION"]):>7}{disc}')

    out = ROOT / 'results/colon/feature_significance_by_cohort.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    all_res = pd.concat([
        results[sh][name].assign(cohort=sh, file=name)
        for sh in SHEETS for name in FILES
    ], ignore_index=True)
    all_res.to_csv(out, index=False)
    print(f'\nGuardado en: {out}')


if __name__ == '__main__':
    main()
