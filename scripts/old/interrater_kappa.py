#!/usr/bin/env python3
"""
Inter-rater reliability between two pathologists (Manolo & Laura).

Aligns patches by IMAGEN column, then computes:
  Weighted kappa (linear weights) for ordinal features:
    - ESTRUCTURA GLANDULAR  (Tubule Formation)     1–3
    - ATIPIA NUCLEAR        (Nuclear Pleomorphism)  1–3
    - MITOSIS               (Mitotic Activity)      0–3
  Cohen's kappa for binary features:
    - NECROSIS, INFILTRADO_LI, INFILTRADO_PMN       0/1

Outputs:
  - results/interrater_kappa.tex
  - results/representative_images_annotation_mean.xlsx  (mean scores per patch)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import cohen_kappa_score

RESULTS = Path(__file__).resolve().parent.parent / 'results'

ORDINAL = {
    'ESTRUCTURA GLANDULAR': 'Tubule Formation',
    'ATIPIA NUCLEAR':       'Nuclear Pleomorphism',
    'MITOSIS':              'Mitotic Activity',
}
BINARY = {
    'NECROSIS':      'Necrosis',
    'INFILTRADO_LI':  'LI',
    'INFILTRADO_PMN': 'PMN',
}

def weighted_kappa_linear(y1, y2):
    return cohen_kappa_score(y1, y2, weights='linear')

def interpret(k):
    if k < 0:       return 'poor'
    if k < 0.20:    return 'slight'
    if k < 0.40:    return 'fair'
    if k < 0.60:    return 'moderate'
    if k < 0.80:    return 'substantial'
    return 'almost perfect'

def main():
    m = pd.read_excel(RESULTS / 'representative_images_annotation_manolo.xlsx')
    l = pd.read_excel(RESULTS / 'representative_images_annotation_laura.xlsx')

    assert len(m) == len(l), f"Row count mismatch: {len(m)} vs {len(l)}"

    # Align by IMAGEN patch filename
    merged = m.merge(l, on='IMAGEN', suffixes=('_m', '_l'))
    assert len(merged) == len(m), \
        f"Merge yielded {len(merged)} rows — some images don't match!"

    print(f"n patches = {len(merged)}")
    print()

    results = []

    def impute(ym, yl):
        """When one rater has NaN, use the other's value for both."""
        ym = ym.astype(float).copy()
        yl = yl.astype(float).copy()
        nan_m = np.isnan(ym)
        nan_l = np.isnan(yl)
        ym[nan_m] = yl[nan_m]   # fill Manolo NaN with Laura
        yl[nan_l] = ym[nan_l]   # fill Laura NaN with Manolo
        valid = ~(np.isnan(ym) | np.isnan(yl))
        return ym[valid].astype(int), yl[valid].astype(int), valid.sum()

    print("── Weighted kappa (linear) — Ordinal ────────────────────────")
    for col, label in ORDINAL.items():
        y_m, y_l, n_valid = impute(merged[f'{col}_m'].values, merged[f'{col}_l'].values)
        k = weighted_kappa_linear(y_m, y_l)
        print(f"  {label:25s}  κ_w = {k:.3f}  ({interpret(k)})  [n={n_valid}]")
        results.append({'Feature': label, 'Type': 'ordinal', 'Kappa': k,
                        'KappaType': r'$\kappa_w$', 'n': n_valid})

    print()
    print("── Cohen's kappa — Binary ───────────────────────────────────")
    for col, label in BINARY.items():
        y_m, y_l, n_valid = impute(merged[f'{col}_m'].values, merged[f'{col}_l'].values)
        k = cohen_kappa_score(y_m, y_l)
        print(f"  {label:25s}  κ   = {k:.3f}  ({interpret(k)})  [n={n_valid}]")
        results.append({'Feature': label, 'Type': 'binary', 'Kappa': k,
                        'KappaType': r'$\kappa$', 'n': n_valid})

    # ── Mean scores per patch (NaN-aware: use other rater's value when one is missing) ──
    mean_df = merged[['IMAGEN', 'ETIQUETA_m']].rename(columns={'ETIQUETA_m': 'ETIQUETA'})
    for col in list(ORDINAL.keys()) + list(BINARY.keys()):
        ym = merged[f'{col}_m'].values.astype(float)
        yl = merged[f'{col}_l'].values.astype(float)
        nan_m = np.isnan(ym)
        nan_l = np.isnan(yl)
        ym_imp = np.where(nan_m, yl, ym)   # Manolo NaN → use Laura
        yl_imp = np.where(nan_l, ym, yl)   # Laura NaN → use Manolo
        mean_df = mean_df.copy()
        mean_df[col] = (ym_imp + yl_imp) / 2.0

    out_xlsx = RESULTS / 'representative_images_annotation_mean.xlsx'
    mean_df.to_excel(out_xlsx, index=False)
    print(f"\nMean scores saved → {out_xlsx}")

    # ── LaTeX table ───────────────────────────────────────────────────────────
    df = pd.DataFrame(results)

    lines = [
        r"\begin{table}[h!]",
        r"\centering",
        r"\caption{Inter-rater reliability between two pathologists (P1, P2) "
        r"across $n=275$ patches. "
        r"Patches were aligned by image identifier prior to computing agreement. "
        r"Weighted $\kappa_w$ (linear weights) is reported for ordinal features "
        r"(Tubule Formation, Nuclear Pleomorphism, Mitotic Activity); "
        r"Cohen's $\kappa$ for binary features (Necrosis, LI, PMN). "
        r"Interpretation follows Landis \& Koch (1977).}",
        r"\label{tab:interrater}",
        r"\small",
        r"\begin{tabular}{llc}",
        r"\toprule",
        r"\textbf{Feature} & \textbf{Statistic} & \textbf{Value} \\",
        r"\midrule",
        r"\multicolumn{3}{l}{\textit{Ordinal features (weighted $\kappa_w$, linear)}} \\",
        r"\midrule",
    ]
    for _, row in df[df['Type'] == 'ordinal'].iterrows():
        k = row['Kappa']
        lines.append(
            f"    {row['Feature']} & {row['KappaType']} & "
            f"${k:.3f}$ ({interpret(k)}) \\\\"
        )
    lines += [
        r"\midrule",
        r"\multicolumn{3}{l}{\textit{Binary features (Cohen's $\kappa$)}} \\",
        r"\midrule",
    ]
    for _, row in df[df['Type'] == 'binary'].iterrows():
        k = row['Kappa']
        lines.append(
            f"    {row['Feature']} & {row['KappaType']} & "
            f"${k:.3f}$ ({interpret(k)}) \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    tex = "\n".join(lines)
    out_tex = RESULTS / 'interrater_kappa.tex'
    out_tex.write_text(tex, encoding='utf-8')
    print(f"LaTeX table saved → {out_tex}")
    print()
    print(tex)


if __name__ == '__main__':
    main()