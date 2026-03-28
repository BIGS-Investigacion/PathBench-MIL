#!/usr/bin/env python3
"""
scripts/generate_tables_latex.py

Generates LaTeX tables from results/per_class_metrics.csv and supporting files.

Output files (all in results/):
    performance_table.tex       — TCGA / CPTAC / RPD per MIL × task × class
    stain_norm_table.tex        — None / Macenko / Δn per MIL × task × class
    prevalence_table.tex        — prev_tcga / prev_cptac / Δp per task × class
    centroid_table.tex          — d_clam / d_dsmil / d_transmil / d_mean per task × class
    morphology_means_table.tex  — mean morphological scores per task × class × cohort

Usage:
    python scripts/generate_tables_latex.py [--metrics results/per_class_metrics.csv]
                                            [--dist_dir results/patch_intersection_all]
                                            [--out_dir results]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ── Display names ─────────────────────────────────────────────────────────────

MIL_DISPLAY = {
    'clam_mil_mb': 'CLAM-MB',
    'dsmil':       'DSMIL',
    'transmil':    'TransMIL',
}
MIL_ORDER = ['clam_mil_mb', 'dsmil', 'transmil']

TASK_DISPLAY = {
    'pam50': 'PAM50',
    'er':    'ER',
    'pr':    'PR',
    'erbb2': 'HER2',
}
TASK_ORDER = ['pam50', 'er', 'pr', 'erbb2']

# Per-class display names for tables
# Unified class display names (all tables use the same names)
CLS_DISPLAY = {
    ('pam50', 0): 'Basal-like',
    ('pam50', 1): 'HER2-enriched',
    ('pam50', 2): 'Luminal A',
    ('pam50', 3): 'Luminal B',
    ('pam50', 4): 'Normal-like',
    ('er',    0): 'ER-negative',
    ('er',    1): 'ER-positive',
    ('pr',    0): 'PR-negative',
    ('pr',    1): 'PR-positive',
    ('erbb2', 0): 'HER2-negative',
    ('erbb2', 1): 'HER2-positive',
}
CLS_PREV_DISPLAY  = CLS_DISPLAY
CLS_MORPH_DISPLAY = CLS_DISPLAY

METRIC_DISPLAY = {'pr_auc': 'PR-AUC', 'f1': 'F1'}

MORPH_COLS = ['tubule', 'nuclear', 'mitotic', 'necrosis', 'li', 'pmn']
MORPH_HEADER = ['Tubule', 'Nuclear', 'Mitotic', 'Necrosis', 'LI', 'PMN']


# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt(v, decimals=3):
    """Format float, NaN → '---'."""
    if isinstance(v, float) and np.isnan(v):
        return r'\text{---}'
    return f'{v:.{decimals}f}'


def fmt_signed(v, decimals=3):
    """Format with explicit sign, NaN → '---'."""
    if isinstance(v, float) and np.isnan(v):
        return r'\text{---}'
    v = abs(v) if v == 0 else v  # avoid -0.000
    sign = '' if v == 0 else ('+' if v > 0 else '')
    return f'${sign}{v:.{decimals}f}$'


def classes_for(task: str, df: pd.DataFrame) -> list[int]:
    """Sorted class ids present for this task."""
    return sorted(df[df['task'] == task]['cls_id'].unique())


def n_classes(task: str, df: pd.DataFrame) -> int:
    return len(classes_for(task, df))


# ── Table 1: Performance (TCGA / CPTAC / RPD) ────────────────────────────────

def make_performance_table(df: pd.DataFrame) -> str:
    lines = [
        r'\begin{table}[ht]',
        r'\centering',
        r'\caption{Performance per MIL aggregator. TCGA: hold-out test score'
        r' (no stain normalisation); CPTAC: external test score'
        r' (no stain normalisation). RPD: relative performance drop.}',
        r'\label{tab:mccv-ho-performance}',
        r'\begin{tabular}{lllcrrr}',
        r'\toprule',
        r'\textbf{MIL} & \textbf{Task} & \textbf{Class} & \textbf{Metric}'
        r' & \textbf{TCGA} & \textbf{CPTAC} & \textbf{RPD} \\',
        r'\midrule',
    ]

    total_rows = sum(n_classes(t, df) for t in TASK_ORDER)

    for mi, mil in enumerate(MIL_ORDER):
        mdf = df[df['mil'] == mil]

        for ti, task in enumerate(TASK_ORDER):
            tdf = mdf[mdf['task'] == task].sort_values('cls_id')
            cls_ids = list(tdf['cls_id'])
            metric = METRIC_DISPLAY[tdf.iloc[0]['metric']]
            n_cls = len(cls_ids)

            for ci, cls_id in enumerate(cls_ids):
                row = tdf[tdf['cls_id'] == cls_id].iloc[0]
                cls_name = CLS_DISPLAY[(task, cls_id)]

                mil_col  = (rf'\multirow{{{total_rows}}}{{*}}{{{MIL_DISPLAY[mil]}}}'
                            if ti == 0 and ci == 0 else '')
                task_col = (rf'\multirow{{{n_cls}}}{{*}}{{{TASK_DISPLAY[task]}}}'
                            if ci == 0 else '')
                met_col  = (rf'\multirow{{{n_cls}}}{{*}}{{{metric}}}'
                            if ci == 0 else '')

                rpd = row['rpd']
                lines.append(
                    f'   {mil_col} & {task_col} & {cls_name} & {met_col}'
                    f' & {fmt(row["none_tcga"])} & {fmt(row["none_cptac"])}'
                    f' & {fmt_signed(rpd)} \\\\'
                )

        if mi < len(MIL_ORDER) - 1:
            lines.append(r'\midrule')

    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


# ── Table 2: Stain normalisation (None / Macenko / Δn) ───────────────────────

def make_stain_norm_table(df: pd.DataFrame) -> str:
    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{Effect of stain normalisation on per-class performance by MIL technique.'
        r' Scores evaluated on CPTAC (external test set).}',
        r'\label{tab:stain-norm-detailed}',
        r'\begin{tabular}{lllrrr}',
        r'\toprule',
        r'\textbf{MIL} & \textbf{Task} & \textbf{Class}'
        r' & \textbf{None} & \textbf{Macenko} & $\Delta n$ \\',
        r'\midrule',
    ]

    total_rows = sum(n_classes(t, df) for t in TASK_ORDER)

    for mi, mil in enumerate(MIL_ORDER):
        mdf = df[df['mil'] == mil]

        for ti, task in enumerate(TASK_ORDER):
            tdf = mdf[mdf['task'] == task].sort_values('cls_id')
            cls_ids = list(tdf['cls_id'])
            n_cls = len(cls_ids)

            for ci, cls_id in enumerate(cls_ids):
                row = tdf[tdf['cls_id'] == cls_id].iloc[0]
                cls_name = CLS_DISPLAY[(task, cls_id)]

                mil_col  = (rf'\multirow{{{total_rows}}}{{*}}{{{MIL_DISPLAY[mil]}}}'
                            if ti == 0 and ci == 0 else ' ')
                task_col = (rf'\multirow{{{n_cls}}}{{*}}{{{TASK_DISPLAY[task]}}}'
                            if ci == 0 else ' ')

                lines.append(
                    f'  {mil_col} & {task_col} & {cls_name}'
                    f' & {fmt(row["none_cptac"])} & {fmt(row["mac_cptac"])}'
                    f' & {fmt_signed(row["delta_n"])} \\\\'
                )

            if ti < len(TASK_ORDER) - 1:
                lines.append(r'\cmidrule{2-6}')

        if mi < len(MIL_ORDER) - 1:
            lines.append(r'\midrule')

    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


# ── Table 3: Prevalence ───────────────────────────────────────────────────────

def make_prevalence_table(df: pd.DataFrame) -> str:
    # Use first MIL's data (prevalence is dataset-level, same for all models)
    ref = df[df['mil'] == MIL_ORDER[0]]

    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{Class prevalence in TCGA-BRCA and CPTAC-BRCA cohorts'
        r' with calculated prevalence shifts.}',
        r'\label{tab:prevalence_shift_calc}',
        r'\begin{tabular}{llccc}',
        r'\toprule',
        r'\textbf{Task} & \textbf{Class}'
        r' & \textbf{$p_{\text{TCGA}}$} & \textbf{$p_{\text{CPTAC}}$}'
        r' & \textbf{$\Delta p$} \\',
        r'\midrule',
    ]

    for ti, task in enumerate(TASK_ORDER):
        tdf = ref[ref['task'] == task].sort_values('cls_id')
        cls_ids = list(tdf['cls_id'])
        n_cls = len(cls_ids)

        for ci, cls_id in enumerate(cls_ids):
            row = tdf[tdf['cls_id'] == cls_id].iloc[0]
            cls_name = CLS_PREV_DISPLAY[(task, cls_id)]
            task_col = (rf'\multirow{{{n_cls}}}{{*}}{{{TASK_DISPLAY[task]}}}'
                        if ci == 0 else '')
            lines.append(
                f'    {task_col} & {cls_name}'
                f' & {fmt(row["prev_tcga"], 3)} & {fmt(row["prev_cptac"], 3)}'
                f' & {fmt_signed(row["delta_p"], 3)} \\\\'
            )

        if ti < len(TASK_ORDER) - 1:
            lines.append(r'    \midrule')

    lines += [r'    \bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


# ── Table 4: Centroid distances ───────────────────────────────────────────────

def make_centroid_table(dist_dir: Path, df: pd.DataFrame) -> str:
    lines = [
        r'\begin{table}[ht]',
        r'\centering',
        r'\caption{Cosine centroid distances per MIL model and mean ($d_c$)'
        r' between TCGA and CPTAC class centroids in Virchow2 feature space.}',
        r'\label{tab:covariate_shift}',
        r'\begin{tabular}{llrrrr}',
        r'\toprule',
        r'Task & Class & CLAM-MB & DSMIL & TransMIL & $d_c$ \\',
        r'\midrule',
    ]

    for task in TASK_ORDER:
        csv = dist_dir / task / 'centroid_distances_all.csv'
        if not csv.exists():
            print(f'[WARN] {csv} not found, skipping task {task}')
            continue
        ddf = pd.read_csv(csv).sort_values('class')
        for _, row in ddf.iterrows():
            cls_id = int(row['class'])
            cls_name = CLS_DISPLAY.get((task, cls_id), str(cls_id))
            lines.append(
                f'    {TASK_DISPLAY[task]} & {cls_name}'
                f' & {fmt(row["d_clam"], 3)}'
                f' & {fmt(row["d_dsmil"], 3)}'
                f' & {fmt(row["d_transmil"], 3)}'
                f' & {fmt(row["d_mean"], 3)} \\\\'
            )

    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


# ── Table 5: Morphological means ──────────────────────────────────────────────

def make_morph_means_table(df: pd.DataFrame) -> str:
    ref = df[df['mil'] == MIL_ORDER[0]]  # morph means are model-independent

    lines = [
        r'\begin{table}[htbp]',
        r'\tiny',
        r'\centering',
        r'\caption{Mean histomorphological feature values for TCGA-BRCA and CPTAC-BRCA'
        r' patches ($n=25$ patches per class per cohort). Values are means of the two'
        r' pathologists\textquotesingle\ scores; when one score was missing,'
        r' the other\textquotesingle s value was used.}',
        r'\label{tab:both-cohorts-means}',
        r'\begin{tabular}{lllcccccc}',
        r'\toprule',
        r'\textbf{Task} & \textbf{Class} & \textbf{Cohort}'
        + ''.join(f' & \\textbf{{{h}}}' for h in MORPH_HEADER) + r' \\',
        r'\midrule',
    ]

    n_task_rows = {task: n_classes(task, df) * 2 for task in TASK_ORDER}

    for ti, task in enumerate(TASK_ORDER):
        tdf = ref[ref['task'] == task].sort_values('cls_id')
        cls_ids = list(tdf['cls_id'])
        n_cls = len(cls_ids)

        for ci, cls_id in enumerate(cls_ids):
            row = tdf[tdf['cls_id'] == cls_id].iloc[0]
            cls_name = CLS_MORPH_DISPLAY[(task, cls_id)]

            task_col = (rf'\multirow{{{n_task_rows[task]}}}{{*}}{{{TASK_DISPLAY[task]}}}'
                        if ci == 0 else ' ')
            cls_col  = rf'\multirow{{2}}{{*}}{{{cls_name}}}'

            cptac_vals = ' & '.join(fmt(row[f'cptac_{c}'], 2) for c in MORPH_COLS)
            tcga_vals  = ' & '.join(fmt(row[f'tcga_{c}'],  2) for c in MORPH_COLS)

            lines.append(f' {task_col} & {cls_col} & CPTAC & {cptac_vals} \\\\')
            lines.append(f'  &  & TCGA & {tcga_vals} \\\\')

        if ti < len(TASK_ORDER) - 1:
            lines.append(r'\midrule')

    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


# ── Table 6: Domain-shift factors summary ────────────────────────────────────

def make_factors_table(df: pd.DataFrame) -> str:
    """
    One row per (task, class), averaged across MIL models.
    Columns: δn, Δp, d_c, B̃_c, RPD.
    d_c and B̃_c are model-independent; δn and RPD are averaged across the 3 MIL models.
    """
    lines = [
        r'\begin{table}[ht]',
        r'\centering',
        r'\caption{Domain shift factors ($\Delta n$, $d_c$, $\Delta p$'
        r' and $\tilde{B}_c$) as independent variables for statistical analysis'
        r' with relative performance degradation (RPD) as dependent variable.'
        r' $\Delta n$ and RPD are averaged across the three MIL models.}',
        r'\label{tab:multivariate-factors}',
        r'\begin{tabular}{llrrrrrr}',
        r'\toprule',
        r'\textbf{Task} & \textbf{Class}'
        r' & $\Delta n$ & $\Delta p$ & $d_c$ & $\tilde{B}_c$ & \textbf{RPD} \\',
        r'\midrule',
    ]

    for ti, task in enumerate(TASK_ORDER):
        tdf = df[df['task'] == task]
        cls_ids = sorted(tdf['cls_id'].unique())
        n_cls = len(cls_ids)

        for ci, cls_id in enumerate(cls_ids):
            cdf = tdf[tdf['cls_id'] == cls_id]
            cls_name = CLS_DISPLAY[(task, cls_id)]

            delta_n = cdf['delta_n'].mean()
            delta_p = cdf['delta_p'].iloc[0]   # same for all MIL models
            d_c     = cdf['d_c'].iloc[0]        # model-independent mean
            b_tilde = cdf['b_tilde'].iloc[0]    # model-independent
            rpd_vals = cdf['rpd'].dropna()
            rpd = rpd_vals.mean() if not rpd_vals.empty else float('nan')

            task_col = (rf'\multirow{{{n_cls}}}{{*}}{{{TASK_DISPLAY[task]}}}'
                        if ci == 0 else ' ')

            lines.append(
                f'  {task_col} & {cls_name}'
                f' & {fmt_signed(delta_n)}'
                f' & {fmt_signed(delta_p)}'
                f' & {fmt(d_c)}'
                f' & {fmt_signed(b_tilde)}'
                f' & {fmt_signed(rpd)} \\\\'
            )

        if ti < len(TASK_ORDER) - 1:
            lines.append(r'\midrule')

    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(metrics_csv: Path, dist_dir: Path, out_dir: Path) -> None:
    df = pd.read_csv(metrics_csv)
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        'performance_table.tex':      make_performance_table(df),
        'stain_norm_table.tex':       make_stain_norm_table(df),
        'prevalence_table.tex':       make_prevalence_table(df),
        'centroid_table.tex':         make_centroid_table(dist_dir, df),
        'morphology_means_table.tex': make_morph_means_table(df),
        'factors_table.tex':          make_factors_table(df),
    }

    for fname, content in tables.items():
        path = out_dir / fname
        path.write_text(content, encoding='utf-8')
        print(f'Saved → {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate LaTeX tables from per_class_metrics.csv.')
    parser.add_argument('--metrics', default='results/per_class_metrics.csv')
    parser.add_argument('--dist_dir', default='results/patch_intersection_all_cosine')
    parser.add_argument('--out_dir', default='results')
    args = parser.parse_args()

    main(Path(args.metrics), Path(args.dist_dir), Path(args.out_dir))