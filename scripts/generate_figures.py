#!/usr/bin/env python3
"""
scripts/generate_figures.py

Generates all figures for the paper from results/per_class_metrics.csv.

Figures produced:
    results/figures/barplot_{task}.eps      — per-task bar chart (TCGA vs CPTAC)
    results/figures/barplot_all.{eps,png}   — combined figure (all tasks)

Usage:
    python scripts/generate_figures.py \\
        [--metrics results/per_class_metrics.csv] \\
        [--output_dir results/figures]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────

MIL_ORDER = ['clam_mil_mb', 'dsmil', 'transmil']

TASK_ORDER = ['er', 'erbb2', 'pr', 'pam50']

TASK_DISPLAY = {
    'er':    'ER',
    'erbb2': 'HER2',
    'pr':    'PR',
    'pam50': 'PAM50',
}

METRIC_LABEL = {
    'er':    'PR-AUC',
    'erbb2': 'PR-AUC',
    'pr':    'PR-AUC',
    'pam50': 'F1',
}

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

COLOR = {
    'clam_mil_mb': '#FADADD',
    'dsmil':       '#F06292',
    'transmil':    '#AD1457',
    'clam_noopt':  '#888888',
}
LABEL = {
    'clam_mil_mb': 'Opt-CLAM',
    'dsmil':       'Opt-DSMIL',
    'transmil':    'Opt-TransMIL',
    'clam_noopt':  'CLAM',
}

ALL_MODELS = MIL_ORDER + ['clam_noopt']

# CLAM without Optuna optimisation (MCCV=TCGA proxy, Hold-out=CPTAC)
CLAM_NOOPT = {
    ('pam50', 0): (0.807, 0.756),
    ('pam50', 1): (0.377, 0.000),
    ('pam50', 2): (0.833, 0.767),
    ('pam50', 3): (0.549, 0.265),
    ('pam50', 4): (0.146, 0.000),
    ('er',    0): (0.590, 0.509),
    ('er',    1): (0.901, 0.687),
    ('pr',    0): (0.579, 0.694),
    ('pr',    1): (0.793, 0.716),
    ('erbb2', 0): (0.859, 0.891),
    ('erbb2', 1): (0.194, 0.109),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_value(df: pd.DataFrame, task: str, mil: str, cls_id: int, col: str) -> float:
    row = df[(df['task'] == task) & (df['mil'] == mil) & (df['cls_id'] == cls_id)]
    return float(row[col].iloc[0]) if len(row) else 0.0


def _style_ax(ax, x_labels, metric_lbl):
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_ylabel(metric_lbl, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='#E8A0B4')
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    for spine in ax.spines.values():
        spine.set_edgecolor('#C47A9A')


def _fill_ax(ax, df: pd.DataFrame, task: str, classes: list,
             x_labels: list, col: str, title: str, fontsize_val: float = 5.5):
    width = 0.18
    x = np.arange(len(classes))
    for i, mil in enumerate(ALL_MODELS):
        values = []
        for cls_id in classes:
            if mil == 'clam_noopt':
                tcga_v, cptac_v = CLAM_NOOPT.get((task, cls_id), (0.0, 0.0))
                val = tcga_v if col == 'none_tcga' else cptac_v
            else:
                val = get_value(df, task, mil, cls_id, col)
            values.append(val)

        offset = (i - (len(ALL_MODELS) - 1) / 2) * width
        edge_col = '#555555' if mil == 'clam_noopt' else '#7B1B3A'
        bars = ax.bar(x + offset, values, width,
                      color=COLOR[mil], edgecolor=edge_col, linewidth=0.6)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01, f'{v:.2f}',
                    ha='center', va='bottom',
                    fontsize=fontsize_val,
                    color='#333333' if mil == 'clam_noopt' else '#4A0020')

    ax.set_title(title, fontsize=10, color='#8B1A4A')
    _style_ax(ax, x_labels, METRIC_LABEL[task])


def _legend_patches():
    return [mpatches.Patch(facecolor=COLOR[m],
                           edgecolor='#555555' if m == 'clam_noopt' else '#7B1B3A',
                           label=LABEL[m]) for m in ALL_MODELS]


# ── Per-task bar charts ───────────────────────────────────────────────────────

def plot_per_task(df: pd.DataFrame, output_dir: Path) -> dict:
    """Generate one EPS per task; return figs_data for combined figure."""
    figs_data = {}
    for task in TASK_ORDER:
        task_df  = df[df['task'] == task]
        classes  = sorted(task_df['cls_id'].unique())
        x_labels = [CLS_DISPLAY.get((task, c), str(c)) for c in classes]

        fig, axes = plt.subplots(1, 2, figsize=(max(9, len(classes) * 2.4), 5),
                                 sharey=False)
        _fill_ax(axes[0], df, task, classes, x_labels, 'none_tcga',
                 'TCGA (train)', fontsize_val=6)
        _fill_ax(axes[1], df, task, classes, x_labels, 'none_cptac',
                 'CPTAC (test)', fontsize_val=6)

        fig.legend(handles=_legend_patches(), loc='lower center', ncol=4,
                   fontsize=10, framealpha=0.8,
                   bbox_to_anchor=(0.5, -0.06), edgecolor='#C47A9A')
        plt.tight_layout(rect=[0, 0.08, 1, 1])

        out = output_dir / f'barplot_{task}.eps'
        fig.savefig(out, format='eps', bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f'Saved → {out}')

        figs_data[task] = (classes, x_labels)
    return figs_data


# ── Combined figure ───────────────────────────────────────────────────────────

def plot_combined(df: pd.DataFrame, figs_data: dict, output_dir: Path) -> None:
    """
    Combined figure:
      Row 0 — PAM50: TCGA (cols 0-3) | CPTAC (cols 4-7)
      Row 1 — ER (cols 0-1), PR (cols 3-4), HER2 (cols 6-7); cols 2,5 = spacers
    """
    fig_all = plt.figure(figsize=(20, 9))
    gs = gridspec.GridSpec(2, 8, figure=fig_all,
                           hspace=0.45, wspace=0.30,
                           width_ratios=[2, 2, 0.5, 2, 2, 0.5, 2, 2])

    # Row 0: PAM50
    if 'pam50' in figs_data:
        classes, x_labels = figs_data['pam50']
        ax_t = fig_all.add_subplot(gs[0, :4])
        ax_c = fig_all.add_subplot(gs[0, 4:])
        _fill_ax(ax_t, df, 'pam50', classes, x_labels, 'none_tcga',
                 f'{TASK_DISPLAY["pam50"]} — TCGA')
        _fill_ax(ax_c, df, 'pam50', classes, x_labels, 'none_cptac',
                 f'{TASK_DISPLAY["pam50"]} — CPTAC')

    # Row 1: ER, PR, HER2
    binary_tasks  = ['er', 'pr', 'erbb2']
    col_starts    = [0, 3, 6]
    for task, col_start in zip(binary_tasks, col_starts):
        if task not in figs_data:
            continue
        classes, x_labels = figs_data[task]
        ax_t = fig_all.add_subplot(gs[1, col_start])
        ax_c = fig_all.add_subplot(gs[1, col_start + 1])
        _fill_ax(ax_t, df, task, classes, x_labels, 'none_tcga',
                 f'{TASK_DISPLAY[task]} — TCGA')
        _fill_ax(ax_c, df, task, classes, x_labels, 'none_cptac',
                 f'{TASK_DISPLAY[task]} — CPTAC')

    fig_all.legend(handles=_legend_patches(), loc='lower center', ncol=4,
                   fontsize=10, framealpha=0.8,
                   bbox_to_anchor=(0.5, -0.02), edgecolor='#C47A9A')

    for ext in ('eps', 'png'):
        out = output_dir / f'barplot_all.{ext}'
        fig_all.savefig(out, format=ext, bbox_inches='tight', facecolor='white')
        print(f'Saved → {out}')
    plt.close(fig_all)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(metrics_csv: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(metrics_csv)

    figs_data = plot_per_task(df, output_dir)
    plot_combined(df, figs_data, output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate paper figures.')
    parser.add_argument('--metrics',    default='results/per_class_metrics.csv')
    parser.add_argument('--output_dir', default='results/figures')
    args = parser.parse_args()
    main(Path(args.metrics), Path(args.output_dir))