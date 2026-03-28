#!/usr/bin/env python3
"""
scripts/generate_figures.py

Generates all figures for the paper from results/per_class_metrics.csv.

Figures produced:
    results/figures/barplot_{task}.eps      — per-task bar chart (one panel per class)
    results/figures/barplot_all.{eps,png}   — combined figure (all tasks)

Layout:
    Each panel = one class/label.
    X-axis within each panel = datasets (TCGA, CPTAC).
    Bars within each dataset = MIL models.

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


def _style_ax(ax, metric_lbl):
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['TCGA', 'CPTAC'], fontsize=9)
    ax.set_ylabel(metric_lbl, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='#E8A0B4')
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    for spine in ax.spines.values():
        spine.set_edgecolor('#C47A9A')


def _fill_ax(ax, df: pd.DataFrame, task: str, cls_id: int,
             title: str, fontsize_val: float = 5.5):
    """
    Fill a single axis for one class.
    X-axis: [TCGA, CPTAC].  Bars: MIL models.
    """
    cols = ['none_tcga', 'none_cptac']
    width = 0.18
    x = np.arange(2)

    for i, mil in enumerate(ALL_MODELS):
        values = []
        for col in cols:
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

    ax.set_title(title, fontsize=9, color='#8B1A4A')
    _style_ax(ax, METRIC_LABEL[task])


def _legend_patches():
    return [mpatches.Patch(facecolor=COLOR[m],
                           edgecolor='#555555' if m == 'clam_noopt' else '#7B1B3A',
                           label=LABEL[m]) for m in ALL_MODELS]


# ── Per-task bar charts ───────────────────────────────────────────────────────

def plot_per_task(df: pd.DataFrame, output_dir: Path) -> dict:
    """Generate one EPS per task; return figs_data for combined figure."""
    figs_data = {}
    for task in TASK_ORDER:
        task_df = df[df['task'] == task]
        classes = sorted(task_df['cls_id'].unique())
        n_cls   = len(classes)

        fig, axes = plt.subplots(1, n_cls,
                                 figsize=(max(6, n_cls * 3.2), 5),
                                 sharey=True)
        if n_cls == 1:
            axes = [axes]

        for ax, cls_id in zip(axes, classes):
            label = CLS_DISPLAY.get((task, cls_id), str(cls_id))
            _fill_ax(ax, df, task, cls_id, title=label, fontsize_val=6)

        axes[0].set_ylabel(METRIC_LABEL[task], fontsize=10)
        for ax in axes[1:]:
            ax.set_ylabel('')

        fig.suptitle(TASK_DISPLAY[task], fontsize=12, color='#8B1A4A', y=1.02)
        fig.legend(handles=_legend_patches(), loc='lower center', ncol=4,
                   fontsize=10, framealpha=0.8,
                   bbox_to_anchor=(0.5, -0.08), edgecolor='#C47A9A')
        plt.tight_layout(rect=[0, 0.10, 1, 1])

        out = output_dir / f'barplot_{task}.eps'
        fig.savefig(out, format='eps', bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f'Saved → {out}')

        figs_data[task] = classes
    return figs_data


# ── Combined figure ───────────────────────────────────────────────────────────

def plot_combined(df: pd.DataFrame, figs_data: dict, output_dir: Path) -> None:
    """
    Combined figure:
      Row 0 — PAM50: one panel per class (5 panels)
      Row 1 — ER (2), PR (2), HER2 (2) with spacers between tasks
    Each panel: x-axis = TCGA | CPTAC, bars = MIL models.
    """
    fig = plt.figure(figsize=(20, 9))

    # Row 0: PAM50 (5 equal panels)
    gs0 = gridspec.GridSpec(1, 5, figure=fig,
                            left=0.05, right=0.97, top=0.93, bottom=0.54,
                            wspace=0.30)

    # Row 1: ER(2) | spacer | PR(2) | spacer | HER2(2)
    gs1 = gridspec.GridSpec(1, 8, figure=fig,
                            left=0.05, right=0.97, top=0.46, bottom=0.12,
                            wspace=0.30,
                            width_ratios=[2, 2, 0.6, 2, 2, 0.6, 2, 2])

    binary_col_starts = {'er': 0, 'pr': 3, 'erbb2': 6}

    # PAM50
    if 'pam50' in figs_data:
        classes = figs_data['pam50']
        for col, cls_id in enumerate(classes):
            ax = fig.add_subplot(gs0[0, col])
            label = CLS_DISPLAY.get(('pam50', cls_id), str(cls_id))
            _fill_ax(ax, df, 'pam50', cls_id, title=label, fontsize_val=5.5)
            if col > 0:
                ax.set_ylabel('')
        # Task label above row
        fig.text(0.51, 0.96, 'PAM50', ha='center', va='bottom',
                 fontsize=11, color='#8B1A4A', fontweight='bold')

    # Binary tasks
    task_label_x = {'er': 0.16, 'pr': 0.50, 'erbb2': 0.84}
    for task, col_start in binary_col_starts.items():
        if task not in figs_data:
            continue
        classes = figs_data[task]
        for j, cls_id in enumerate(classes):
            ax = fig.add_subplot(gs1[0, col_start + j])
            label = CLS_DISPLAY.get((task, cls_id), str(cls_id))
            _fill_ax(ax, df, task, cls_id, title=label, fontsize_val=5.5)
            if j > 0:
                ax.set_ylabel('')
        fig.text(task_label_x[task], 0.49, TASK_DISPLAY[task],
                 ha='center', va='bottom',
                 fontsize=11, color='#8B1A4A', fontweight='bold')

    fig.legend(handles=_legend_patches(), loc='lower center', ncol=4,
               fontsize=10, framealpha=0.8,
               bbox_to_anchor=(0.5, 0.01), edgecolor='#C47A9A')

    for ext in ('eps', 'png'):
        out = output_dir / f'barplot_all.{ext}'
        fig.savefig(out, format=ext, bbox_inches='tight', facecolor='white')
        print(f'Saved → {out}')
    plt.close(fig)


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