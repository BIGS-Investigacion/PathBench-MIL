#!/usr/bin/env python3
"""
Visualización de los cuatro análisis univariantes:
  RPD ~ δn    (variabilidad de tinción)
  RPD ~ Δp    (desplazamiento de prevalencia)
  RPD ~ d     (distancia en espacio de embeddings)
  RPD ~ B̃_c  (separabilidad morfológica)

Uso:
  python scripts/plot_univariates.py [ruta_results_dir]
"""

import sys
from itertools import groupby
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# ── Importar funciones del script principal ────────────────────────────────────
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from generate_results_table import (
    RESULTS_DIR,
    PREVALENCE_SHIFT,
    compute_rows,
    load_centroid_distances,
    load_morphological_separability,
)

# ── Paleta de colores por tarea ────────────────────────────────────────────────
TASK_COLORS = {
    'PAM50': '#e07b39',
    'ER':    '#4c8be2',
    'PR':    '#52a85a',
    'HER2': '#9b59b6',
}

TASK_MARKERS = {
    'PAM50': 'o',
    'ER':    's',
    'PR':    '^',
    'HER2': 'D',
}

# Etiquetas cortas para anotar puntos
CLASS_LABELS = {
    ('PAM50', 0): 'Basal',
    ('PAM50', 1): 'HER2-e',
    ('PAM50', 2): 'LumA',
    ('PAM50', 3): 'LumB',
    ('PAM50', 4): 'Norm',
    ('ER',    0): 'ER-',
    ('ER',    1): 'ER+',
    ('PR',    0): 'PR-',
    ('PR',    1): 'PR+',
    ('HER2', 0): 'HER2-',
    ('HER2', 1): 'HER2+',
}


def _regression_band(x, y, x_line):
    """Devuelve y_hat, lower CI y upper CI para x_line dado un ajuste OLS."""
    n = len(x)
    slope, intercept, r_value, p_value, se_slope = stats.linregress(x, y)
    y_hat_line = intercept + slope * x_line

    # Error estándar de la predicción media
    x_mean = np.mean(x)
    ss_xx   = np.sum((x - x_mean) ** 2)
    ss_res  = np.sum((y - (intercept + slope * x)) ** 2)
    se_res  = np.sqrt(ss_res / (n - 2))

    se_pred = se_res * np.sqrt(1 / n + (x_line - x_mean) ** 2 / ss_xx)
    t_crit  = stats.t.ppf(0.975, df=n - 2)

    r2 = r_value ** 2
    rho, p_rho = stats.spearmanr(x, y)

    return y_hat_line, y_hat_line - t_crit * se_pred, y_hat_line + t_crit * se_pred, \
           slope, intercept, r2, p_value, rho, p_rho


def _scatter_panel(ax, x_vals, y_vals, tasks, cls_ids,
                   xlabel, title, show_ci=True, ylabel='RPD'):
    """Dibuja un panel scatter con regresión y banda de confianza."""
    x = np.array(x_vals)
    y = np.array(y_vals)

    # Línea de regresión
    x_line = np.linspace(x.min() - 0.05 * np.ptp(x),
                         x.max() + 0.05 * np.ptp(x), 200)
    y_line, ci_low, ci_high, slope, intercept, r2, p_ols, rho, p_rho = \
        _regression_band(x, y, x_line)

    if show_ci:
        ax.fill_between(x_line, ci_low, ci_high, alpha=0.15, color='grey',
                        label='IC 95%')
    ax.plot(x_line, y_line, color='black', lw=1.5, zorder=3)

    # Puntos
    for xi, yi, task, cid in zip(x, y, tasks, cls_ids):
        color  = TASK_COLORS.get(task, 'grey')
        marker = TASK_MARKERS.get(task, 'o')
        ax.scatter(xi, yi, color=color, marker=marker, s=70, zorder=4,
                   edgecolors='white', linewidths=0.5)
        label = CLASS_LABELS.get((task, cid), f'{task}-{cid}')
        ax.annotate(label, (xi, yi),
                    textcoords='offset points', xytext=(5, 4),
                    fontsize=7.5, color=color)

    # Anotación estadística
    sig_ols = '***' if p_ols < 0.001 else ('**' if p_ols < 0.01 else
               ('*' if p_ols < 0.05 else 'ns'))
    sig_rho = '***' if p_rho < 0.001 else ('**' if p_rho < 0.01 else
               ('*' if p_rho < 0.05 else 'ns'))
    stat_text = (f'$R^2={r2:.3f}$, $p_{{OLS}}={p_ols:.3f}$ ({sig_ols})\n'
                 f'$\\rho={rho:+.3f}$, $p_\\rho={p_rho:.3f}$ ({sig_rho})')
    ax.text(0.03, 0.97, stat_text, transform=ax.transAxes,
            va='top', ha='left', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7, ec='lightgrey'))

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    # Ajustar ylim al rango de los datos con margen del 20%
    y_pad = 0.20 * np.ptp(y) if np.ptp(y) > 0 else 0.1
    ax.set_ylim(y.min() - y_pad, y.max() + y_pad)
    ax.axhline(0, color='lightgrey', lw=0.8, ls='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def build_dataset(results_dir):
    """Reúne los datos necesarios para los cuatro paneles."""
    rows = compute_rows(results_dir)

    key_fn = lambda r: (r['task'], r['cls_id'], r['cls_name'])
    rows_mean = []
    for (task, cls_id, cls_name), group in groupby(
            sorted(rows, key=key_fn), key=key_fn):
        group = list(group)
        valid_rpd = [r['rpd'] for r in group if not np.isnan(r['rpd'])]
        rows_mean.append({
            'task':     task,
            'cls_id':   cls_id,
            'cls_name': cls_name,
            'delta_n':  np.mean([r['delta_n'] for r in group]),
            'rpd':      np.mean(valid_rpd) if valid_rpd else float('nan'),
        })

    centroid_dists = load_centroid_distances(top_k=8)
    morph_sep      = load_morphological_separability()

    records = []
    for r in rows_mean:
        task   = r['task']
        cid    = r['cls_id']
        rpd    = r['rpd']
        if np.isnan(rpd):
            continue
        dp   = PREVALENCE_SHIFT.get((task, cid))
        dist = centroid_dists.get((task, cid))
        bs   = morph_sep.get((task, cid))
        records.append({
            'task':    'HER2' if task == 'ERBB2' else task,
            'cls_id':  cid,
            'rpd':     rpd,
            'delta_n': r['delta_n'],
            'dp':      dp,
            'dist':    dist,
            'bs':      bs,
        })
    return records


def main():
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS_DIR
    records = build_dataset(results_dir)

    # ── Extraer vectores por análisis (sólo filas con dato completo) ──────────
    def _select(field):
        sel = [(r['rpd'], r[field], r['task'], r['cls_id'])
               for r in records if r[field] is not None]
        y   = [s[0] for s in sel]
        x   = [s[1] for s in sel]
        tsk = [s[2] for s in sel]
        cid = [s[3] for s in sel]
        return x, y, tsk, cid

    x_dn,  y_dn,  t_dn,  c_dn  = _select('delta_n')
    x_dp,  y_dp,  t_dp,  c_dp  = _select('dp')
    x_d,   y_d,   t_d,   c_d   = _select('dist')
    x_bs,  y_bs,  t_bs,  c_bs  = _select('bs')

    # Para δn ~ B̃_c necesitamos filas con ambos campos
    dn_bs_records = [(r['bs'], r['delta_n'], r['task'], r['cls_id'])
                     for r in records if r['bs'] is not None]
    x_dn_bs = [s[0] for s in dn_bs_records]
    y_dn_bs = [s[1] for s in dn_bs_records]
    t_dn_bs = [s[2] for s in dn_bs_records]
    c_dn_bs = [s[3] for s in dn_bs_records]

    # Para d_c ~ B̃_c
    dc_bs_records = [(r['bs'], r['dist'], r['task'], r['cls_id'])
                     for r in records if r['bs'] is not None and r['dist'] is not None]
    x_dc_bs = [s[0] for s in dc_bs_records]
    y_dc_bs = [s[1] for s in dc_bs_records]
    t_dc_bs = [s[2] for s in dc_bs_records]
    c_dc_bs = [s[3] for s in dc_bs_records]

    # ── Cuatro figuras individuales: univariantes RPD ───────────────────────
    legend_handles = [
        mpatches.Patch(color=TASK_COLORS[t], label=t)
        for t in ['PAM50', 'ER', 'PR', 'HER2']
    ]

    panels = [
        ('rpd_vs_delta_n',  x_dn, y_dn, t_dn, c_dn,
         r'$\Delta_n$ (Macenko gain)',          r'RPD ~ $\Delta_n$',    True),
        ('rpd_vs_delta_p',  x_dp, y_dp, t_dp, c_dp,
         r'$\Delta p$ (prevalence shift)',       r'RPD ~ $\Delta p$',    False),
        ('rpd_vs_dc',       x_d,  y_d,  t_d,  c_d,
         r'$d_c$ (cosine centroid distance)',     r'RPD ~ $d_c$',          True),
        ('rpd_vs_morphology', x_bs, y_bs, t_bs, c_bs,
         r'$\tilde{B}_c$ (morphological separability)', r'RPD ~ $\tilde{B}_c$', True),
    ]

    for name, xv, yv, tv, cv, xlabel, title, show_ci in panels:
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.subplots_adjust(bottom=0.18)
        _scatter_panel(ax, xv, yv, tv, cv, xlabel=xlabel, title=title,
                       show_ci=show_ci)
        ax.legend(handles=legend_handles, title='Task', fontsize=8,
                  loc='upper right')
        for ext in ('pdf', 'png'):
            out = results_dir / f'{name}.{ext}'
            fig.savefig(out, bbox_inches='tight', dpi=200)
            print(f'Figura guardada en {out}', file=sys.stderr)
        plt.close(fig)

    # ── Figura combinada 2×2 ──────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.subplots_adjust(hspace=0.38, wspace=0.32)
    for ax, (_, xv, yv, tv, cv, xlabel, title, show_ci) in zip(axes.flat, panels):
        _scatter_panel(ax, xv, yv, tv, cv, xlabel=xlabel, title=title, show_ci=show_ci)
    fig.legend(handles=legend_handles, title='Task', loc='lower center', ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    # no suptitle
    for ext in ('pdf', 'png'):
        out = results_dir / f'univariate_plots.{ext}'
        fig.savefig(out, bbox_inches='tight', dpi=200)
        print(f'Figura guardada en {out}', file=sys.stderr)
    plt.close(fig)

    # ── Figura adicional: δn ~ B̃_c ───────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    fig2.subplots_adjust(bottom=0.18)

    # Reutilizamos _scatter_panel adaptando ylabel
    x = np.array(x_dn_bs)
    y = np.array(y_dn_bs)
    x_line = np.linspace(x.min() - 0.05 * np.ptp(x),
                         x.max() + 0.05 * np.ptp(x), 200)
    y_line, ci_low, ci_high, slope, intercept, r2, p_ols, rho, p_rho = \
        _regression_band(x, y, x_line)

    ax2.fill_between(x_line, ci_low, ci_high, alpha=0.15, color='grey')
    ax2.plot(x_line, y_line, color='black', lw=1.5, zorder=3)

    for xi, yi, task, cid in zip(x, y, t_dn_bs, c_dn_bs):
        color  = TASK_COLORS.get(task, 'grey')
        marker = TASK_MARKERS.get(task, 'o')
        ax2.scatter(xi, yi, color=color, marker=marker, s=70, zorder=4,
                    edgecolors='white', linewidths=0.5)
        label = CLASS_LABELS.get((task, cid), f'{task}-{cid}')
        ax2.annotate(label, (xi, yi),
                     textcoords='offset points', xytext=(5, 4),
                     fontsize=7.5, color=color)

    sig_ols = '***' if p_ols < 0.001 else ('**' if p_ols < 0.01 else
               ('*' if p_ols < 0.05 else 'ns'))
    sig_rho = '***' if p_rho < 0.001 else ('**' if p_rho < 0.01 else
               ('*' if p_rho < 0.05 else 'ns'))
    stat_text = (f'$R^2={r2:.3f}$, $p_{{OLS}}={p_ols:.3f}$ ({sig_ols})\n'
                 f'$\\rho={rho:+.3f}$, $p_\\rho={p_rho:.3f}$ ({sig_rho})')
    ax2.text(0.03, 0.97, stat_text, transform=ax2.transAxes,
             va='top', ha='left', fontsize=8,
             bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7, ec='lightgrey'))

    ax2.axhline(0, color='lightgrey', lw=0.8, ls='--')
    ax2.set_xlabel(r'$\tilde{B}_c$ (morphological separability)', fontsize=10)
    ax2.set_ylabel(r'$\Delta_n$ (Macenko gain on CPTAC)', fontsize=10)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    legend_handles2 = [
        mpatches.Patch(color=TASK_COLORS[t], label=t)
        for t in ['PAM50', 'ER', 'PR', 'HER2']
    ]
    ax2.legend(handles=legend_handles2, title='Task', fontsize=8,
               loc='upper right')

    out2_pdf = results_dir / 'delta_n_vs_morphology.pdf'
    fig2.savefig(out2_pdf, bbox_inches='tight', dpi=200)
    print(f'Figura guardada en {out2_pdf}', file=sys.stderr)

    out2_png = results_dir / 'delta_n_vs_morphology.png'
    fig2.savefig(out2_png, bbox_inches='tight', dpi=200)
    print(f'Figura guardada en {out2_png}', file=sys.stderr)

    plt.close(fig2)

    # ── Figura individual: d_c ~ B̃_c ─────────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    fig3.subplots_adjust(bottom=0.18)
    _scatter_panel(ax3, x_dc_bs, y_dc_bs, t_dc_bs, c_dc_bs,
                   xlabel=r'$\tilde{B}_c$ (morphological separability)',
                   title='',
                   show_ci=True,
                   ylabel=r'$d_c$ (cosine centroid distance)')
    ax3.legend(handles=legend_handles, title='Task', fontsize=8, loc='upper right')
    for ext in ('pdf', 'png'):
        out3 = results_dir / f'dc_vs_morphology.{ext}'
        fig3.savefig(out3, bbox_inches='tight', dpi=200)
        print(f'Figura guardada en {out3}', file=sys.stderr)
    plt.close(fig3)

    # ── Figura combinada: colinealidad (Δn~B̃_c | d_c~B̃_c) ──────────────────
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(12, 5))
    fig4.subplots_adjust(wspace=0.35, bottom=0.18)
    _scatter_panel(ax4a, x_dn_bs, y_dn_bs, t_dn_bs, c_dn_bs,
                   xlabel=r'$\tilde{B}_c$ (morphological separability)',
                   title='',
                   show_ci=True,
                   ylabel=r'$\Delta_n$ (Macenko gain on CPTAC)')
    _scatter_panel(ax4b, x_dc_bs, y_dc_bs, t_dc_bs, c_dc_bs,
                   xlabel=r'$\tilde{B}_c$ (morphological separability)',
                   title='',
                   show_ci=True,
                   ylabel=r'$d_c$ (cosine centroid distance)')
    fig4.legend(handles=legend_handles, title='Task', loc='lower center',
                ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.04))
    for ext in ('pdf', 'png'):
        out4 = results_dir / f'collinearity_plots.{ext}'
        fig4.savefig(out4, bbox_inches='tight', dpi=200)
        print(f'Figura guardada en {out4}', file=sys.stderr)
    plt.close(fig4)


if __name__ == '__main__':
    main()