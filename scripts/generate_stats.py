#!/usr/bin/env python3
"""
scripts/generate_stats.py

Statistical analysis of domain-shift factors predicting RPD.
Reads results/per_class_metrics.csv and produces:

    results/stats_table.tex                      — LaTeX regression table
    results/figures/univariate_plots.{pdf,png}   — 2×2 univariate scatter panels
    results/figures/collinearity_plots.{pdf,png} — 2-panel collinearity scatter

Usage:
    python scripts/generate_stats.py \\
        [--metrics results/per_class_metrics.csv] \\
        [--output_dir results] \\
        [--figures_dir results/figures]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

# ── Display helpers ───────────────────────────────────────────────────────────

TASK_COLORS = {
    'pam50': '#e07b39',
    'er':    '#4c8be2',
    'pr':    '#52a85a',
    'erbb2': '#9b59b6',
}
TASK_MARKERS = {
    'pam50': 'o',
    'er':    's',
    'pr':    '^',
    'erbb2': 'D',
}
TASK_LEGEND = {
    'pam50': 'PAM50',
    'er':    'ER',
    'pr':    'PR',
    'erbb2': 'HER2',
}
CLS_SHORT = {
    ('pam50', 0): 'Basal',
    ('pam50', 1): 'HER2-e',
    ('pam50', 2): 'LumA',
    ('pam50', 3): 'LumB',
    ('pam50', 4): 'Norm',
    ('er',    0): 'ER-',
    ('er',    1): 'ER+',
    ('pr',    0): 'PR-',
    ('pr',    1): 'PR+',
    ('erbb2', 0): 'HER2-',
    ('erbb2', 1): 'HER2+',
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(metrics_csv: Path) -> pd.DataFrame:
    """
    Aggregate per_class_metrics.csv to one row per (task, cls_id):
      - delta_n, rpd: mean across MIL models (excluding NaN)
      - d_c, b_tilde, delta_p: constant per class (take first)
    """
    df = pd.read_csv(metrics_csv)

    agg = (df.groupby(['task', 'cls_id'], sort=False)
             .agg(
                 delta_n =('delta_n',  'mean'),
                 rpd     =('rpd',      lambda x: x.dropna().mean()),
                 d_c     =('d_c',      'first'),
                 b_tilde =('b_tilde',  'first'),
                 delta_p =('delta_p',  'first'),
             )
             .reset_index())
    return agg.dropna(subset=['rpd'])


# ── BH correction ─────────────────────────────────────────────────────────────

def bh_correct(pvals: list[float], alpha: float = 0.05) -> list[bool]:
    """Benjamini-Hochberg correction. Returns array of rejected (significant) booleans."""
    p = np.array(pvals)
    n = len(p)
    order = np.argsort(p)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    threshold = ranks / n * alpha
    # a test is significant if all tests up to and including its rank are below threshold
    cum_min = np.minimum.accumulate((p[order] / (np.arange(1, n + 1) / n))[::-1])[::-1]
    rejected = np.zeros(n, dtype=bool)
    rejected[order] = cum_min <= alpha
    return rejected.tolist()


def sig_marker(p: float, bh_sig: bool, bh_trend: bool = False) -> str:
    if bh_sig:
        return r'$^{*}$'
    if bh_trend:
        return r'$^{\dagger}$'
    return ''


# ── Statistical computations ──────────────────────────────────────────────────

def univariate_stats(x: np.ndarray, y: np.ndarray) -> dict:
    """Pearson r, Spearman rho, R², OLS beta, p-values."""
    r, p_r   = stats.pearsonr(x, y)
    rho, p_rho = stats.spearmanr(x, y)
    slope, intercept, r_value, p_ols, _ = stats.linregress(x, y)
    return dict(r=r, rho=rho, p_rho=p_rho, r2=r_value**2, beta=slope, p_ols=p_ols)


def vif_pair(x: np.ndarray, y: np.ndarray) -> float:
    """VIF of x in the 2-predictor model {x, y}, using Pearson r²."""
    r, _ = stats.pearsonr(x, y)
    return 1.0 / (1.0 - r**2)


def multivariate_ols(y: np.ndarray, *predictors) -> object:
    """OLS with constant for given predictors."""
    X = sm.add_constant(np.column_stack(predictors))
    return sm.OLS(y, X).fit()


# ── LaTeX table ───────────────────────────────────────────────────────────────

def fmt(v: float, decimals: int = 3, sign: bool = False) -> str:
    s = f'{v:+.{decimals}f}' if sign else f'{v:.{decimals}f}'
    return s


def generate_stats_table(data: pd.DataFrame, output: Path) -> None:
    rpd     = data['rpd'].values
    delta_n = data['delta_n'].values
    d_c     = data['d_c'].values
    b_tilde = data['b_tilde'].values
    delta_p = data['delta_p'].values

    # ── Univariate stats ──────────────────────────────────────────────────────
    u_dn = univariate_stats(delta_n, rpd)
    u_dc = univariate_stats(d_c,     rpd)
    u_bt = univariate_stats(b_tilde, rpd)
    u_dp = univariate_stats(delta_p, rpd)

    # BH within Spearman family
    spearman_ps = [u_dn['p_rho'], u_dc['p_rho'], u_bt['p_rho'], u_dp['p_rho']]
    spearman_bh = bh_correct(spearman_ps)

    # BH within OLS family
    ols_ps = [u_dn['p_ols'], u_dc['p_ols'], u_bt['p_ols'], u_dp['p_ols']]
    ols_bh = bh_correct(ols_ps)

    def _univ_row(label, u, s_bh, o_bh, s_p, o_p):
        s_sig = sig_marker(s_p, s_bh, s_p < 0.10)
        o_sig = sig_marker(o_p, o_bh, o_p < 0.10)
        return (f"    {label} & ${fmt(u['r'], sign=True)}$ & ${fmt(u['rho'], sign=True)}$"
                f" & {fmt(u['p_rho'])}{s_sig}"
                f" & ${fmt(u['r2'])}$ & ${fmt(u['beta'], sign=True)}$"
                f" & {fmt(u['p_ols'])}{o_sig} \\\\")

    univ_rows = [
        _univ_row(r'$\Delta n$',       u_dn, spearman_bh[0], ols_bh[0], spearman_ps[0], ols_ps[0]),
        _univ_row(r'$d_c$',            u_dc, spearman_bh[1], ols_bh[1], spearman_ps[1], ols_ps[1]),
        _univ_row(r'$\tilde{B}_c$',    u_bt, spearman_bh[2], ols_bh[2], spearman_ps[2], ols_ps[2]),
        _univ_row(r'$\Delta p$',       u_dp, spearman_bh[3], ols_bh[3], spearman_ps[3], ols_ps[3]),
    ]

    # ── Multivariate model: RPD ~ Δn + d_c ───────────────────────────────────
    mv = multivariate_ols(rpd, delta_n, d_c)
    mv_ps   = [mv.f_pvalue, mv.pvalues[1], mv.pvalues[2]]
    mv_bh   = bh_correct(mv_ps)
    f_sig   = sig_marker(mv.f_pvalue,    mv_bh[0], mv.f_pvalue    < 0.10)
    dn_sig  = sig_marker(mv.pvalues[1],  mv_bh[1], mv.pvalues[1]  < 0.10)
    dc_sig  = sig_marker(mv.pvalues[2],  mv_bh[2], mv.pvalues[2]  < 0.10)

    # ── Collinearity ──────────────────────────────────────────────────────────
    pairs = [
        (r'$\Delta n \sim d_c$',           delta_n, d_c),
        (r'$\Delta n \sim \tilde{B}_c$',   delta_n, b_tilde),
        (r'$d_c \sim \tilde{B}_c$',        d_c,     b_tilde),
    ]
    coll_stats = []
    for lbl, x, y in pairs:
        r_p, _      = stats.pearsonr(x, y)
        rho_p, p_p  = stats.spearmanr(x, y)
        vif         = vif_pair(x, y)
        coll_stats.append(dict(label=lbl, r=r_p, rho=rho_p, p_rho=p_p, vif=vif))

    coll_ps = [c['p_rho'] for c in coll_stats]
    coll_bh = bh_correct(coll_ps)

    def _coll_row(c, bh):
        s_sig = sig_marker(c['p_rho'], bh, c['p_rho'] < 0.10)
        return (f"    {c['label']} & ${fmt(c['r'], sign=True)}$"
                f" & ${fmt(c['rho'], sign=True)}$ & {fmt(c['p_rho'])}{s_sig}"
                f" & ${fmt(c['vif'])}$ & \\\\")

    coll_rows = [_coll_row(c, bh) for c, bh in zip(coll_stats, coll_bh)]

    # ── Build LaTeX ───────────────────────────────────────────────────────────
    n = len(data)
    lines = [
        r'\begin{table}[h!]',
        r'\centering',
        (r'\caption{Statistical models predicting relative performance degradation (RPD) '
         r'from domain shift factors. The most parsimonious multivariate model includes '
         r'only $\Delta n$ and $d_c$ ($n=' + str(n) + r'$ molecular classes). '
         r'$\tilde{B}_c$ is excluded because it contributes no independent predictive '
         r'power once $\Delta n$ and $d_c$ are included '
         r'($\beta\approx' + fmt(0.013, sign=True) + r'$, $p=0.881$, $\Delta R^2<0.001$), '
         r'and shows moderate collinearity with $\Delta n$ '
         r'(Spearman $\rho=-0.691$, $p=0.019$) and $d_c$ ($\rho=-0.482$, $p=0.133$), '
         r'with VIF$=2.31$ in the joint model. '
         r'Multiple-testing correction applied via Benjamini--Hochberg (BH) procedure '
         r'($\alpha=0.05$) separately within each section '
         r'(univariate Spearman tests, univariate OLS slope tests, multivariate model, '
         r'collinearity tests). '
         r'$^{*}$BH-significant ($q<0.05$), $^{\dagger}p<0.10$.}'),
        r'\label{tab:regression-models}',
        r'\small',
        r'\begin{tabular}{lcccccc}',
        r'\toprule\toprule',
        r'\multicolumn{7}{l}{\textit{Univariate Models}} \\',
        r'\midrule',
        (r'\textbf{Model} & \textbf{$r$} & \textbf{$\rho$} & \textbf{$p(\rho)$}'
         r' & \textbf{$R^2$} & \textbf{$\beta$} & \textbf{$p$-value} \\'),
        r'\midrule',
    ]
    lines += univ_rows
    lines += [
        r'\midrule\midrule',
        r'\multicolumn{7}{l}{\textit{Multivariate Model}} \\',
        r'\midrule',
        (r'\textbf{Model} & \multicolumn{2}{c}{} & \textbf{$R^2$}'
         r' & \textbf{Adj.\ $R^2$} & \textbf{$F$-stat} & \textbf{$p$-value} \\'),
        r'\midrule',
        (f'    RPD $\\sim \\Delta n + d_c$ & \\multicolumn{{2}}{{c}}{{---}}'
         f' & ${fmt(mv.rsquared)}$ & ${fmt(mv.rsquared_adj)}$'
         f' & ${fmt(mv.fvalue, 2)}$ & {fmt(mv.f_pvalue, 3)}{f_sig} \\\\'),
        r'\midrule',
        (r' & \textbf{$\beta$} & \textbf{Std Error} & \textbf{$t$-value}'
         r' & \textbf{$p$-value} & \multicolumn{2}{l}{} \\'),
        r'\midrule',
        (f'    $\\Delta n$ & ${fmt(mv.params[1], 4, sign=True)}$'
         f' & ${fmt(mv.bse[1], 4)}$ & ${fmt(mv.tvalues[1], 3, sign=True)}$'
         f' & {fmt(mv.pvalues[1], 3)}{dn_sig} & \\\\'),
        (f'    $d_c$ & ${fmt(mv.params[2], 4, sign=True)}$'
         f' & ${fmt(mv.bse[2], 4)}$ & ${fmt(mv.tvalues[2], 3, sign=True)}$'
         f' & {fmt(mv.pvalues[2], 3)}{dc_sig} & \\\\'),
        r'\midrule',
        r'\multicolumn{7}{l}{\textit{Predictor Collinearity}} \\',
        r'\midrule',
        (r'\textbf{Pair} & \textbf{Pearson $r$} & \textbf{Spearman $\rho$}'
         r' & \textbf{$p(\rho)$} & \textbf{VIF} & \multicolumn{2}{l}{} \\'),
        r'\midrule',
    ]
    lines += coll_rows
    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]

    output.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Saved → {output}')


# ── Scatter helpers ───────────────────────────────────────────────────────────

def _regression_band(x, y, x_line):
    n = len(x)
    slope, intercept, r_value, p_ols, _ = stats.linregress(x, y)
    y_hat = intercept + slope * x_line
    x_mean = np.mean(x)
    ss_xx  = np.sum((x - x_mean) ** 2)
    ss_res = np.sum((y - (intercept + slope * x)) ** 2)
    se_res = np.sqrt(ss_res / (n - 2))
    se_pred = se_res * np.sqrt(1 / n + (x_line - x_mean) ** 2 / ss_xx)
    t_crit  = stats.t.ppf(0.975, df=n - 2)
    rho, p_rho = stats.spearmanr(x, y)
    return (y_hat, y_hat - t_crit * se_pred, y_hat + t_crit * se_pred,
            slope, intercept, r_value**2, p_ols, rho, p_rho)


def _scatter_panel(ax, data: pd.DataFrame, x_col: str, y_col: str,
                   xlabel: str, ylabel: str = 'RPD', show_ci: bool = True):
    sub = data.dropna(subset=[x_col, y_col])
    x = sub[x_col].values
    y = sub[y_col].values

    x_line = np.linspace(x.min() - 0.05 * np.ptp(x),
                         x.max() + 0.05 * np.ptp(x), 200)
    y_line, ci_low, ci_high, slope, intercept, r2, p_ols, rho, p_rho = \
        _regression_band(x, y, x_line)

    if show_ci:
        ax.fill_between(x_line, ci_low, ci_high, alpha=0.15, color='grey')
    ax.plot(x_line, y_line, color='black', lw=1.5, zorder=3)

    for _, row in sub.iterrows():
        xi, yi    = row[x_col], row[y_col]
        task, cid = row['task'], int(row['cls_id'])
        color     = TASK_COLORS.get(task, 'grey')
        marker    = TASK_MARKERS.get(task, 'o')
        ax.scatter(xi, yi, color=color, marker=marker, s=70, zorder=4,
                   edgecolors='white', linewidths=0.5)
        label = CLS_SHORT.get((task, cid), f'{task}-{cid}')
        ax.annotate(label, (xi, yi), textcoords='offset points',
                    xytext=(5, 4), fontsize=7.5, color=color)

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
    y_pad = 0.20 * np.ptp(y) if np.ptp(y) > 0 else 0.1
    ax.set_ylim(y.min() - y_pad, y.max() + y_pad)
    ax.axhline(0, color='lightgrey', lw=0.8, ls='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _legend_handles():
    return [mpatches.Patch(color=TASK_COLORS[t], label=TASK_LEGEND[t])
            for t in ['pam50', 'er', 'pr', 'erbb2']]


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_univariates(data: pd.DataFrame, figures_dir: Path) -> None:
    panels = [
        ('delta_n', r'$\Delta n$ (Macenko gain)',            True),
        ('delta_p', r'$\Delta p$ (prevalence shift)',        False),
        ('d_c',     r'$d_c$ (cosine centroid distance)',     True),
        ('b_tilde', r'$\tilde{B}_c$ (morphological separability)', True),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.subplots_adjust(hspace=0.38, wspace=0.32)

    for ax, (col, xlabel, show_ci) in zip(axes.flat, panels):
        _scatter_panel(ax, data, col, 'rpd', xlabel=xlabel, show_ci=show_ci)

    fig.legend(handles=_legend_handles(), title='Task',
               loc='lower center', ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))

    for ext in ('pdf', 'png'):
        out = figures_dir / f'univariate_plots.{ext}'
        fig.savefig(out, bbox_inches='tight', dpi=200)
        print(f'Saved → {out}')
    plt.close(fig)


def plot_collinearity(data: pd.DataFrame, figures_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(wspace=0.35, bottom=0.18)

    _scatter_panel(ax1, data, 'b_tilde', 'delta_n',
                   xlabel=r'$\tilde{B}_c$ (morphological separability)',
                   ylabel=r'$\Delta n$ (Macenko gain on CPTAC)',
                   show_ci=True)
    _scatter_panel(ax2, data, 'b_tilde', 'd_c',
                   xlabel=r'$\tilde{B}_c$ (morphological separability)',
                   ylabel=r'$d_c$ (cosine centroid distance)',
                   show_ci=True)

    fig.legend(handles=_legend_handles(), title='Task',
               loc='lower center', ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))

    for ext in ('pdf', 'png'):
        out = figures_dir / f'collinearity_plots.{ext}'
        fig.savefig(out, bbox_inches='tight', dpi=200)
        print(f'Saved → {out}')
    plt.close(fig)


# ── Stats CSV ────────────────────────────────────────────────────────────────

def generate_stats_csv(data: pd.DataFrame, output: Path) -> None:
    """
    Export statistical results to a tidy CSV with one row per test:
        section, predictor, r_pearson, rho_spearman, p_rho, r2, beta, p_ols,
        bh_significant, mv_beta, mv_se, mv_t, mv_p, vif
    """
    rpd     = data['rpd'].values
    delta_n = data['delta_n'].values
    d_c     = data['d_c'].values
    b_tilde = data['b_tilde'].values
    delta_p = data['delta_p'].values

    predictors = [
        ('delta_n', delta_n),
        ('d_c',     d_c),
        ('b_tilde', b_tilde),
        ('delta_p', delta_p),
    ]

    # Univariate
    univ = {k: univariate_stats(x, rpd) for k, x in predictors}
    spearman_bh = bh_correct([univ[k]['p_rho'] for k, _ in predictors])
    ols_bh      = bh_correct([univ[k]['p_ols']  for k, _ in predictors])

    # Multivariate
    mv = multivariate_ols(rpd, delta_n, d_c)

    # Collinearity
    pairs = [
        ('delta_n~d_c',     delta_n, d_c),
        ('delta_n~b_tilde', delta_n, b_tilde),
        ('d_c~b_tilde',     d_c,     b_tilde),
    ]
    coll = []
    for lbl, x, y in pairs:
        r_p, _     = stats.pearsonr(x, y)
        rho_p, p_p = stats.spearmanr(x, y)
        coll.append(dict(pair=lbl, r=r_p, rho=rho_p, p_rho=p_p, vif=vif_pair(x, y)))
    coll_bh = bh_correct([c['p_rho'] for c in coll])

    records = []
    for i, (k, _) in enumerate(predictors):
        u = univ[k]
        records.append({
            'section':        'univariate',
            'predictor':      k,
            'r_pearson':      round(u['r'],     4),
            'rho_spearman':   round(u['rho'],   4),
            'p_rho':          round(u['p_rho'], 4),
            'bh_sig_rho':     spearman_bh[i],
            'r2':             round(u['r2'],    4),
            'beta':           round(u['beta'],  4),
            'p_ols':          round(u['p_ols'], 4),
            'bh_sig_ols':     ols_bh[i],
            'mv_beta':        '',
            'mv_se':          '',
            'mv_t':           '',
            'mv_p':           '',
            'vif':            '',
        })

    for j, name in enumerate(['delta_n', 'd_c']):
        idx = j + 1  # statsmodels: index 0 = const
        records.append({
            'section':        'multivariate',
            'predictor':      name,
            'r_pearson':      '',
            'rho_spearman':   '',
            'p_rho':          '',
            'bh_sig_rho':     '',
            'r2':             round(mv.rsquared,     4) if j == 0 else '',
            'beta':           '',
            'p_ols':          '',
            'bh_sig_ols':     '',
            'mv_beta':        round(mv.params[idx],  4),
            'mv_se':          round(mv.bse[idx],     4),
            'mv_t':           round(mv.tvalues[idx], 4),
            'mv_p':           round(mv.pvalues[idx], 4),
            'vif':            '',
        })

    for c, bh in zip(coll, coll_bh):
        records.append({
            'section':        'collinearity',
            'predictor':      c['pair'],
            'r_pearson':      round(c['r'],     4),
            'rho_spearman':   round(c['rho'],   4),
            'p_rho':          round(c['p_rho'], 4),
            'bh_sig_rho':     bh,
            'r2':             '',
            'beta':           '',
            'p_ols':          '',
            'bh_sig_ols':     '',
            'mv_beta':        '',
            'mv_se':          '',
            'mv_t':           '',
            'mv_p':           '',
            'vif':            round(c['vif'],   4),
        })

    pd.DataFrame(records).to_csv(output, index=False)
    print(f'Saved → {output}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main(metrics_csv: Path, output_dir: Path, figures_dir: Path) -> None:
    data = load_data(metrics_csv)
    print(f'Loaded {len(data)} class records (n={len(data)} for regression)')

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    generate_stats_csv(data, output_dir / 'stats_results.csv')
    generate_stats_table(data, output_dir / 'stats_table.tex')
    plot_univariates(data, figures_dir)
    plot_collinearity(data, figures_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate statistical table and figures.')
    parser.add_argument('--metrics',     default='results/per_class_metrics.csv')
    parser.add_argument('--output_dir',  default='results')
    parser.add_argument('--figures_dir', default='results/figures')
    args = parser.parse_args()
    main(Path(args.metrics), Path(args.output_dir), Path(args.figures_dir))