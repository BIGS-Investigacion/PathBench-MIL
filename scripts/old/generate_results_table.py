#!/usr/bin/env python3
"""
Genera tabla LaTeX de resultados desde los ficheros en results/.

Para tareas binarias (ER, ERBB2, PR): extrae PR-AUC por clase.
Para PAM50: extrae F1 por clase.

Métricas derivadas:
  delta_n = mac_CPTAC  - none_CPTAC      (efecto normalización Macenko en test)
  RPD     = (none_CPTAC - none_TCGA) / (-none_TCGA)   (caída relativa de dominio)

Uso:
  python scripts/generate_results_table.py [ruta_results_dir]
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

# ── Configuración ─────────────────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).resolve().parent.parent / 'results'

TASK_METRIC = {
    'er':    'pr_auc',
    'erbb2': 'pr_auc',
    'pr':    'pr_auc',
    'pam50': 'f1',
}

CLASS_LABELS = {
    'er':    {0: 'neg', 1: 'pos'},
    'erbb2': {0: 'neg', 1: 'pos'},
    'pr':    {0: 'neg', 1: 'pos'},
    'pam50': {0: 'Basal', 1: 'Her2', 2: 'LumA', 3: 'LumB', 4: 'Normal'},
}

MIL_DISPLAY = {
    'clam_mil_mb': 'CLAM-MB',
    'dsmil':       'DSMIL',
    'transmil':    'TransMIL',
}

MIL_ORDER  = ['clam_mil_mb', 'dsmil', 'transmil']
TASK_ORDER   = ['er', 'erbb2', 'pr', 'pam50']
TASK_DISPLAY = {'ER': 'ER', 'ERBB2': 'HER2', 'PR': 'PR', 'PAM50': 'PAM50'}


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_results_file(filepath: Path, task: str) -> dict:
    """
    Parsea un fichero de resultados y devuelve:
      {(mil, norm, dataset, cls_id): metric_value}
    donde dataset es 'cptac' o 'tcga'.
    """
    metric = TASK_METRIC[task]
    data   = {}

    with open(filepath, encoding='utf-8') as fh:
        lines = fh.readlines()

    current_mil     = None
    current_norm    = None
    current_dataset = None
    in_classif_rep  = False

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        # ── Cabecera MIL / Norm ───────────────────────────────────────────
        m = re.match(r'MIL:\s+(\S+)\s+\|\s+Norm:\s+(\S+)', line)
        if m:
            current_mil     = m.group(1)
            current_norm    = m.group(2)
            current_dataset = None
            in_classif_rep  = False
            i += 1
            continue

        # ── Cabecera Dataset ──────────────────────────────────────────────
        m = re.match(r'Dataset:\s+(\S+)\s+\|', line)
        if m:
            current_dataset = m.group(1).lower()
            in_classif_rep  = False
            i += 1
            continue

        # ── Bloque PR-AUC por clase ───────────────────────────────────────
        if metric == 'pr_auc' and line.strip() == 'PR-AUC por clase:':
            i += 1
            while i < len(lines):
                m2 = re.match(r'\s+(\d+)\(\d+\):\s+([\d.]+)', lines[i])
                if m2:
                    cls_id = int(m2.group(1))
                    value  = float(m2.group(2))
                    key    = (current_mil, current_norm, current_dataset, cls_id)
                    data[key] = value
                    i += 1
                else:
                    break
            continue

        # ── F1 desde Classification Report ───────────────────────────────
        if metric == 'f1':
            if 'Classification Report' in line:
                in_classif_rep = True
                i += 1
                continue

            if in_classif_rep:
                # Línea de clase: "        0(0)      0.xxx  0.xxx  0.xxx  nnn"
                m2 = re.match(
                    r'\s+(\d+)\(\d+\)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\d+',
                    line)
                if m2:
                    cls_id = int(m2.group(1))
                    f1     = float(m2.group(4))   # columna f1-score
                    key    = (current_mil, current_norm, current_dataset, cls_id)
                    data[key] = f1
                elif re.match(r'\s*accuracy\b', line):
                    in_classif_rep = False   # fin del bloque
                # líneas en blanco y cabecera (precision/recall) se ignoran

        i += 1

    return data


# ── Cómputo de métricas derivadas ─────────────────────────────────────────────

def compute_rows(results_dir: Path) -> list:
    rows = []

    for task in TASK_ORDER:
        filepath = results_dir / f'{task}_results.txt'
        if not filepath.exists():
            print(f'[WARN] {filepath} no encontrado — omitido', file=sys.stderr)
            continue

        data         = parse_results_file(filepath, task)
        class_labels = CLASS_LABELS[task]

        for mil in MIL_ORDER:
            for cls_id, cls_name in class_labels.items():
                none_cptac = data.get((mil, 'none',    'cptac', cls_id))
                none_tcga  = data.get((mil, 'none',    'tcga',  cls_id))
                mac_cptac  = data.get((mil, 'macenko', 'cptac', cls_id))

                if any(v is None for v in [none_cptac, none_tcga, mac_cptac]):
                    print(f'[WARN] dato faltante: task={task} mil={mil} '
                          f'cls={cls_id}', file=sys.stderr)
                    continue

                delta_n = mac_cptac - none_cptac
                rpd     = ((none_cptac - none_tcga) / (-none_tcga)
                           if none_tcga != 0 else float('nan'))

                rows.append(dict(
                    task=task.upper(),
                    mil=mil,
                    cls_id=cls_id,
                    cls_name=cls_name,
                    delta_n=delta_n,
                    rpd=rpd,
                    none_cptac=none_cptac,
                    none_tcga=none_tcga,
                    mac_cptac=mac_cptac,
                ))

    return rows


# ── Generación LaTeX ──────────────────────────────────────────────────────────

def _fmt_delta(v: float) -> str:
    return f'${v:+.3f}$'


def _fmt_rpd(v: float) -> str:
    if abs(v) < 1e-9:
        return r'$\phantom{+}0.000$'
    if v >= 1.0 - 1e-9:
        return r'$+1.000$'
    return f'${v:+.3f}$'


def generate_latex(rows: list) -> str:
    # Organizar: mil → task → [filas ordenadas por cls_id]
    mil_task = defaultdict(lambda: defaultdict(list))
    for r in rows:
        mil_task[r['mil']][r['task']].append(r)

    for mil in mil_task:
        for task in mil_task[mil]:
            mil_task[mil][task].sort(key=lambda r: r['cls_id'])

    # Número de filas por MIL (para \multirow)
    mil_nrows = {mil: sum(len(v) for v in mil_task[mil].values())
                 for mil in MIL_ORDER}

    buf = []

    buf.append(r'\begin{table}[ht]')
    buf.append(r'\centering')
    buf.append(
        r'\caption{Comparison of MIL aggregators across tasks and classes. '
        r'PR-AUC is reported for binary tasks (ER, HER2, PR) and F1 for PAM50. '
        r'\textbf{none}: PR-AUC or F1 on CPTAC without stain normalisation. '
        r'\textbf{macenko}: same metric with Macenko stain normalisation. '
        r'$\Delta n = \text{macenko} - \text{none}$ on CPTAC.}'
    )
    buf.append(r'\label{tab:results_summary}')
    buf.append(r'\begin{tabular}{lllrrr}')
    buf.append(r'\toprule')
    buf.append(r'\textbf{MIL} & \textbf{Task} & \textbf{Class} '
               r'& \textbf{none} & \textbf{macenko} & $\Delta n$ \\')
    buf.append(r'\midrule')

    for mil_idx, mil in enumerate(MIL_ORDER):
        task_dict  = mil_task[mil]
        n_mil_rows = mil_nrows[mil]
        mil_disp   = MIL_DISPLAY[mil]

        tasks_present = [t.upper() for t in TASK_ORDER if t.upper() in task_dict]

        first_mil = True

        for task_idx, task in enumerate(tasks_present):
            task_rows  = task_dict[task]
            n_task     = len(task_rows)
            first_task = True

            for row in task_rows:
                cls_label = f"{row['cls_name']}\\,({row['cls_id']})"
                none_str  = f"{row['none_cptac']:.3f}"
                mac_str   = f"{row['mac_cptac']:.3f}"
                delta_str = _fmt_delta(row['delta_n'])

                mil_col  = (f'\\multirow{{{n_mil_rows}}}{{*}}{{{mil_disp}}}'
                            if first_mil else '')
                task_col = (f'\\multirow{{{n_task}}}{{*}}{{{TASK_DISPLAY.get(task, task)}}}'
                            if first_task else '')

                buf.append(f' {mil_col} & {task_col} & {cls_label}'
                           f' & {none_str} & {mac_str} & {delta_str} \\\\')

                first_mil  = False
                first_task = False

            if task_idx < len(tasks_present) - 1:
                buf.append(r'\cmidrule{2-6}')

        if mil_idx < len(MIL_ORDER) - 1:
            buf.append(r'\midrule')

    buf.append(r'\bottomrule')
    buf.append(r'\end{tabular}')
    buf.append(r'\end{table}')

    return '\n'.join(buf)


# ── Análisis univariante RPD ~ Δn ────────────────────────────────────────────

def univariate_analysis(rows: list) -> None:
    """
    Regresión lineal simple OLS:  RPD = β0 + β1·Δn + ε

    Se excluyen filas con RPD = NaN o RPD = 1.0 exacto
    (colapso total, p.ej. Her2 en CPTAC con F1=0 para todos los modelos).
    """
    valid = [r for r in rows if not np.isnan(r['rpd'])]

    x = np.array([r['delta_n'] for r in valid])
    y = np.array([r['rpd']     for r in valid])
    n = len(x)

    # ── Regresión OLS ────────────────────────────��────────────────────────
    slope, intercept, r_value, p_value, se_slope = stats.linregress(x, y)

    r2     = r_value ** 2
    y_hat  = intercept + slope * x
    ss_res = np.sum((y - y_hat) ** 2)
    se_res = np.sqrt(ss_res / (n - 2))

    t_crit  = stats.t.ppf(0.975, df=n - 2)
    ci_low  = slope - t_crit * se_slope
    ci_high = slope + t_crit * se_slope

    # ── Correlación de Spearman ───────────────────────────────────────────
    rho, p_spearman = stats.spearmanr(x, y)

    # ── Effect sizes ──────────────────────────────────────────────────────
    # Pearson r  (Cohen 1988: pequeño=0.10, mediano=0.30, grande=0.50)
    r_pearson = r_value

    # Cohen's f²  (pequeño=0.02, mediano=0.15, grande=0.35)
    f2 = r2 / (1 - r2) if r2 < 1.0 else float('inf')

    # η² (eta cuadrado): en regresión simple = R²
    eta2 = r2

    # Cohen's d equivalente a partir de r: d = 2r / sqrt(1-r²)
    cohen_d = 2 * r_pearson / np.sqrt(1 - r_pearson ** 2) if abs(r_pearson) < 1 else float('inf')

    def _label_r(r):
        r = abs(r)
        if r >= 0.50: return 'grande'
        if r >= 0.30: return 'mediano'
        if r >= 0.10: return 'pequeño'
        return 'despreciable'

    def _label_f2(f):
        if f >= 0.35: return 'grande'
        if f >= 0.15: return 'mediano'
        if f >= 0.02: return 'pequeño'
        return 'despreciable'

    def _label_d(d):
        d = abs(d)
        if d >= 0.80: return 'grande'
        if d >= 0.50: return 'mediano'
        if d >= 0.20: return 'pequeño'
        return 'despreciable'

    # ── Impresión ─────────────────────────────────────────────────────────
    sep = '─' * 60
    print(sep)
    print('ANÁLISIS UNIVARIANTE:  RPD ~ Δn')
    print(sep)
    print(f'  N (filas válidas)    : {n}  '
          f'(excluidas por NaN: {len(rows) - n})')
    print()
    print('  Regresión OLS')
    print(f'    Intercepto (β0)    : {intercept:+.4f}')
    print(f'    Pendiente  (β1)    : {slope:+.4f}  '
          f'(IC 95%: [{ci_low:+.4f}, {ci_high:+.4f}])')
    print(f'    SE residual        : {se_res:.4f}')
    print(f'    R²                 : {r2:.4f}')
    print(f'    p-valor (β1=0)     : {p_value:.4e}')
    print()
    print('  Correlación de Spearman')
    print(f'    ρ                  : {rho:+.4f}')
    print(f'    p-valor            : {p_spearman:.4e}')
    print()
    print('  Effect sizes')
    print(f'    Pearson r          : {r_pearson:+.4f}  → {_label_r(r_pearson)}')
    print(f'    η² (= R²)          :  {eta2:.4f}  → {_label_r(np.sqrt(eta2))}')
    print(f'    Cohen f²           :  {f2:.4f}  → {_label_f2(f2)}')
    print(f'    Cohen d (equiv.)   : {cohen_d:+.4f}  → {_label_d(cohen_d)}')
    print()

    sig       = p_value < 0.05
    direction = 'negativa' if slope < 0 else 'positiva'
    print('  Interpretación:')
    if sig:
        print(f'    Relación significativa (p={p_value:.3e}).')
        print(f'    Pendiente {direction}: mayor Δn → '
              + ('menor' if slope < 0 else 'mayor') + ' RPD.')
        print(f'    Δn explica el {r2 * 100:.1f}% de la varianza en RPD (R²).')
        print(f'    Tamaño del efecto: r={r_pearson:+.3f} ({_label_r(r_pearson)}), '
              f'f²={f2:.3f} ({_label_f2(f2)}).')
    else:
        print(f'    Relación NO significativa (p={p_value:.3e}, α=0.05).')
        print(f'    Δn no predice linealmente el RPD en esta muestra.')
    print(sep)
    print()

    # ── Correlaciones de Spearman por subgrupo ────────────────────────────
    print('  Correlaciones de Spearman por subgrupo')
    print(f'  {"Subgrupo":<20}  {"N":>3}  {"ρ":>7}  {"p":>10}')
    print('  ' + '─' * 46)

    print('  Por tarea:')
    for g in sorted(set(r['task'] for r in valid)):
        sub = [r for r in valid if r['task'] == g]
        if len(sub) < 3:
            continue
        xi = np.array([r['delta_n'] for r in sub])
        yi = np.array([r['rpd']     for r in sub])
        rho_g, p_g = stats.spearmanr(xi, yi)
        mark = '*' if p_g < 0.05 else ' '
        print(f'    {g:<18}  {len(sub):>3}  {rho_g:>+7.3f}  {p_g:>10.4e} {mark}')
    print()


# ── Diagrama de barras por tarea ─────────────────────────────────────────────

# Datos CLAM sin optimización Optuna (MCCV=train proxy, Hold-out=CPTAC)
CLAM_NOOPT = {
    ('PAM50', 0): (0.807, 0.756),
    ('PAM50', 1): (0.377, 0.000),
    ('PAM50', 2): (0.833, 0.767),
    ('PAM50', 3): (0.549, 0.265),
    ('PAM50', 4): (0.146, 0.000),
    ('ER',    0): (0.590, 0.509),
    ('ER',    1): (0.901, 0.687),
    ('PR',    0): (0.579, 0.694),
    ('PR',    1): (0.793, 0.716),
    ('ERBB2', 0): (0.859, 0.891),
    ('ERBB2', 1): (0.194, 0.109),
}


def plot_bar_charts(rows: list, results_dir: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

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
    METRIC_LABEL = {
        'ER': 'PR-AUC', 'ERBB2': 'PR-AUC', 'PR': 'PR-AUC', 'PAM50': 'F1',
    }

    ALL_MODELS = MIL_ORDER + ['clam_noopt']
    tasks = [t.upper() for t in TASK_ORDER]

    figs_data = []  # collect per-task data for combined figure
    for task in tasks:
        task_rows = [r for r in rows if r['task'] == task]
        if not task_rows:
            continue

        classes   = sorted(set(r['cls_id'] for r in task_rows))
        cls_names = {r['cls_id']: r['cls_name'] for r in task_rows}
        x_labels  = [cls_names[c] for c in classes]
        n_cls     = len(classes)
        n_all     = len(ALL_MODELS)
        width     = 0.18
        x         = np.arange(n_cls)

        fig, axes = plt.subplots(1, 2, figsize=(max(9, n_cls * 2.4), 5),
                                 sharey=False)
        # no title

        for ax, (dataset_key, dataset_label) in zip(
                axes, [('none_tcga',  'TCGA (train)'),
                       ('none_cptac', 'CPTAC (test)')]):

            for i, mil in enumerate(ALL_MODELS):
                values = []
                for cls_id in classes:
                    if mil == 'clam_noopt':
                        mccv_val, ho_val = CLAM_NOOPT.get((task, cls_id), (0.0, 0.0))
                        val = mccv_val if dataset_key == 'none_tcga' else ho_val
                    else:
                        match = [r for r in task_rows
                                 if r['mil'] == mil and r['cls_id'] == cls_id]
                        val = match[0][dataset_key] if match else 0.0
                    values.append(val)

                offset = (i - (n_all - 1) / 2) * width
                edge_col = '#555555' if mil == 'clam_noopt' else '#7B1B3A'
                bars = ax.bar(x + offset, values, width,
                              color=COLOR[mil], edgecolor=edge_col,
                              linewidth=0.6, label=LABEL[mil])

                for bar, v in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.01,
                            f'{v:.2f}', ha='center', va='bottom',
                            fontsize=6, color='#333333' if mil == 'clam_noopt' else '#4A0020')

            ax.set_title(dataset_label, fontsize=11, color='#8B1A4A')
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, fontsize=9)
            ax.set_ylabel(METRIC_LABEL[task], fontsize=10)
            ax.set_ylim(0, 1.15)
            ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='#E8A0B4')
            ax.set_axisbelow(True)
            ax.spines[['top', 'right']].set_visible(False)
            for spine in ax.spines.values():
                spine.set_edgecolor('#C47A9A')

        legend_patches = [mpatches.Patch(facecolor=COLOR[m],
                                         edgecolor='#555555' if m == 'clam_noopt' else '#7B1B3A',
                                         label=LABEL[m]) for m in ALL_MODELS]
        fig.legend(handles=legend_patches, loc='lower center', ncol=4,
                   fontsize=10, framealpha=0.8,
                   bbox_to_anchor=(0.5, -0.06),
                   edgecolor='#C47A9A')

        plt.tight_layout(rect=[0, 0.08, 1, 1])
        out = results_dir / f'barplot_{task.lower()}.eps'
        fig.savefig(out, format='eps', bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f'  Guardado: {out}', file=sys.stderr)
        figs_data.append((task, task_rows, classes, cls_names, x_labels))


    # ── Figura combinada: fila 1 = PAM50, fila 2 = ER + PR + HER2 ──────────
    # GridSpec: 2 filas × 8 cols
    # PAM50: cols 0-3 (TCGA) y 4-7 (CPTAC)
    # Binarias: ER cols 0-1, PR cols 3-4, HER2 cols 6-7 (cols 2,5 vacías = separador)
    import matplotlib.gridspec as gridspec
    fig_all = plt.figure(figsize=(20, 9))
    gs = gridspec.GridSpec(2, 8, figure=fig_all,
                           hspace=0.45, wspace=0.30,
                           width_ratios=[2, 2, 0.5, 2, 2, 0.5, 2, 2])

    # Índice de figs_data por tarea
    fd = {d[0]: d for d in figs_data}
    TASK_ORDER_COMBINED = ['PAM50', 'ER', 'PR', 'ERBB2']

    def _fill_ax(ax, task, task_rows, classes, cls_names, x_labels, dataset_key, dataset_label):
        n_all = len(ALL_MODELS)
        width = 0.18
        x = np.arange(len(classes))
        for i, mil in enumerate(ALL_MODELS):
            values = []
            for cls_id in classes:
                if mil == 'clam_noopt':
                    mccv_val, ho_val = CLAM_NOOPT.get((task, cls_id), (0.0, 0.0))
                    val = mccv_val if dataset_key == 'none_tcga' else ho_val
                else:
                    match = [r for r in task_rows if r['mil'] == mil and r['cls_id'] == cls_id]
                    val = match[0][dataset_key] if match else 0.0
                values.append(val)
            offset = (i - (n_all - 1) / 2) * width
            edge_col = '#555555' if mil == 'clam_noopt' else '#7B1B3A'
            bars = ax.bar(x + offset, values, width,
                          color=COLOR[mil], edgecolor=edge_col, linewidth=0.6)
            for bar, v in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01, f'{v:.2f}',
                        ha='center', va='bottom',
                        fontsize=5.5, color='#333333' if mil == 'clam_noopt' else '#4A0020')
        ax.set_title(f'{TASK_DISPLAY.get(task, task)} — {dataset_label}', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_ylabel(METRIC_LABEL[task], fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='#E8A0B4')
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        for spine in ax.spines.values():
            spine.set_edgecolor('#C47A9A')

    # Fila 0: PAM50 (cols 0-2 TCGA, cols 3-5 CPTAC)
    if 'PAM50' in fd:
        _, task_rows, classes, cls_names, x_labels = fd['PAM50']
        ax_t = fig_all.add_subplot(gs[0, :4])
        ax_c = fig_all.add_subplot(gs[0, 4:])
        _fill_ax(ax_t, 'PAM50', task_rows, classes, cls_names, x_labels, 'none_tcga', 'TCGA')
        _fill_ax(ax_c, 'PAM50', task_rows, classes, cls_names, x_labels, 'none_cptac', 'CPTAC')

    # Fila 1: ER (cols 0-1), PR (cols 2-3), HER2/ERBB2 (cols 4-5)
    binary_tasks = [t for t in ['ER', 'PR', 'ERBB2'] if t in fd]
    binary_col_starts = [0, 3, 6]  # col 2 y 5 son separadores vacíos
    for bt_idx, bt in enumerate(binary_tasks):
        _, task_rows, classes, cls_names, x_labels = fd[bt]
        col_start = binary_col_starts[bt_idx]
        ax_t = fig_all.add_subplot(gs[1, col_start])
        ax_c = fig_all.add_subplot(gs[1, col_start + 1])
        _fill_ax(ax_t, bt, task_rows, classes, cls_names, x_labels, 'none_tcga', 'TCGA')
        _fill_ax(ax_c, bt, task_rows, classes, cls_names, x_labels, 'none_cptac', 'CPTAC')

    legend_patches = [mpatches.Patch(facecolor=COLOR[m],
                                     edgecolor='#555555' if m == 'clam_noopt' else '#7B1B3A',
                                     label=LABEL[m]) for m in ALL_MODELS]
    fig_all.legend(handles=legend_patches, loc='lower center', ncol=4,
                   fontsize=10, framealpha=0.8,
                   bbox_to_anchor=(0.5, -0.02), edgecolor='#C47A9A')
    for ext in ('eps', 'png'):
        out_all = results_dir / f'barplot_all.{ext}'
        fig_all.savefig(out_all, format=ext, bbox_inches='tight', facecolor='white')
        print(f'  Guardado: {out_all}', file=sys.stderr)
    plt.close(fig_all)


# ── Datos de prevalencia por clase ────────────────────────────────────────────
# Δp = p_CPTAC - p_TCGA  (positivo → más frecuente en CPTAC)

PREVALENCE_SHIFT: dict[tuple, float] = {
    # task_upper, cls_id → Δp
    ('ER',    0): +0.159,   # ER-negative
    ('ER',    1): -0.159,   # ER-positive
    ('PR',    0): +0.127,   # PR-negative
    ('PR',    1): -0.127,   # PR-positive
    ('ERBB2', 0): +0.049,   # HER2-negative
    ('ERBB2', 1): -0.049,   # HER2-positive
    ('PAM50', 0): +0.131,   # Basal
    ('PAM50', 1): +0.023,   # HER2-enriched
    ('PAM50', 2): -0.071,   # Luminal A
    ('PAM50', 3): +0.085,   # Luminal B (≈ 0.114-0.198 rounded)
    ('PAM50', 4): +0.001,   # Normal-like
}


# ── Carga de separabilidad morfológica ────────────────────────────────────────
# B̃_c = min_{d≠c} B_{c,d} − B_{c,c}   (ecuación del paper)

BIO_DIR = Path(__file__).resolve().parent.parent / 'results'

MORPH_SEP_MATRICES = {
    'PAM50': {
        'csv': 'biological_comparison_BASAL_HER2-ENRICHED_LUMINAL-A_LUMINAL-B_NORMAL-LIKE_matriz.csv',
        'labels': ['BASAL', 'HER2-ENRICHED', 'LUMINAL-A', 'LUMINAL-B', 'NORMAL-LIKE'],
        'cls_ids': [0, 1, 2, 3, 4],
    },
    'ER': {
        'csv': 'biological_comparison_ER-NEGATIVE_ER-POSITIVE_matriz.csv',
        'labels': ['ER-NEGATIVE', 'ER-POSITIVE'],
        'cls_ids': [0, 1],
    },
    'PR': {
        'csv': 'biological_comparison_PR-NEGATIVE_PR-POSITIVE_matriz.csv',
        'labels': ['PR-NEGATIVE', 'PR-POSITIVE'],
        'cls_ids': [0, 1],
    },
    'ERBB2': {
        'csv': 'biological_comparison_HER2-NEGATIVE_HER2-POSITIVE_matriz.csv',
        'labels': ['HER2-NEGATIVE', 'HER2-POSITIVE'],
        'cls_ids': [0, 1],
    },
}


def load_morphological_separability() -> dict:
    """Devuelve {(task_upper, cls_id): B̃_c}."""
    import csv
    result = {}
    for task, info in MORPH_SEP_MATRICES.items():
        p = BIO_DIR / info['csv']
        if not p.exists():
            print(f'[WARN] No encontrado: {p}', file=sys.stderr)
            continue
        with open(p) as f:
            rows_csv = list(csv.DictReader(f))
        n = len(info['labels'])
        mat = [[float(rows_csv[i][col]) for col in info['labels']] for i in range(n)]
        for i, cls_id in enumerate(info['cls_ids']):
            B_c = mat[i][i]
            off_diag = [mat[i][j] for j in range(n) if j != i]
            result[(task, cls_id)] = min(off_diag) - B_c
    return result


# ── Carga de distancias de centroides ─────────────────────────────────────────

PATCH_INTER_DIR = Path(__file__).resolve().parent.parent / 'results' / 'patch_intersection'


def load_centroid_distances(top_k: int = 8) -> dict:
    """Devuelve {(task_upper, cls_id): centroid_dist} como media de los 3 modelos
    (d_mean de patch_intersection_all/), coherente con Δn que también es media de 3."""
    dists = {}
    for task in TASK_ORDER:
        csv_path = PATCH_INTER_ALL_DIR / task / 'centroid_distances_all.csv'
        if not csv_path.exists():
            print(f'[WARN] No encontrado: {csv_path}', file=sys.stderr)
            continue
        import csv
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                dists[(task.upper(), int(row['class']))] = float(row['d_mean'])
    return dists



PATCH_INTER_ALL_DIR = Path(__file__).resolve().parent.parent / 'results' / 'patch_intersection_all'

MIL_DC_COL = {
    'clam_mil_mb': 'd_clam',
    'dsmil':       'd_dsmil',
    'transmil':    'd_transmil',
}

def load_centroid_distances_per_model() -> dict:
    """Devuelve {(mil, task_upper, cls_id): centroid_dist} desde patch_intersection_all/."""
    dists = {}
    for task in TASK_ORDER:
        csv_path = PATCH_INTER_ALL_DIR / task / 'centroid_distances_all.csv'
        if not csv_path.exists():
            continue
        import csv as _csv
        with open(csv_path) as f:
            for row in _csv.DictReader(f):
                cls_id = int(row['class'])
                for mil, col in MIL_DC_COL.items():
                    val = row.get(col)
                    if val:
                        dists[(mil, task.upper(), cls_id)] = float(val)
    return dists

def generate_centroid_latex() -> str:
    """Tabla LaTeX con d_c por modelo, tarea y clase."""
    CLASS_LABELS_DISPLAY = {
        'ER':    {0: 'neg', 1: 'pos'},
        'ERBB2': {0: 'neg', 1: 'pos'},
        'PR':    {0: 'neg', 1: 'pos'},
        'PAM50': {0: 'LumA', 1: 'LumB', 2: 'Basal', 3: 'Her2e', 4: 'Normal'},
    }
    rows_out = []
    for task in TASK_ORDER:
        task_up = task.upper()
        task_disp = TASK_DISPLAY.get(task_up, task_up)
        csv_path = PATCH_INTER_ALL_DIR / task / 'centroid_distances_all.csv'
        if not csv_path.exists():
            continue
        import csv as _csv
        with open(csv_path) as f:
            for row in _csv.DictReader(f):
                cls_id = int(row['class'])
                cls_disp = CLASS_LABELS_DISPLAY.get(task_up, {}).get(cls_id, str(cls_id))
                d_clam     = float(row['d_clam'])
                d_dsmil    = float(row['d_dsmil'])
                d_transmil = float(row['d_transmil'])
                d_mean     = float(row['d_mean'])
                rows_out.append(
                    f'    {task_disp} & {cls_disp} & {d_clam:.3f} & {d_dsmil:.3f}'
                    f' & {d_transmil:.3f} & {d_mean:.3f} \\\\'
                )

    table_lines = [
        r'\begin{table}[ht]',
        r'\centering',
        r'\caption{Cosine centroid distances $d_c$ between TCGA and CPTAC class centroids in Virchow2 feature space, per MIL model.}',
        r'\label{tab:centroid_distances}',
        r'\begin{tabular}{llrrrr}',
        r'\toprule',
        r'Task & Class & CLAM-MB & DSMIL & TransMIL & Mean \\',
        r'\midrule',
    ] + rows_out + [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]
    return '\n'.join(table_lines)



def generate_performance_per_mil_latex(rows: list) -> str:
    """Tabla de rendimiento por modelo MIL (TCGA / CPTAC / RPD)."""
    METRIC_LABEL = {'ER': 'PR-AUC', 'ERBB2': 'PR-AUC', 'PR': 'PR-AUC', 'PAM50': 'F1'}
    MIL_DISP = {'clam_mil_mb': 'CLAM-MB', 'dsmil': 'DSMIL', 'transmil': 'TransMIL'}

    from collections import defaultdict as _dd
    mil_task = _dd(lambda: _dd(list))
    for r in rows:
        mil_task[r['mil']][r['task']].append(r)
    for mil in mil_task:
        for task in mil_task[mil]:
            mil_task[mil][task].sort(key=lambda r: r['cls_id'])

    buf = []
    buf.append(r'\begin{table}[ht]')
    buf.append(r'\centering')
    buf.append(
        r'\caption{Performance per MIL aggregator. '
        r'TCGA: MCCV score (train/val, no stain normalisation); '
        r'CPTAC: hold-out external test score (no stain normalisation). '
        r'RPD: relative performance drop.}'
    )
    buf.append(r'\label{tab:performance_per_mil}')
    buf.append(r'\begin{tabular}{lllcrrr}')
    buf.append(r'\toprule')
    buf.append(r'\textbf{MIL} & \textbf{Task} & \textbf{Class} & \textbf{Metric} '
               r'& \textbf{TCGA} & \textbf{CPTAC} & \textbf{RPD} \\')
    buf.append(r'\midrule')

    for mil_idx, mil in enumerate(MIL_ORDER):
        task_dict = mil_task[mil]
        mil_disp  = MIL_DISP.get(mil, mil)
        n_mil     = sum(len(v) for v in task_dict.values())
        first_mil = True
        for task in [t.upper() for t in TASK_ORDER]:
            task_rows = task_dict.get(task, [])
            if not task_rows:
                continue
            metric = METRIC_LABEL[task]
            n_task = len(task_rows)
            for i, r in enumerate(task_rows):
                mil_col   = f'\\multirow{{{n_mil}}}{{*}}{{{mil_disp}}}' if first_mil else ''
                task_col  = f'\\multirow{{{n_task}}}{{*}}{{{TASK_DISPLAY.get(task, task)}}}' if i == 0 else ''
                met_col   = f'\\multirow{{{n_task}}}{{*}}{{{metric}}}' if i == 0 else ''
                tcga_val  = f"{r['none_tcga']:.3f}"
                cptac_val = f"{r['none_cptac']:.3f}"
                rpd_str   = '---' if np.isnan(r['rpd']) else f"${r['rpd']:+.3f}$"
                cls_name_ = r['cls_name']
                buf.append(f'  {mil_col} & {task_col} & {cls_name_} & {met_col} '
                           f'& {tcga_val} & {cptac_val} & {rpd_str} \\\\')
                first_mil = False
        if mil_idx < len(MIL_ORDER) - 1:
            buf.append(r'\midrule')

    buf.append(r'\bottomrule')
    buf.append(r'\end{tabular}')
    buf.append(r'\end{table}')
    return '\n'.join(buf)


# ── Tabla resumen: medias de los 3 modelos Opt- + distancia de centroide ──────

def generate_summary_latex(rows_mean: list, centroid_dists: dict,
                           morph_sep: dict = None) -> str:
    METRIC_LABEL = {'ER': 'PR-AUC', 'ERBB2': 'PR-AUC', 'PR': 'PR-AUC', 'PAM50': 'F1'}

    def _task_rows_sorted(task):
        return sorted([r for r in rows_mean if r['task'] == task],
                      key=lambda r: r['cls_id'])

    # ── Tabla 1: TCGA / CPTAC / RPD ──────────────────────────────────────────
    buf1 = []
    buf1.append(r'\begin{table}[ht]')
    buf1.append(r'\centering')
    buf1.append(
        r'\caption{Mean performance across CLAM-MB, DSMIL and TransMIL. '
        r'TCGA: MCCV score (train/val, no stain normalisation); '
        r'CPTAC: hold-out external test score (no stain normalisation). '
        r'RPD: relative performance drop, $\mathrm{RPD}=(\mathrm{CPTAC}-\mathrm{TCGA})\,/\,(-\mathrm{TCGA})$.}'
    )
    buf1.append(r'\label{tab:summary_performance}')
    buf1.append(r'\begin{tabular}{llcrrr}')
    buf1.append(r'\toprule')
    buf1.append(r'\textbf{Task} & \textbf{Class} & \textbf{Metric} & \textbf{TCGA} & \textbf{CPTAC} & \textbf{RPD} \\')
    buf1.append(r'\midrule')

    for task in [t.upper() for t in TASK_ORDER]:
        task_rows = _task_rows_sorted(task)
        if not task_rows:
            continue
        metric = METRIC_LABEL[task]
        n = len(task_rows)
        for i, r in enumerate(task_rows):
            task_col = f'\\multirow{{{n}}}{{*}}{{{TASK_DISPLAY.get(task, task)}}}' if i == 0 else ''
            met_col  = f'\\multirow{{{n}}}{{*}}{{{metric}}}' if i == 0 else ''
            tcga_val  = f"{r['none_tcga_mean']:.3f}"
            cptac_val = f"{r['none_cptac_mean']:.3f}"
            rpd_str   = '---' if np.isnan(r['rpd']) else f"${r['rpd']:+.3f}$"
            cls_name_ = r['cls_name']
            buf1.append(f'  {task_col} & {cls_name_} & {met_col} '
                        f'& {tcga_val} & {cptac_val} & {rpd_str} \\\\')
        buf1.append(r'\midrule' if task != 'PAM50' else r'\bottomrule')

    buf1.append(r'\end{tabular}')
    buf1.append(r'\end{table}')

    # ── Tabla 2: variables independientes + RPD ───────────────────────────────
    buf2 = []
    buf2.append(r'\begin{table}[ht]')
    buf2.append(r'\centering')
    buf2.append(
        r'\caption{Independent variables used in the statistical analysis and RPD (mean across models). '
        r'$\Delta n$: Macenko gain on CPTAC (Macenko$-$none). '
        r'$d_c$: cosine distance between TCGA and CPTAC class centroids in Virchow2 space '
        r'(top-8 attention patches, mean of three MIL models). '
        r'$\Delta p$: prevalence shift ($p_{\mathrm{CPTAC}}-p_{\mathrm{TCGA}}$). '
        r'$\tilde{B}_c$: morphological separability ($\min_{d\neq c}B_{c,d}-B_{c,c}$).}'
    )
    buf2.append(r'\label{tab:summary_predictors}')
    buf2.append(r'\begin{tabular}{llrrrrrr}')
    buf2.append(r'\toprule')
    buf2.append(r'\textbf{Task} & \textbf{Class} & $\Delta n$ & $d_c$ & $\Delta p$ & $\tilde{B}_c$ & \textbf{RPD} \\')
    buf2.append(r'\midrule')

    for task in [t.upper() for t in TASK_ORDER]:
        task_rows = _task_rows_sorted(task)
        if not task_rows:
            continue
        n = len(task_rows)
        for i, r in enumerate(task_rows):
            task_col = f'\\multirow{{{n}}}{{*}}{{{TASK_DISPLAY.get(task, task)}}}' if i == 0 else ''
            dn_str   = f"${r['delta_n']:+.3f}$"
            rpd_str  = '---' if np.isnan(r['rpd']) else f"${r['rpd']:+.3f}$"
            dist     = centroid_dists.get((task, r['cls_id']))
            dist_str = f"{dist:.3f}" if dist is not None else '---'
            dp       = PREVALENCE_SHIFT.get((task, r['cls_id']))
            dp_str   = f'${dp:+.3f}$' if dp is not None else '---'
            bs       = morph_sep.get((task, r['cls_id'])) if morph_sep else None
            bs_str   = f'${bs:+.3f}$' if bs is not None else '---'
            cls_name_ = r['cls_name']
            buf2.append(f'  {task_col} & {cls_name_} '
                        f'& {dn_str} & {dist_str} & {dp_str} & {bs_str} & {rpd_str} \\\\')
        buf2.append(r'\midrule' if task != 'PAM50' else r'\bottomrule')

    buf2.append(r'\end{tabular}')
    buf2.append(r'\end{table}')

    return '\n'.join(buf1) + '\n\n' + '\n'.join(buf2)


# ── Análisis univariante RPD ~ |Δp| ──────────────────────────────────────────

def univariate_prevalence(rows_mean: list) -> None:
    valid = []
    for r in rows_mean:
        dp = PREVALENCE_SHIFT.get((r['task'], r['cls_id']))
        if dp is None or np.isnan(r['rpd']):
            continue
        valid.append({'rpd': r['rpd'], 'dp': dp,
                      'task': r['task'], 'cls_id': r['cls_id']})

    if len(valid) < 4:
        print('[WARN] Insuficientes datos para regresión RPD ~ Δp',
              file=sys.stderr)
        return

    x = np.array([v['dp'] for v in valid])
    y = np.array([v['rpd']    for v in valid])
    n = len(x)

    slope, intercept, r_value, p_value, se_slope = stats.linregress(x, y)
    r2    = r_value ** 2
    y_hat = intercept + slope * x
    ss_res = np.sum((y - y_hat) ** 2)
    se_res = np.sqrt(ss_res / (n - 2))
    t_crit = stats.t.ppf(0.975, df=n - 2)
    ci_low, ci_high = slope - t_crit * se_slope, slope + t_crit * se_slope
    rho, p_spearman = stats.spearmanr(x, y)

    sep = '─' * 60
    print(sep)
    print('ANÁLISIS UNIVARIANTE:  RPD ~ Δp')
    print(sep)
    print(f'  N (filas válidas): {n}')
    print()
    print('  Regresión OLS')
    print(f'    Intercepto (β0): {intercept:+.4f}')
    print(f'    Pendiente  (β1): {slope:+.4f}  '
          f'(IC 95%: [{ci_low:+.4f}, {ci_high:+.4f}])')
    print(f'    SE residual    : {se_res:.4f}')
    print(f'    R²             : {r2:.4f}')
    print(f'    p-valor (β1=0) : {p_value:.4e}')
    print()
    print('  Correlación de Spearman')
    print(f'    ρ              : {rho:+.4f}')
    print(f'    p-valor        : {p_spearman:.4e}')
    print(sep)
    print()


# ── Análisis univariante RPD ~ d(centroide) ───────────────────────────────────

def univariate_centroid(rows_mean: list, centroid_dists: dict) -> None:
    valid = []
    for r in rows_mean:
        dist = centroid_dists.get((r['task'], r['cls_id']))
        if dist is None or np.isnan(r['rpd']):
            continue
        valid.append({'rpd': r['rpd'], 'dist': dist,
                      'task': r['task'], 'cls_id': r['cls_id']})

    if len(valid) < 4:
        print('[WARN] Insuficientes datos para regresión RPD ~ d(centroide)',
              file=sys.stderr)
        return

    x = np.array([v['dist'] for v in valid])
    y = np.array([v['rpd']  for v in valid])
    n = len(x)

    slope, intercept, r_value, p_value, se_slope = stats.linregress(x, y)
    r2    = r_value ** 2
    y_hat = intercept + slope * x
    ss_res = np.sum((y - y_hat) ** 2)
    se_res = np.sqrt(ss_res / (n - 2))
    t_crit = stats.t.ppf(0.975, df=n - 2)
    ci_low, ci_high = slope - t_crit * se_slope, slope + t_crit * se_slope
    rho, p_spearman = stats.spearmanr(x, y)

    sep = '─' * 60
    print(sep)
    print('ANÁLISIS UNIVARIANTE:  RPD ~ d(centroide)')
    print(sep)
    print(f'  N (filas válidas): {n}')
    print()
    print('  Regresión OLS')
    print(f'    Intercepto (β0): {intercept:+.4f}')
    print(f'    Pendiente  (β1): {slope:+.4f}  '
          f'(IC 95%: [{ci_low:+.4f}, {ci_high:+.4f}])')
    print(f'    SE residual    : {se_res:.4f}')
    print(f'    R²             : {r2:.4f}')
    print(f'    p-valor (β1=0) : {p_value:.4e}')
    print()
    print('  Correlación de Spearman')
    print(f'    ρ              : {rho:+.4f}')
    print(f'    p-valor        : {p_spearman:.4e}')
    print(sep)
    print()


# ── Análisis univariante d(centroide) ~ B̃_c ──────────────────────────────────

def univariate_d_vs_morph(rows_mean: list, centroid_dists: dict,
                           morph_sep: dict) -> None:
    valid = []
    for r in rows_mean:
        dist = centroid_dists.get((r['task'], r['cls_id']))
        bs   = morph_sep.get((r['task'], r['cls_id']))
        if dist is None or bs is None:
            continue
        valid.append({'dist': dist, 'bs': bs,
                      'task': r['task'], 'cls_id': r['cls_id']})

    if len(valid) < 4:
        print('[WARN] Insuficientes datos para regresión d ~ B̃_c', file=sys.stderr)
        return

    x = np.array([v['bs']   for v in valid])
    y = np.array([v['dist'] for v in valid])
    n = len(x)

    slope, intercept, r_value, p_value, se_slope = stats.linregress(x, y)
    r2    = r_value ** 2
    y_hat = intercept + slope * x
    se_res = np.sqrt(np.sum((y - y_hat) ** 2) / (n - 2))
    t_crit = stats.t.ppf(0.975, df=n - 2)
    ci_low, ci_high = slope - t_crit * se_slope, slope + t_crit * se_slope
    rho, p_spearman = stats.spearmanr(x, y)

    sep = '─' * 60
    print(sep)
    print('ANÁLISIS UNIVARIANTE:  d(centroide) ~ B̃_c')
    print(sep)
    print(f'  N (filas válidas): {n}')
    print()
    print('  Datos:')
    print(f'  {"Task":<8} {"Cls":<8} {"B̃_c":>7} {"d":>7}')
    for v in valid:
        print(f'    {v["task"]:<8} {v["cls_id"]:<8} {v["bs"]:>+7.3f} {v["dist"]:>7.3f}')
    print()
    print('  Regresión OLS  (X = B̃_c,  Y = d)')
    print(f'    Intercepto (β0): {intercept:+.4f}')
    print(f'    Pendiente  (β1): {slope:+.4f}  '
          f'(IC 95%: [{ci_low:+.4f}, {ci_high:+.4f}])')
    print(f'    SE residual    : {se_res:.4f}')
    print(f'    R²             : {r2:.4f}')
    print(f'    p-valor (β1=0) : {p_value:.4e}')
    print()
    print('  Correlación de Spearman')
    print(f'    ρ              : {rho:+.4f}')
    print(f'    p-valor        : {p_spearman:.4e}')
    print(sep)
    print()


# ── Análisis univariante RPD ~ B̃_c ───────────────────────────────────────────

def univariate_rpd_vs_morph(rows_mean: list, morph_sep: dict) -> None:
    valid = []
    for r in rows_mean:
        bs  = morph_sep.get((r['task'], r['cls_id']))
        rpd = r.get('rpd')
        if bs is None or rpd is None:
            continue
        valid.append({'bs': bs, 'rpd': rpd,
                      'task': r['task'], 'cls_id': r['cls_id']})

    if len(valid) < 4:
        print('[WARN] Insuficientes datos para regresión RPD ~ B̃_c', file=sys.stderr)
        return

    x = np.array([v['bs']  for v in valid])
    y = np.array([v['rpd'] for v in valid])
    n = len(x)

    slope, intercept, r_value, p_value, se_slope = stats.linregress(x, y)
    r2    = r_value ** 2
    y_hat = intercept + slope * x
    se_res = np.sqrt(np.sum((y - y_hat) ** 2) / (n - 2))
    t_crit = stats.t.ppf(0.975, df=n - 2)
    ci_low, ci_high = slope - t_crit * se_slope, slope + t_crit * se_slope
    rho, p_spearman = stats.spearmanr(x, y)

    sep = '─' * 60
    print(sep)
    print('ANÁLISIS UNIVARIANTE:  RPD ~ B̃_c')
    print(sep)
    print(f'  N (filas válidas): {n}')
    print()
    print('  Datos:')
    print(f'  {"Task":<8} {"Cls":<8} {"B̃_c":>7} {"RPD":>7}')
    for v in valid:
        print(f'    {v["task"]:<8} {v["cls_id"]:<8} {v["bs"]:>+7.3f} {v["rpd"]:>+7.3f}')
    print()
    print('  Regresión OLS  (X = B̃_c,  Y = RPD)')
    print(f'    Intercepto (β0): {intercept:+.4f}')
    print(f'    Pendiente  (β1): {slope:+.4f}  '
          f'(IC 95%: [{ci_low:+.4f}, {ci_high:+.4f}])')
    print(f'    SE residual    : {se_res:.4f}')
    print(f'    R²             : {r2:.4f}')
    print(f'    p-valor (β1=0) : {p_value:.4e}')
    print()
    print('  Correlación de Spearman')
    print(f'    ρ              : {rho:+.4f}')
    print(f'    p-valor        : {p_spearman:.4e}')
    print(sep)
    print()


# ── Análisis multivariante RPD ~ Δn + d + B̃_c ────────────────────────────────

def multivariate_analysis(rows_mean: list, centroid_dists: dict,
                           morph_sep: dict) -> None:
    """
    Regresión OLS múltiple: RPD = β0 + β1·Δn + β2·d + β3·B̃_c + ε
    Se ajustan también modelos parciales para valorar contribución individual.
    Se reporta VIF para detectar multicolinealidad.
    """
    valid = []
    for r in rows_mean:
        dist = centroid_dists.get((r["task"], r["cls_id"]))
        bs   = morph_sep.get((r["task"], r["cls_id"]))
        if dist is None or bs is None or np.isnan(r["rpd"]):
            continue
        valid.append({
            "rpd": r["rpd"], "dn": r["delta_n"],
            "dist": dist, "bs": bs,
            "task": r["task"], "cls_id": r["cls_id"],
        })

    if len(valid) < 5:
        print("[WARN] Insuficientes datos para análisis multivariante", file=sys.stderr)
        return

    from scipy.linalg import lstsq as sp_lstsq

    rpd = np.array([v["rpd"]  for v in valid])
    dn  = np.array([v["dn"]   for v in valid])
    d   = np.array([v["dist"] for v in valid])
    bs  = np.array([v["bs"]   for v in valid])
    n   = len(valid)

    def ols(X, y):
        X1 = np.column_stack([np.ones(len(y)), X])
        k  = X1.shape[1] - 1
        b, _, _, _ = sp_lstsq(X1, y)
        y_hat  = X1 @ b
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2     = 1 - ss_res / ss_tot
        r2_adj = 1 - (1 - r2) * (n - 1) / (n - k - 1)
        ms_res = ss_res / (n - k - 1) if n - k - 1 > 0 else np.nan
        f_stat = ((ss_tot - ss_res) / k) / ms_res if ms_res else np.nan
        p_f    = 1 - stats.f.cdf(f_stat, k, n - k - 1) if not np.isnan(f_stat) else np.nan
        cov_b  = ms_res * np.linalg.pinv(X1.T @ X1) if ms_res else np.full((k+1, k+1), np.nan)
        se_b   = np.sqrt(np.diag(cov_b))
        t_b    = b / se_b
        p_b    = 2 * (1 - stats.t.cdf(np.abs(t_b), df=n - k - 1))
        return b, r2, r2_adj, f_stat, p_f, se_b, t_b, p_b

    def vif(X):
        vifs = []
        for i in range(X.shape[1]):
            _, r2_i, *_ = ols(np.delete(X, i, axis=1), X[:, i])
            vifs.append(1 / (1 - r2_i) if r2_i < 1 else np.inf)
        return vifs

    sep = "═" * 60
    print(sep)
    print("ANÁLISIS MULTIVARIANTE:  RPD ~ Δn + d + B̃_c")
    print(sep)
    print(f"  N = {n}")

    MODELS = [
        ("Δn + d + B̃_c", np.column_stack([dn, d, bs]), ["Δn", "d", "B̃_c"]),
        ("Δn + B̃_c",     np.column_stack([dn, bs]),    ["Δn", "B̃_c"]),
        ("d  + B̃_c",     np.column_stack([d,  bs]),    ["d",  "B̃_c"]),
        ("Δn + d",        np.column_stack([dn, d]),     ["Δn", "d"]),
    ]

    k_models   = len(MODELS)                # modelos comparados
    alpha_bonf = 0.05 / k_models            # α corregido = 0.0125
    print(f"\n  Corrección Bonferroni: k={k_models} modelos → α_corr={alpha_bonf:.4f}")

    for label, X, names in MODELS:
        b, r2, r2_adj, f_stat, p_f, se_b, t_b, p_b = ols(X, rpd)
        sig_f = "**" if p_f < alpha_bonf else ("*" if p_f < 0.05 else "")
        print(f"\n  ── Modelo: RPD ~ {label} ──")
        print(f"    R²={r2:.3f}   R²_adj={r2_adj:.3f}   "
              f"F={f_stat:.2f}   p(F)={p_f:.4e}{sig_f}")
        for nm, bi, si, ti, pi in zip(["Intercept"] + names, b, se_b, t_b, p_b):
            sig = "*" if pi < 0.05 else (" †" if pi < 0.10 else "")
            print(f"    {nm:<12}  β={bi:+.4f}  SE={si:.4f}  "
                  f"t={ti:+.3f}  p={pi:.4e}{sig}")
        if X.shape[1] == 3:
            vifs = vif(X)
            print("    VIF: " + "  ".join(f"{nm}={v:.2f}" for nm, v in zip(names, vifs)))

    print(f"\n  (* p<0.05  ** p<α_Bonf={alpha_bonf:.4f}  † p<0.10)")
    print(sep)

    # ── Correlaciones entre predictores (Pearson vs Spearman) ────────────────
    predictors = [("Δn", dn), ("d", d), ("B̃_c", bs)]
    print("\n  Correlaciones entre predictores (linealidad vs. monotonicidad)")
    print(f"  {'Par':<16}  {'Pearson r':>10}  {'Spearman ρ':>11}  {'p(Spearman)':>12}")
    print("  " + "─" * 56)
    for i in range(len(predictors)):
        for j in range(i + 1, len(predictors)):
            na, xa = predictors[i]
            nb, xb = predictors[j]
            pr = np.corrcoef(xa, xb)[0, 1]
            rho_s, p_s = stats.spearmanr(xa, xb)
            par = f"{na} ~ {nb}"
            sig = "*" if p_s < 0.05 else ""
            print(f"  {par:<16}  {pr:>+10.3f}  {rho_s:>+11.3f}  {p_s:>12.4e} {sig}")
    print("  " + "─" * 56)
    print("  (Si |Pearson r| ≈ |Spearman ρ|, la relación es aproximadamente lineal.)")
    print(sep)
    print()



def generate_stats_latex(rows_mean: list, centroid_dists: dict,
                         morph_sep: dict) -> str:
    """
    Tabla LaTeX unificada con modelos univariantes, multivariante y colinealidad,
    siguiendo el formato del paper.
    """
    valid = []
    for r in rows_mean:
        dist = centroid_dists.get((r["task"], r["cls_id"]))
        bs   = morph_sep.get((r["task"], r["cls_id"]))
        dp   = PREVALENCE_SHIFT.get((r["task"], r["cls_id"]))
        if dist is None or bs is None or np.isnan(r["rpd"]):
            continue
        valid.append({"rpd": r["rpd"], "dn": r["delta_n"],
                      "dist": dist, "bs": bs,
                      "dp": dp if dp is not None else float("nan")})

    if len(valid) < 5:
        return "% Insuficientes datos para análisis estadístico\n"

    from scipy.linalg import lstsq as sp_lstsq

    rpd = np.array([v["rpd"]  for v in valid])
    dn  = np.array([v["dn"]   for v in valid])
    d   = np.array([v["dist"] for v in valid])
    bs  = np.array([v["bs"]   for v in valid])
    dp  = np.array([v["dp"]   for v in valid])
    n   = len(valid)

    def ols(X, y):
        X1 = np.column_stack([np.ones(len(y)), X])
        k  = X1.shape[1] - 1
        b, _, _, _ = sp_lstsq(X1, y)
        y_hat  = X1 @ b
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2     = 1 - ss_res / ss_tot
        r2_adj = 1 - (1 - r2) * (n - 1) / (n - k - 1)
        ms_res = ss_res / (n - k - 1) if n - k - 1 > 0 else np.nan
        f_stat = ((ss_tot - ss_res) / k) / ms_res if ms_res else np.nan
        p_f    = 1 - stats.f.cdf(f_stat, k, n - k - 1) if not np.isnan(f_stat) else np.nan
        cov_b  = ms_res * np.linalg.pinv(X1.T @ X1) if ms_res else np.full((k+1, k+1), np.nan)
        se_b   = np.sqrt(np.diag(cov_b))
        t_b    = b / se_b
        p_b    = 2 * (1 - stats.t.cdf(np.abs(t_b), df=n - k - 1))
        return b, r2, r2_adj, f_stat, p_f, se_b, t_b, p_b

    def vif_pair(x1, x2):
        _, r2_i, *_ = ols(x2.reshape(-1, 1), x1)
        return 1 / (1 - r2_i) if r2_i < 1 else float("inf")

    def pf(p):
        if p < 0.001: return r"$<$0.001"
        return f"{p:.3f}"

    def bh_reject(pvals, alpha=0.05):
        """Benjamini-Hochberg procedure. Returns boolean array (True = rejected)."""
        pvals = np.array(pvals, dtype=float)
        m = len(pvals)
        idx = np.argsort(pvals)
        thresholds = (np.arange(1, m + 1) / m) * alpha
        rejected = pvals[idx] <= thresholds
        result = np.zeros(m, dtype=bool)
        if rejected.any():
            last = np.where(rejected)[0][-1]
            result[idx[:last + 1]] = True
        return result

    def sig_bh(rejected, p):
        if rejected: return r"$^{*}$"
        if p < 0.10: return r"$^{\dagger}$"
        return ""

    def pearson(a, b):
        return float(np.corrcoef(a, b)[0, 1])

    # ── Univariantes ──────────────────────────────────────────────────────────
    uni_predictors = [
        (r"$\Delta n$",       dn),
        (r"$d_c$",            d),
        (r"$\tilde{B}_c$",   bs),
        (r"$\Delta p$",       dp),
    ]
    # Compute all univariate stats first (need all p-values for BH)
    uni_stats = []
    for label, x in uni_predictors:
        b, r2, *_, se_b, t_b, p_b = ols(x.reshape(-1, 1), rpd)
        r_val = pearson(x, rpd)
        rho_u, p_rho_u = stats.spearmanr(x, rpd)
        uni_stats.append((label, r_val, rho_u, p_rho_u, r2, b[1], p_b[1]))

    # BH over all univariate p-values: Spearman (m=4) + OLS slope (m=4)
    uni_p_rho = [s[3] for s in uni_stats]
    uni_p_ols = [s[6] for s in uni_stats]
    uni_bh_rho = bh_reject(uni_p_rho)
    uni_bh_ols = bh_reject(uni_p_ols)

    uni_rows = []
    for i, (label, r_val, rho_u, p_rho_u, r2, slope, p_ols) in enumerate(uni_stats):
        uni_rows.append(
            f"    {label} & ${r_val:+.3f}$ & ${rho_u:+.3f}$ & "
            f"{pf(p_rho_u)}{sig_bh(uni_bh_rho[i], p_rho_u)} & "
            f"${r2:.3f}$ & ${slope:+.3f}$ & "
            f"{pf(p_ols)}{sig_bh(uni_bh_ols[i], p_ols)} \\\\"
        )

    # ── Modelo completo (para caption: VIF y p de B̃_c) ──────────────────────
    def vif3(X):
        """VIF de cada columna en un diseño de 3 predictores."""
        vifs = []
        for i in range(X.shape[1]):
            _, r2_i, *_ = ols(np.delete(X, i, axis=1), X[:, i])
            vifs.append(1 / (1 - r2_i) if r2_i < 1 else float("inf"))
        return vifs

    X_full = np.column_stack([dn, d, bs])
    b_full, r2_full, _, _, _, _, _, p_full = ols(X_full, rpd)
    vifs_full = vif3(X_full)          # [VIF_dn, VIF_d, VIF_bs]
    p_bs_full  = p_full[3]            # p-value of B̃_c in full model
    vif_bs_full = vifs_full[2]        # VIF of B̃_c in full model

    # Collinearity stats needed for caption (computed below, use placeholders here)
    # Will be filled after collin_stats is built

    # ── Multivariante ─────────────────────────────────────────────────────────
    b_mv, r2_mv, r2_adj_mv, f_mv, pf_mv, se_mv, t_mv, p_mv = ols(
        np.column_stack([dn, d]), rpd)

    # BH over multivariate p-values: F-test + 2 coefficients
    mv_pvals = [pf_mv, p_mv[1], p_mv[2]]
    mv_bh = bh_reject(mv_pvals)

    mv_fit_row = (
        r"    RPD $\sim \Delta n + d_c$ & \multicolumn{2}{c}{---} & "
        f"${r2_mv:.3f}$ & ${r2_adj_mv:.3f}$ & ${f_mv:.2f}$ & {pf(pf_mv)}{sig_bh(mv_bh[0], pf_mv)} \\\\"
    )
    # coeficientes (sin intercept)
    coef_rows = []
    for j, (nm, bi, si, ti, pi) in enumerate(zip([r"$\Delta n$", r"$d_c$"],
                                                   b_mv[1:], se_mv[1:], t_mv[1:], p_mv[1:])):
        coef_rows.append(
            f"    {nm} & ${bi:+.4f}$ & ${si:.4f}$ & ${ti:+.3f}$ & {pf(pi)}{sig_bh(mv_bh[1 + j], pi)} & \\\\"
        )

    # ── Colinealidad ──────────────────────────────────────────────────────────
    collin_pairs = [
        (r"$\Delta n \sim d_c$",           dn, d),
        (r"$\Delta n \sim \tilde{B}_c$",   dn, bs),
        (r"$d_c \sim \tilde{B}_c$",        d,  bs),
    ]
    collin_stats = []
    for label, xa, xb in collin_pairs:
        pr     = pearson(xa, xb)
        rho_s, p_s = stats.spearmanr(xa, xb)
        v      = vif_pair(xa, xb)
        collin_stats.append((label, pr, rho_s, p_s, v))

    collin_bh = bh_reject([s[3] for s in collin_stats])
    collin_rows = []
    for i, (label, pr, rho_s, p_s, v) in enumerate(collin_stats):
        collin_rows.append(
            f"    {label} & ${pr:+.3f}$ & ${rho_s:+.3f}$ & {pf(p_s)}{sig_bh(collin_bh[i], p_s)} & ${v:.2f}$ & \\\\"
        )

    # ── Ensamblado final ──────────────────────────────────────────────────────
    lines_out = [
        r"\begin{table}[h!]",
        r"\centering",
        (r"\caption{Statistical models predicting relative performance degradation (RPD) "
         r"from domain shift factors. The most parsimonious multivariate model includes only "
         r"$\Delta n$ and $d_c$ ($n=" + str(n) + r"$ molecular classes). "
         r"$\tilde{B}_c$ is excluded because it contributes no independent predictive power "
         r"once $\Delta n$ and $d_c$ are included "
         r"($\beta\approx" + f"{b_full[3]:+.3f}" + r"$, $p=" + f"{p_bs_full:.3f}" + r"$, "
         r"$\Delta R^2<0.001$), and shows moderate collinearity with $\Delta n$ "
         r"(Spearman $\rho=" + f"{collin_stats[1][2]:+.3f}" + r"$, $p=" + f"{collin_stats[1][3]:.3f}" + r"$) "
         r"and $d_c$ ($\rho=" + f"{collin_stats[2][2]:+.3f}" + r"$, $p=" + f"{collin_stats[2][3]:.3f}" + r"$), "
         r"with VIF$=" + f"{vif_bs_full:.2f}" + r"$ in the joint model. "
         r"Multiple-testing correction applied via Benjamini--Hochberg (BH) procedure "
         r"($\alpha=0.05$) separately within each section (univariate Spearman tests, "
         r"univariate OLS slope tests, multivariate model, collinearity tests). "
         r"$^{*}$BH-significant ($q<0.05$), $^{\dagger}p<0.10$.}"),
        r"\label{tab:regression-models}",
        r"\small",
        r"\begin{tabular}{lccccccl}",
        r"\toprule\toprule",
        r"\multicolumn{8}{l}{\textit{Univariate Models}} \\",
        r"\midrule",
        r"\textbf{Model} & \textbf{$r$} & \textbf{$\rho$} & \textbf{$p(\rho)$} & \textbf{$R^2$} & \textbf{$\beta$} & \textbf{$p$-value} \\",
        r"\midrule",
    ] + uni_rows + [
        r"\midrule\midrule",
        r"\multicolumn{8}{l}{\textit{Multivariate Model}} \\",
        r"\midrule",
        r"\textbf{Model} & \multicolumn{2}{c}{} & \textbf{$R^2$} & \textbf{Adj.\ $R^2$} & \textbf{$F$-stat} & \textbf{$p$-value} \\",
        r"\midrule",
        mv_fit_row,
        r"\midrule",
        r" & \textbf{$\beta$} & \textbf{Std Error} & \textbf{$t$-value} & \textbf{$p$-value} & \multicolumn{2}{l}{} \\",
        r"\midrule",
    ] + coef_rows + [
        r"\midrule",
        r"\multicolumn{8}{l}{\textit{Predictor Collinearity}} \\",
        r"\midrule",
        r"\textbf{Pair} & \textbf{Pearson $r$} & \textbf{Spearman $\rho$} & \textbf{$p(\rho)$} & \textbf{VIF} & \multicolumn{2}{l}{} \\",
        r"\midrule",
    ] + collin_rows + [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines_out)



# ── Análisis confusiones PAM50 vs. disimilitud biológica ─────────────────────

PAM50_CLS = {0: 'Basal', 1: 'HER2-enr', 2: 'LumA', 3: 'LumB', 4: 'Normal'}
PAM50_LABELS = [PAM50_CLS[i] for i in range(5)]
BIO_RENAME = {
    'BASAL': 'Basal', 'HER2-ENRICHED': 'HER2-enr',
    'LUMINAL-A': 'LumA', 'LUMINAL-B': 'LumB', 'NORMAL-LIKE': 'Normal',
}
BIO_MATRIX_CSV = (
    PATCH_INTER_DIR.parent
    / 'biological_comparison_BASAL_HER2-ENRICHED_LUMINAL-A_LUMINAL-B_NORMAL-LIKE_matriz.csv'
)


def _parse_pam50_cms(results_dir: Path) -> dict:
    """Parsea matrices de confusión (absolutas) de pam50_results.txt.
    Devuelve {(mil, norm, dataset): ndarray 5×5}."""
    import csv as _csv
    filepath = results_dir / 'pam50_results.txt'
    if not filepath.exists():
        return {}

    lines = filepath.read_text(encoding='utf-8').splitlines()
    cur_mil = cur_norm = cur_ds = None
    in_cm = False
    cm_rows = []
    cms = {}

    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r'MIL:\s+(\S+)\s+\|\s+Norm:\s+(\S+)', line)
        if m:
            cur_mil, cur_norm = m.group(1), m.group(2)
            cur_ds = None; in_cm = False; i += 1; continue
        m = re.match(r'Dataset:\s+(\S+)', line)
        if m:
            cur_ds = m.group(1).lower(); in_cm = False; i += 1; continue
        if 'Confusion Matrix (filas=real' in line:
            in_cm = True; cm_rows = []; i += 2; continue  # salta cabecera cols
        if in_cm:
            m2 = re.match(r'\d+\(\d+\)((?:\s+\d+)+)', line)
            if m2:
                cm_rows.append(list(map(int, m2.group(1).split())))
                if len(cm_rows) == 5:
                    cms[(cur_mil, cur_norm, cur_ds)] = np.array(cm_rows)
                    in_cm = False
            else:
                in_cm = False
        i += 1
    return cms


def analyse_confusions_vs_biology(results_dir: Path) -> None:
    cms = _parse_pam50_cms(results_dir)
    if not cms:
        print('[WARN] No se encontraron matrices de confusión PAM50.', file=sys.stderr)
        return

    # Suma de los 3 modelos Optuna / none / cptac
    total_cm = np.zeros((5, 5), dtype=int)
    for mil in MIL_ORDER:
        key = (mil, 'none', 'cptac')
        if key in cms:
            total_cm += cms[key]
        else:
            print(f'[WARN] CM no encontrada: {key}', file=sys.stderr)

    labels = PAM50_LABELS
    df_total = _cm_df(total_cm, labels)

    # Off-diagonal: errores absolutos y % por clase real
    err_cm = total_cm.copy()
    np.fill_diagonal(err_cm, 0)
    df_err = _cm_df(err_cm, labels)

    row_totals = total_cm.sum(axis=1)
    df_err_pct = _cm_df(
        (err_cm / row_totals[:, None] * 100).round(1), labels)

    sep = '─' * 60
    print(sep)
    print('CONFUSIONES PAM50  —  3 modelos Optuna × CPTAC × none')
    print(sep)
    print(f'  Total slides evaluados: {total_cm.sum()} (3 × 387)')
    print()
    print('  Predicciones absolutas (filas=real, columnas=predicho):')
    print(df_total.to_string(col_space=10))
    print()
    print('  Errores absolutos (off-diagonal):')
    print(df_err.to_string(col_space=10))
    print(f'  Total errores: {err_cm.sum()}')
    print()
    print('  Errores (% sobre total real por clase):')
    print(df_err_pct.to_string(col_space=10))
    print()

    # Matriz biológica
    if not BIO_MATRIX_CSV.exists():
        print(f'[WARN] Matriz biológica no encontrada: {BIO_MATRIX_CSV}', file=sys.stderr)
        print(sep)
        return

    import csv as _csv
    bio_raw = {}
    with open(BIO_MATRIX_CSV) as f:
        reader = _csv.DictReader(f)
        col_keys = [BIO_RENAME.get(c, c) for c in reader.fieldnames[1:]]
        for row in reader:
            row_key = BIO_RENAME.get(row[reader.fieldnames[0]], row[reader.fieldnames[0]])
            bio_raw[row_key] = [float(row[c]) for c in reader.fieldnames[1:]]

    bio_arr = np.array([[bio_raw[r][j] for j in range(5)] for r in labels])
    df_bio = _cm_df(bio_arr, labels)

    print('  Matriz biológica (Σ|r| comparaciones sig., filas=TCGA, cols=CPTAC):')
    print(df_bio.round(3).to_string(col_space=10))
    print()

    # Correlación Spearman off-diagonal: disimilitud bio vs. tasa de confusión
    mask = ~np.eye(5, dtype=bool)
    x_bio = bio_arr[mask]
    x_err = (err_cm / row_totals[:, None])[mask]
    rho, p_sp = stats.spearmanr(x_bio, x_err)
    print('  Correlación Spearman off-diagonal (disim. bio. ~ tasa confusión):')
    print(f'    ρ = {rho:+.3f}   p = {p_sp:.4e}')
    if p_sp < 0.05:
        direction = 'mayor disimilitud biológica → más confusión' if rho > 0 else \
                    'mayor disimilitud biológica → menos confusión'
        print(f'    Significativa: {direction}')
    else:
        print('    No significativa: los errores no siguen la disimilitud biológica.')
    print(sep)
    print()


def _cm_df(arr, labels):
    import pandas as pd
    return pd.DataFrame(arr, index=labels, columns=labels)



# ── Confusion matrices for CLAM-MB CPTAC ───────────────────────────────────────────

def _parse_binary_cms(results_dir: Path, task: str) -> dict:
    """Parse confusion matrices from binary task results files.
    Returns {(mil, norm, dataset): ndarray NxN}."""
    filepath = results_dir / f'{task}_results.txt'
    if not filepath.exists():
        return {}

    lines = filepath.read_text(encoding='utf-8').splitlines()
    cur_mil = cur_norm = cur_ds = None
    in_cm = False
    cm_rows = []
    cms = {}
    n_classes = 2

    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r'MIL:\s+(\S+)\s+\|\s+Norm:\s+(\S+)', line)
        if m:
            cur_mil, cur_norm = m.group(1), m.group(2)
            cur_ds = None; in_cm = False; i += 1; continue
        m = re.match(r'Dataset:\s+(\S+)', line)
        if m:
            cur_ds = m.group(1).lower(); in_cm = False; i += 1; continue
        if 'Confusion Matrix (filas=real' in line:
            in_cm = True; cm_rows = []; i += 2; continue  # skip header
        if in_cm:
            m2 = re.match(r'\d+\(\d+\)((?:\s+\d+)+)', line)
            if m2:
                cm_rows.append(list(map(int, m2.group(1).split())))
                if len(cm_rows) == n_classes:
                    cms[(cur_mil, cur_norm, cur_ds)] = np.array(cm_rows)
                    in_cm = False
            else:
                in_cm = False
        i += 1
    return cms


def plot_confusion_matrices_clam(results_dir: Path) -> None:
    """Plot CLAM-MB CPTAC confusion matrices for all tasks as heatmaps."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    figures_dir = results_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)

    TASK_LABELS = {
        'pam50': ['Basal', 'Her2-e', 'LumA', 'LumB', 'Normal'],
        'er':    ['neg', 'pos'],
        'pr':    ['neg', 'pos'],
        'erbb2': ['neg', 'pos'],
    }

    for task in TASK_ORDER:
        labels = TASK_LABELS[task]
        n_cls  = len(labels)

        if task == 'pam50':
            cms = _parse_pam50_cms(results_dir)
        else:
            cms = _parse_binary_cms(results_dir, task)

        key = ('clam_mil_mb', 'none', 'cptac')
        if key not in cms:
            print(f'[WARN] CM not found for {task} clam_mil_mb none cptac', file=sys.stderr)
            continue

        cm = cms[key].astype(float)
        row_totals = cm.sum(axis=1, keepdims=True)
        cm_norm = np.where(row_totals > 0, cm / row_totals, 0.0)

        fig, axes = plt.subplots(1, 2, figsize=(max(8, n_cls * 2), max(4, n_cls * 1.5)))
        fig.suptitle(f'CLAM-MB  —  {task.upper()}  —  CPTAC (none)',
                     fontsize=12, fontweight='bold')

        # Absolute counts
        sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues',
                    xticklabels=labels, yticklabels=labels,
                    ax=axes[0], cbar=True, linewidths=0.5)
        axes[0].set_title('Absolute counts')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('True')

        # Recall-normalised
        annot_norm = np.array([[f'{v:.2f}' for v in row] for row in cm_norm])
        sns.heatmap(cm_norm, annot=annot_norm, fmt='', cmap='Blues',
                    xticklabels=labels, yticklabels=labels,
                    vmin=0, vmax=1,
                    ax=axes[1], cbar=True, linewidths=0.5)
        axes[1].set_title('Recall-normalised')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('True')

        plt.tight_layout()
        for ext in ('pdf', 'png'):
            out = figures_dir / f'cm_clam_{task}_cptac.{ext}'
            fig.savefig(out, bbox_inches='tight', facecolor='white')
            print(f'  Guardado: {out}', file=sys.stderr)
        plt.close(fig)


# ── Biological B_{{c,d}} matrix heatmaps ─────────────────────────────────────────

def plot_biological_matrix(results_dir: Path) -> None:
    """Plot full B_{{c,d}} biological comparison matrices as heatmaps."""
    import csv as _csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    figures_dir = results_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)

    DISPLAY_LABELS = {
        'PAM50': ['Basal', 'Her2-e', 'LumA', 'LumB', 'Normal'],
        'ER':    ['neg', 'pos'],
        'PR':    ['neg', 'pos'],
        'ERBB2': ['neg', 'pos'],
    }

    for task_upper, info in MORPH_SEP_MATRICES.items():
        p = BIO_DIR / info['csv']
        if not p.exists():
            print(f'[WARN] No encontrado: {p}', file=sys.stderr)
            continue

        with open(p) as f:
            rows_csv = list(_csv.DictReader(f))

        raw_labels = info['labels']
        n = len(raw_labels)
        mat = np.array(
            [[float(rows_csv[i][col]) for col in raw_labels] for i in range(n)]
        )

        display_labels = DISPLAY_LABELS[task_upper]
        vmax = np.abs(mat).max()
        vmax = vmax if vmax > 0 else 1.0

        fig, ax = plt.subplots(figsize=(max(5, n * 1.4), max(4, n * 1.2)))
        annot = np.array([[f'{v:.2f}' for v in row] for row in mat])
        sns.heatmap(mat, annot=annot, fmt='', cmap='RdBu_r',
                    vmin=-vmax, vmax=vmax,
                    xticklabels=display_labels, yticklabels=display_labels,
                    ax=ax, cbar=True, linewidths=0.5)
        ax.set_title(
            f'Biological matrix B_{{c,d}} — {task_upper}\n'
            f'Rows=TCGA classes, Cols=CPTAC classes | Values=\u03a3|effect_r| sig.',
            fontsize=10)
        ax.set_xlabel('CPTAC class')
        ax.set_ylabel('TCGA class')
        plt.tight_layout()

        task_lower = task_upper.lower()
        for ext in ('pdf', 'png'):
            out = figures_dir / f'bio_matrix_{task_lower}.{ext}'
            fig.savefig(out, bbox_inches='tight', facecolor='white')
            print(f'  Guardado: {out}', file=sys.stderr)
        plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    _arg = sys.argv[1] if len(sys.argv) > 1 else None
    results_dir = Path(_arg) if (_arg and Path(_arg).is_dir()) else RESULTS_DIR

    rows = compute_rows(results_dir)
    if not rows:
        print('[ERROR] No se extrajeron datos. Revisa los ficheros de resultados.',
              file=sys.stderr)
        sys.exit(1)

    # Agregar por (task, cls_id): media de los tres modelos (CLAM-MB, DSMIL, TransMIL)
    from itertools import groupby
    key_fn = lambda r: (r['task'], r['cls_id'], r['cls_name'])
    rows_mean = []
    for (task, cls_id, cls_name), group in groupby(
            sorted(rows, key=key_fn), key=key_fn):
        group = list(group)
        valid_rpd = [r['rpd'] for r in group if not np.isnan(r['rpd'])]
        rows_mean.append({
            'task':           task,
            'mil':            'mean',
            'cls_id':         cls_id,
            'cls_name':       cls_name,
            'delta_n':        np.mean([r['delta_n'] for r in group]),
            'rpd':            np.mean(valid_rpd) if valid_rpd else float('nan'),
            'none_tcga_mean': np.mean([r['none_tcga']  for r in group]),
            'none_cptac_mean':np.mean([r['none_cptac'] for r in group]),
        })

    centroid_dists = load_centroid_distances(top_k=8)

    univariate_analysis(rows_mean)
    univariate_prevalence(rows_mean)
    univariate_centroid(rows_mean, centroid_dists)
    morph_sep_early = load_morphological_separability()
    univariate_d_vs_morph(rows_mean, centroid_dists, morph_sep_early)
    univariate_rpd_vs_morph(rows_mean, morph_sep_early)
    multivariate_analysis(rows_mean, centroid_dists, morph_sep_early)
    analyse_confusions_vs_biology(results_dir)


def plot_morphology_vs_confusion(results_dir: Path, show_diag: bool = True) -> None:
    """
    Per tarea: figura con 4 paneles (CM clam | CM dsmil | CM transmil | Bio matrix).
    Plus scatter global B_c,d vs confusion rate. Tonos rosados.
    show_diag: si True muestra diagonal con color; si False la deja en blanco.
    """
    import csv as _csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import seaborn as sns
    import pandas as _pd
    from sklearn.metrics import confusion_matrix as _cm_fn

    figures_dir = results_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)

    PINK     = mcolors.LinearSegmentedColormap.from_list(
        'pink_mono', ['#FFFFFF', '#AD1457'], N=256)
    PINK_INV = mcolors.LinearSegmentedColormap.from_list(
        'pink_inv', ['#AD1457', '#FFFFFF'], N=256)

    def _row_heatmap(ax, mat, cmap, fmt, xlabel, ylabel, title, row_labels, col_labels, global_max=None, row_maxes=None):
        """Heatmap con normalización por filas: vmin=0, vmax=max de la fila (o global_max/row_maxes si se pasan)."""
        n = mat.shape[0]
        is_inv = (cmap is PINK_INV)
        rgba = np.ones((n, n, 4))  # blanco por defecto
        for ri in range(n):
            row_max = (row_maxes[ri] if row_maxes is not None else (global_max if global_max is not None else max(mat[ri, ci] for ci in range(n))))
            rrange = row_max if row_max > 0 else 1.0
            for ci in range(n):
                if not show_diag and ri == ci:
                    continue
                norm_val = max(0.0, min(1.0, mat[ri, ci] / rrange))
                rgba[ri, ci] = cmap(norm_val)
        ax.imshow(rgba, aspect='auto', interpolation='nearest',
                  extent=[-0.5, n - 0.5, n - 0.5, -0.5])
        for k in range(n + 1):
            ax.axhline(k - 0.5, color='#E8A0B4', lw=0.5)
            ax.axvline(k - 0.5, color='#E8A0B4', lw=0.5)
        for ri in range(n):
            row_max = (row_maxes[ri] if row_maxes is not None else (global_max if global_max is not None else max(mat[ri, ci] for ci in range(n))))
            rrange = row_max if row_max > 0 else 1.0
            for ci in range(n):
                if not show_diag and ri == ci:
                    continue
                norm_val = max(0.0, min(1.0, mat[ri, ci] / rrange))
                dark_bg = (norm_val < 0.5) if is_inv else (norm_val > 0.5)
                ax.text(ci, ri, format(mat[ri, ci], fmt),
                        ha='center', va='center',
                        color='white' if dark_bg else '#2D0A1A', fontsize=9)
        ax.set_xticks(range(n))
        ax.set_xticklabels(col_labels, fontsize=8)
        ax.set_yticks(range(n))
        ax.set_yticklabels(row_labels, fontsize=8)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=10, color='#6D1B3A')

    DISPLAY_LABELS = {
        'PAM50': ['Basal', 'Her2-e', 'LumA', 'LumB', 'Normal'],
        'ER':    ['neg', 'pos'],
        'PR':    ['neg', 'pos'],
        'ERBB2': ['neg', 'pos'],
    }
    MIL_TITLES = {
        'clam_mil_mb': 'CLAM-MB',
        'dsmil':       'DSMIL',
        'transmil':    'TransMIL',
    }
    TASK_N_CLS = {'er': 2, 'erbb2': 2, 'pr': 2, 'pam50': 5}
    EXP_BASE   = Path('/shared/home/jorgarcia/PathBench-MIL/experiments')

    def _load_cm_counts(task, mil, dataset='cptac'):
        if task == 'pam50':
            cms = _parse_pam50_cms(results_dir)
        else:
            cms = _parse_binary_cms(results_dir, task)
        cm = cms.get((mil, 'none', dataset))
        if cm is None:
            return None
        return cm.astype(float)

    def _load_bio(task_upper):
        info = MORPH_SEP_MATRICES.get(task_upper)
        if not info:
            return None
        p = BIO_DIR / info['csv']
        if not p.exists():
            return None
        with open(p) as f:
            rows_csv = list(_csv.DictReader(f))
        raw = info['labels']
        return np.array([[float(rows_csv[i][c]) for c in raw]
                         for i in range(len(raw))])

    x_sc, y_sc, c_sc, t_sc = [], [], [], []
    # Máximo global entre todas las matrices biológicas
    _global_bio_max = max(
        (m.max() for t in TASK_ORDER
         for m in [_load_bio(t.upper())] if m is not None),
        default=1.0)
    TCOL = {'ER': '#F06292', 'ERBB2': '#9b59b6',
            'PR': '#AD1457', 'PAM50': '#FADADD'}

    for task in TASK_ORDER:
        task_up = task.upper()
        disp    = DISPLAY_LABELS[task_up]
        bio_mat = _load_bio(task_up)
        n_cls   = len(disp)

        def _off_diag(m):
            off = m[~np.eye(m.shape[0], dtype=bool)]
            return float(off.min()), float(off.max()) if len(off) > 0 else (0.0, 1.0)

        panels = []  # (title, mat, vmin, vmax, fmt, cbar_lbl, cmap)
        for mil in MIL_ORDER:
            cm = _load_cm_counts(task, mil)
            if cm is not None:
                _cm_vmin, _cm_vmax = _off_diag(cm)
                panels.append((MIL_TITLES[mil], cm,
                                _cm_vmin, max(_cm_vmax, 1), '.0f', 'Count', PINK))
        if bio_mat is not None:
            _bio_vmin, _bio_vmax_off = _off_diag(bio_mat)
            _bio_vmax = max(_bio_vmax_off, 0.1)
            panels.append(('Morphological dist.', bio_mat,
                           _bio_vmin, _bio_vmax, '.2f', 'B_cd', PINK_INV))
            for mil in MIL_ORDER:
                cm = _load_cm_counts(task, mil)
                if cm is None:
                    continue
                cm_pct = cm / cm.sum(axis=1, keepdims=True).clip(1) * 100
                for i in range(bio_mat.shape[0]):
                    for j in range(bio_mat.shape[0]):
                        if i == j:
                            continue
                        x_sc.append(bio_mat[i, j])
                        y_sc.append(cm_pct[i, j])
                        c_sc.append(TCOL.get(task_up, '#F06292'))
                        t_sc.append(task_up)

        if not panels:
            continue

        n_panels = len(panels)
        fig_w    = max(5 * n_panels, 14)
        fig_h    = max(4, n_cls * 1.3)
        fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, fig_h))
        if n_panels == 1:
            axes = [axes]
        fig.subplots_adjust(wspace=0.45)

        for ax, (title, mat, vmin, vmax, fmt, cbar_lbl, cmap) in zip(axes, panels):
            _gmax = None  # siempre normalización por fila
            _ylabel = 'True / TCGA class' if cmap is PINK_INV else 'True / CPTAC class'
            _row_heatmap(ax, mat, cmap, fmt,
                        xlabel='Predicted / CPTAC class',
                        ylabel=_ylabel,
                        title=title,
                        row_labels=disp, col_labels=disp,
                        global_max=_gmax)

        task_disp = TASK_DISPLAY.get(task_up, task_up)
        fig.suptitle(task_disp, fontsize=13, fontweight='bold', color='#6D1B3A')
        plt.tight_layout(rect=[0, 0, 1, 0.93])

        for ext in ('pdf', 'png'):
            diag_sfx = '_diag' if show_diag else '_nodiag'
            out = figures_dir / f'morph_vs_conf_{task}{diag_sfx}.{ext}'
            fig.savefig(out, bbox_inches='tight', dpi=200, facecolor='white')
            print(f'  Guardado: {out}', file=sys.stderr)
        plt.close(fig)

        # ── Figura combinada: CM global (suma 3 modelos) + bio matrix ────────
        cms_all = [_load_cm_counts(task, mil) for mil in MIL_ORDER]
        cms_all = [c for c in cms_all if c is not None]
        cms_tcga = [_load_cm_counts(task, mil, dataset='tcga') for mil in MIL_ORDER]
        cms_tcga = [c for c in cms_tcga if c is not None]
        if cms_all and bio_mat is not None:
            cm_total = sum(cms_all)
            _cmt_vmin, _cmt_vmax = _off_diag(cm_total)
            combined_panels = []
            if cms_tcga:
                cm_tcga_total = sum(cms_tcga)
                _tcga_vmin, _tcga_vmax = _off_diag(cm_tcga_total)
                combined_panels.append(('Confusion TCGA (sum 3 models)', cm_tcga_total,
                    _tcga_vmin, max(_tcga_vmax, 1), '.0f', 'Count', PINK))
            combined_panels += [
                ('Confusion CPTAC (sum 3 models)', cm_total,
                 _cmt_vmin, max(_cmt_vmax, 1), '.0f', 'Count', PINK),
                ('Morphological dist.', bio_mat,
                 _bio_vmin, _bio_vmax, '.2f', 'B_cd', PINK_INV),
            ]
            n_cpanels = len(combined_panels)
            fig2_w = max(5 * n_cpanels, n_cls * 2.8 * n_cpanels / 2)
            fig2_h = max(4, n_cls * 1.3)
            fig2, axes2 = plt.subplots(1, n_cpanels, figsize=(fig2_w, fig2_h))
            if n_cpanels == 1:
                axes2 = [axes2]
            fig2.subplots_adjust(wspace=0.45)
            for ax2, (title2, mat2, vmin2, vmax2, fmt2, lbl2, cmap2) in zip(axes2, combined_panels):
                _gmax2 = float(mat2.max()) if mat2.max() > 0 else 1.0
                # Para la bio matrix: normalizar cada fila por B̃_c de esa clase
                if cmap2 is PINK_INV and bio_mat is not None:
                    # Bio matrix: normalización por max global entre todas las tareas
                    _gmax2_use, _rmaxes2_use = _global_bio_max, None
                else:
                    # CM: normalización por fila (global_max=None)
                    _gmax2_use, _rmaxes2_use = None, None
                _ylabel2 = 'True / TCGA class' if cmap2 is PINK_INV else 'True / CPTAC class'
                _row_heatmap(ax2, mat2, cmap2, fmt2,
                            xlabel='Predicted / CPTAC class',
                            ylabel=_ylabel2,
                            title=title2,
                            row_labels=disp, col_labels=disp,
                            global_max=_gmax2_use, row_maxes=_rmaxes2_use)
            fig2.suptitle(task_disp, fontsize=13, fontweight='bold', color='#6D1B3A')
            plt.tight_layout(rect=[0, 0, 1, 0.93])
            for ext in ('pdf', 'png'):
                out2 = figures_dir / f'morph_vs_conf_{task}_combined{diag_sfx}.{ext}'
                fig2.savefig(out2, bbox_inches='tight', dpi=200, facecolor='white')
                print(f'  Guardado: {out2}', file=sys.stderr)
            plt.close(fig2)

    # Scatter global
    if x_sc:
        from scipy import stats as _stats
        x_arr = np.array(x_sc)
        y_arr = np.array(y_sc)
        rho, p_rho = _stats.spearmanr(x_arr, y_arr)
        slope, intercept, *_ = _stats.linregress(x_arr, y_arr)

        fig, ax = plt.subplots(figsize=(6, 5))
        for task_up in MORPH_SEP_MATRICES:
            idx = [i for i, t in enumerate(t_sc) if t == task_up]
            if idx:
                ax.scatter([x_sc[i] for i in idx], [y_sc[i] for i in idx],
                           color=TCOL.get(task_up, '#F06292'), s=40,
                           edgecolors='#4A0020', linewidth=0.5,
                           label=TASK_DISPLAY.get(task_up, task_up), zorder=3)
        x_line = np.linspace(x_arr.min(), x_arr.max(), 200)
        ax.plot(x_line, intercept + slope * x_line,
                color='#AD1457', linewidth=1.5, linestyle='--')
        ax.set_xlabel('B_cd (morphological distance)', fontsize=11)
        ax.set_ylabel('Confusion rate (%, per model)', fontsize=11)
        p_str = f'{p_rho:.3f}' if p_rho >= 0.001 else '<0.001'
        sign = '+' if rho >= 0 else ''
        ax.set_title(f'rho={sign}{rho:.3f}, p={p_str}', fontsize=10, color='#6D1B3A')
        ax.legend(title='Task', fontsize=8, loc='upper right')
        ax.spines[['top', 'right']].set_visible(False)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4, color='#E8A0B4')
        ax.set_axisbelow(True)
        plt.tight_layout()
        for ext in ('pdf', 'png'):
            out = figures_dir / f'morph_vs_conf_scatter.{ext}'
            fig.savefig(out, bbox_inches='tight', dpi=200, facecolor='white')
            print(f'  Guardado: {out}', file=sys.stderr)
        plt.close(fig)


#!/usr/bin/env python3
"""
Genera tabla LaTeX de resultados desde los ficheros en results/.

Para tareas binarias (ER, ERBB2, PR): extrae PR-AUC por clase.
Para PAM50: extrae F1 por clase.

Métricas derivadas:
  delta_n = mac_CPTAC  - none_CPTAC      (efecto normalización Macenko en test)
  RPD     = (none_CPTAC - none_TCGA) / (-none_TCGA)   (caída relativa de dominio)

Uso:
  python scripts/generate_results_table.py [ruta_results_dir]
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

# ── Configuración ─────────────────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).resolve().parent.parent / 'results'

TASK_METRIC = {
    'er':    'pr_auc',
    'erbb2': 'pr_auc',
    'pr':    'pr_auc',
    'pam50': 'f1',
}

CLASS_LABELS = {
    'er':    {0: 'neg', 1: 'pos'},
    'erbb2': {0: 'neg', 1: 'pos'},
    'pr':    {0: 'neg', 1: 'pos'},
    'pam50': {0: 'Basal', 1: 'Her2', 2: 'LumA', 3: 'LumB', 4: 'Normal'},
}

MIL_DISPLAY = {
    'clam_mil_mb': 'CLAM-MB',
    'dsmil':       'DSMIL',
    'transmil':    'TransMIL',
}

MIL_ORDER  = ['clam_mil_mb', 'dsmil', 'transmil']
TASK_ORDER   = ['er', 'erbb2', 'pr', 'pam50']
TASK_DISPLAY = {'ER': 'ER', 'ERBB2': 'HER2', 'PR': 'PR', 'PAM50': 'PAM50'}


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_results_file(filepath: Path, task: str) -> dict:
    """
    Parsea un fichero de resultados y devuelve:
      {(mil, norm, dataset, cls_id): metric_value}
    donde dataset es 'cptac' o 'tcga'.
    """
    metric = TASK_METRIC[task]
    data   = {}

    with open(filepath, encoding='utf-8') as fh:
        lines = fh.readlines()

    current_mil     = None
    current_norm    = None
    current_dataset = None
    in_classif_rep  = False

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        # ── Cabecera MIL / Norm ───────────────────────────────────────────
        m = re.match(r'MIL:\s+(\S+)\s+\|\s+Norm:\s+(\S+)', line)
        if m:
            current_mil     = m.group(1)
            current_norm    = m.group(2)
            current_dataset = None
            in_classif_rep  = False
            i += 1
            continue

        # ── Cabecera Dataset ──────────────────────────────────────────────
        m = re.match(r'Dataset:\s+(\S+)\s+\|', line)
        if m:
            current_dataset = m.group(1).lower()
            in_classif_rep  = False
            i += 1
            continue

        # ── Bloque PR-AUC por clase ───────────────────────────────────────
        if metric == 'pr_auc' and line.strip() == 'PR-AUC por clase:':
            i += 1
            while i < len(lines):
                m2 = re.match(r'\s+(\d+)\(\d+\):\s+([\d.]+)', lines[i])
                if m2:
                    cls_id = int(m2.group(1))
                    value  = float(m2.group(2))
                    key    = (current_mil, current_norm, current_dataset, cls_id)
                    data[key] = value
                    i += 1
                else:
                    break
            continue

        # ── F1 desde Classification Report ───────────────────────────────
        if metric == 'f1':
            if 'Classification Report' in line:
                in_classif_rep = True
                i += 1
                continue

            if in_classif_rep:
                # Línea de clase: "        0(0)      0.xxx  0.xxx  0.xxx  nnn"
                m2 = re.match(
                    r'\s+(\d+)\(\d+\)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\d+',
                    line)
                if m2:
                    cls_id = int(m2.group(1))
                    f1     = float(m2.group(4))   # columna f1-score
                    key    = (current_mil, current_norm, current_dataset, cls_id)
                    data[key] = f1
                elif re.match(r'\s*accuracy\b', line):
                    in_classif_rep = False   # fin del bloque
                # líneas en blanco y cabecera (precision/recall) se ignoran

        i += 1

    return data


# ── Cómputo de métricas derivadas ─────────────────────────────────────────────

def compute_rows(results_dir: Path) -> list:
    rows = []

    for task in TASK_ORDER:
        filepath = results_dir / f'{task}_results.txt'
        if not filepath.exists():
            print(f'[WARN] {filepath} no encontrado — omitido', file=sys.stderr)
            continue

        data         = parse_results_file(filepath, task)
        class_labels = CLASS_LABELS[task]

        for mil in MIL_ORDER:
            for cls_id, cls_name in class_labels.items():
                none_cptac = data.get((mil, 'none',    'cptac', cls_id))
                none_tcga  = data.get((mil, 'none',    'tcga',  cls_id))
                mac_cptac  = data.get((mil, 'macenko', 'cptac', cls_id))

                if any(v is None for v in [none_cptac, none_tcga, mac_cptac]):
                    print(f'[WARN] dato faltante: task={task} mil={mil} '
                          f'cls={cls_id}', file=sys.stderr)
                    continue

                delta_n = mac_cptac - none_cptac
                rpd     = ((none_cptac - none_tcga) / (-none_tcga)
                           if none_tcga != 0 else float('nan'))

                rows.append(dict(
                    task=task.upper(),
                    mil=mil,
                    cls_id=cls_id,
                    cls_name=cls_name,
                    delta_n=delta_n,
                    rpd=rpd,
                    none_cptac=none_cptac,
                    none_tcga=none_tcga,
                    mac_cptac=mac_cptac,
                ))

    return rows


# ── Generación LaTeX ──────────────────────────────────────────────────────────

def _fmt_delta(v: float) -> str:
    return f'${v:+.3f}$'


def _fmt_rpd(v: float) -> str:
    if abs(v) < 1e-9:
        return r'$\phantom{+}0.000$'
    if v >= 1.0 - 1e-9:
        return r'$+1.000$'
    return f'${v:+.3f}$'


def generate_latex(rows: list) -> str:
    # Organizar: mil → task → [filas ordenadas por cls_id]
    mil_task = defaultdict(lambda: defaultdict(list))
    for r in rows:
        mil_task[r['mil']][r['task']].append(r)

    for mil in mil_task:
        for task in mil_task[mil]:
            mil_task[mil][task].sort(key=lambda r: r['cls_id'])

    # Número de filas por MIL (para \multirow)
    mil_nrows = {mil: sum(len(v) for v in mil_task[mil].values())
                 for mil in MIL_ORDER}

    buf = []

    buf.append(r'\begin{table}[ht]')
    buf.append(r'\centering')
    buf.append(
        r'\caption{Comparison of MIL aggregators across tasks and classes. '
        r'PR-AUC is reported for binary tasks (ER, HER2, PR) and F1 for PAM50. '
        r'\textbf{none}: PR-AUC or F1 on CPTAC without stain normalisation. '
        r'\textbf{macenko}: same metric with Macenko stain normalisation. '
        r'$\Delta n = \text{macenko} - \text{none}$ on CPTAC.}'
    )
    buf.append(r'\label{tab:results_summary}')
    buf.append(r'\begin{tabular}{lllrrr}')
    buf.append(r'\toprule')
    buf.append(r'\textbf{MIL} & \textbf{Task} & \textbf{Class} '
               r'& \textbf{none} & \textbf{macenko} & $\Delta n$ \\')
    buf.append(r'\midrule')

    for mil_idx, mil in enumerate(MIL_ORDER):
        task_dict  = mil_task[mil]
        n_mil_rows = mil_nrows[mil]
        mil_disp   = MIL_DISPLAY[mil]

        tasks_present = [t.upper() for t in TASK_ORDER if t.upper() in task_dict]

        first_mil = True

        for task_idx, task in enumerate(tasks_present):
            task_rows  = task_dict[task]
            n_task     = len(task_rows)
            first_task = True

            for row in task_rows:
                cls_label = f"{row['cls_name']}\\,({row['cls_id']})"
                none_str  = f"{row['none_cptac']:.3f}"
                mac_str   = f"{row['mac_cptac']:.3f}"
                delta_str = _fmt_delta(row['delta_n'])

                mil_col  = (f'\\multirow{{{n_mil_rows}}}{{*}}{{{mil_disp}}}'
                            if first_mil else '')
                task_col = (f'\\multirow{{{n_task}}}{{*}}{{{TASK_DISPLAY.get(task, task)}}}'
                            if first_task else '')

                buf.append(f' {mil_col} & {task_col} & {cls_label}'
                           f' & {none_str} & {mac_str} & {delta_str} \\\\')

                first_mil  = False
                first_task = False

            if task_idx < len(tasks_present) - 1:
                buf.append(r'\cmidrule{2-6}')

        if mil_idx < len(MIL_ORDER) - 1:
            buf.append(r'\midrule')

    buf.append(r'\bottomrule')
    buf.append(r'\end{tabular}')
    buf.append(r'\end{table}')

    return '\n'.join(buf)


# ── Análisis univariante RPD ~ Δn ────────────────────────────────────────────

def univariate_analysis(rows: list) -> None:
    """
    Regresión lineal simple OLS:  RPD = β0 + β1·Δn + ε

    Se excluyen filas con RPD = NaN o RPD = 1.0 exacto
    (colapso total, p.ej. Her2 en CPTAC con F1=0 para todos los modelos).
    """
    valid = [r for r in rows if not np.isnan(r['rpd'])]

    x = np.array([r['delta_n'] for r in valid])
    y = np.array([r['rpd']     for r in valid])
    n = len(x)

    # ── Regresión OLS ─────────────────────────────────────────────────────
    slope, intercept, r_value, p_value, se_slope = stats.linregress(x, y)

    r2     = r_value ** 2
    y_hat  = intercept + slope * x
    ss_res = np.sum((y - y_hat) ** 2)
    se_res = np.sqrt(ss_res / (n - 2))

    t_crit  = stats.t.ppf(0.975, df=n - 2)
    ci_low  = slope - t_crit * se_slope
    ci_high = slope + t_crit * se_slope

    # ── Correlación de Spearman ───────────────────────────────────────────
    rho, p_spearman = stats.spearmanr(x, y)

    # ── Effect sizes ──────────────────────────────────────────────────────
    # Pearson r  (Cohen 1988: pequeño=0.10, mediano=0.30, grande=0.50)
    r_pearson = r_value

    # Cohen's f²  (pequeño=0.02, mediano=0.15, grande=0.35)
    f2 = r2 / (1 - r2) if r2 < 1.0 else float('inf')

    # η² (eta cuadrado): en regresión simple = R²
    eta2 = r2

    # Cohen's d equivalente a partir de r: d = 2r / sqrt(1-r²)
    cohen_d = 2 * r_pearson / np.sqrt(1 - r_pearson ** 2) if abs(r_pearson) < 1 else float('inf')

    def _label_r(r):
        r = abs(r)
        if r >= 0.50: return 'grande'
        if r >= 0.30: return 'mediano'
        if r >= 0.10: return 'pequeño'
        return 'despreciable'

    def _label_f2(f):
        if f >= 0.35: return 'grande'
        if f >= 0.15: return 'mediano'
        if f >= 0.02: return 'pequeño'
        return 'despreciable'

    def _label_d(d):
        d = abs(d)
        if d >= 0.80: return 'grande'
        if d >= 0.50: return 'mediano'
        if d >= 0.20: return 'pequeño'
        return 'despreciable'

    # ── Impresión ─────────────────────────────────────────────────────────
    sep = '─' * 60
    print(sep)
    print('ANÁLISIS UNIVARIANTE:  RPD ~ Δn')
    print(sep)
    print(f'  N (filas válidas)    : {n}  '
          f'(excluidas por NaN: {len(rows) - n})')
    print()
    print('  Regresión OLS')
    print(f'    Intercepto (β0)    : {intercept:+.4f}')
    print(f'    Pendiente  (β1)    : {slope:+.4f}  '
          f'(IC 95%: [{ci_low:+.4f}, {ci_high:+.4f}])')
    print(f'    SE residual        : {se_res:.4f}')
    print(f'    R²                 : {r2:.4f}')
    print(f'    p-valor (β1=0)     : {p_value:.4e}')
    print()
    print('  Correlación de Spearman')
    print(f'    ρ                  : {rho:+.4f}')
    print(f'    p-valor            : {p_spearman:.4e}')
    print()
    print('  Effect sizes')
    print(f'    Pearson r          : {r_pearson:+.4f}  → {_label_r(r_pearson)}')
    print(f'    η² (= R²)          :  {eta2:.4f}  → {_label_r(np.sqrt(eta2))}')
    print(f'    Cohen f²           :  {f2:.4f}  → {_label_f2(f2)}')
    print(f'    Cohen d (equiv.)   : {cohen_d:+.4f}  → {_label_d(cohen_d)}')
    print()

    sig       = p_value < 0.05
    direction = 'negativa' if slope < 0 else 'positiva'
    print('  Interpretación:')
    if sig:
        print(f'    Relación significativa (p={p_value:.3e}).')
        print(f'    Pendiente {direction}: mayor Δn → '
              + ('menor' if slope < 0 else 'mayor') + ' RPD.')
        print(f'    Δn explica el {r2 * 100:.1f}% de la varianza en RPD (R²).')
        print(f'    Tamaño del efecto: r={r_pearson:+.3f} ({_label_r(r_pearson)}), '
              f'f²={f2:.3f} ({_label_f2(f2)}).')
    else:
        print(f'    Relación NO significativa (p={p_value:.3e}, α=0.05).')
        print(f'    Δn no predice linealmente el RPD en esta muestra.')
    print(sep)
    print()

    # ── Correlaciones de Spearman por subgrupo ────────────────────────────
    print('  Correlaciones de Spearman por subgrupo')
    print(f'  {"Subgrupo":<20}  {"N":>3}  {"ρ":>7}  {"p":>10}')
    print('  ' + '─' * 46)

    print('  Por tarea:')
    for g in sorted(set(r['task'] for r in valid)):
        sub = [r for r in valid if r['task'] == g]
        if len(sub) < 3:
            continue
        xi = np.array([r['delta_n'] for r in sub])
        yi = np.array([r['rpd']     for r in sub])
        rho_g, p_g = stats.spearmanr(xi, yi)
        mark = '*' if p_g < 0.05 else ' '
        print(f'    {g:<18}  {len(sub):>3}  {rho_g:>+7.3f}  {p_g:>10.4e} {mark}')
    print()


# ── Diagrama de barras por tarea ─────────────────────────────────────────────

# Datos CLAM sin optimización Optuna (MCCV=train proxy, Hold-out=CPTAC)
CLAM_NOOPT = {
    ('PAM50', 0): (0.807, 0.756),
    ('PAM50', 1): (0.377, 0.000),
    ('PAM50', 2): (0.833, 0.767),
    ('PAM50', 3): (0.549, 0.265),
    ('PAM50', 4): (0.146, 0.000),
    ('ER',    0): (0.590, 0.509),
    ('ER',    1): (0.901, 0.687),
    ('PR',    0): (0.579, 0.694),
    ('PR',    1): (0.793, 0.716),
    ('ERBB2', 0): (0.859, 0.891),
    ('ERBB2', 1): (0.194, 0.109),
}


def plot_bar_charts(rows: list, results_dir: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

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
    METRIC_LABEL = {
        'ER': 'PR-AUC', 'ERBB2': 'PR-AUC', 'PR': 'PR-AUC', 'PAM50': 'F1',
    }

    ALL_MODELS = MIL_ORDER + ['clam_noopt']
    tasks = [t.upper() for t in TASK_ORDER]

    figs_data = []  # collect per-task data for combined figure
    for task in tasks:
        task_rows = [r for r in rows if r['task'] == task]
        if not task_rows:
            continue

        classes   = sorted(set(r['cls_id'] for r in task_rows))
        cls_names = {r['cls_id']: r['cls_name'] for r in task_rows}
        x_labels  = [cls_names[c] for c in classes]
        n_cls     = len(classes)
        n_all     = len(ALL_MODELS)
        width     = 0.18
        x         = np.arange(n_cls)

        fig, axes = plt.subplots(1, 2, figsize=(max(9, n_cls * 2.4), 5),
                                 sharey=False)
        # no title

        for ax, (dataset_key, dataset_label) in zip(
                axes, [('none_tcga',  'TCGA (train)'),
                       ('none_cptac', 'CPTAC (test)')]):

            for i, mil in enumerate(ALL_MODELS):
                values = []
                for cls_id in classes:
                    if mil == 'clam_noopt':
                        mccv_val, ho_val = CLAM_NOOPT.get((task, cls_id), (0.0, 0.0))
                        val = mccv_val if dataset_key == 'none_tcga' else ho_val
                    else:
                        match = [r for r in task_rows
                                 if r['mil'] == mil and r['cls_id'] == cls_id]
                        val = match[0][dataset_key] if match else 0.0
                    values.append(val)

                offset = (i - (n_all - 1) / 2) * width
                edge_col = '#555555' if mil == 'clam_noopt' else '#7B1B3A'
                bars = ax.bar(x + offset, values, width,
                              color=COLOR[mil], edgecolor=edge_col,
                              linewidth=0.6, label=LABEL[mil])

                for bar, v in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.01,
                            f'{v:.2f}', ha='center', va='bottom',
                            fontsize=6, color='#333333' if mil == 'clam_noopt' else '#4A0020')

            ax.set_title(dataset_label, fontsize=11, color='#8B1A4A')
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, fontsize=9)
            ax.set_ylabel(METRIC_LABEL[task], fontsize=10)
            ax.set_ylim(0, 1.15)
            ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='#E8A0B4')
            ax.set_axisbelow(True)
            ax.spines[['top', 'right']].set_visible(False)
            for spine in ax.spines.values():
                spine.set_edgecolor('#C47A9A')

        legend_patches = [mpatches.Patch(facecolor=COLOR[m],
                                         edgecolor='#555555' if m == 'clam_noopt' else '#7B1B3A',
                                         label=LABEL[m]) for m in ALL_MODELS]
        fig.legend(handles=legend_patches, loc='lower center', ncol=4,
                   fontsize=10, framealpha=0.8,
                   bbox_to_anchor=(0.5, -0.06),
                   edgecolor='#C47A9A')

        plt.tight_layout(rect=[0, 0.08, 1, 1])
        out = results_dir / f'barplot_{task.lower()}.eps'
        fig.savefig(out, format='eps', bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f'  Guardado: {out}', file=sys.stderr)
        figs_data.append((task, task_rows, classes, cls_names, x_labels))


    # ── Figura combinada: fila 1 = PAM50, fila 2 = ER + PR + HER2 ──────────
    # GridSpec: 2 filas × 8 cols
    # PAM50: cols 0-3 (TCGA) y 4-7 (CPTAC)
    # Binarias: ER cols 0-1, PR cols 3-4, HER2 cols 6-7 (cols 2,5 vacías = separador)
    import matplotlib.gridspec as gridspec
    fig_all = plt.figure(figsize=(20, 9))
    gs = gridspec.GridSpec(2, 8, figure=fig_all,
                           hspace=0.45, wspace=0.30,
                           width_ratios=[2, 2, 0.5, 2, 2, 0.5, 2, 2])

    # Índice de figs_data por tarea
    fd = {d[0]: d for d in figs_data}
    TASK_ORDER_COMBINED = ['PAM50', 'ER', 'PR', 'ERBB2']

    def _fill_ax(ax, task, task_rows, classes, cls_names, x_labels, dataset_key, dataset_label):
        n_all = len(ALL_MODELS)
        width = 0.18
        x = np.arange(len(classes))
        for i, mil in enumerate(ALL_MODELS):
            values = []
            for cls_id in classes:
                if mil == 'clam_noopt':
                    mccv_val, ho_val = CLAM_NOOPT.get((task, cls_id), (0.0, 0.0))
                    val = mccv_val if dataset_key == 'none_tcga' else ho_val
                else:
                    match = [r for r in task_rows if r['mil'] == mil and r['cls_id'] == cls_id]
                    val = match[0][dataset_key] if match else 0.0
                values.append(val)
            offset = (i - (n_all - 1) / 2) * width
            edge_col = '#555555' if mil == 'clam_noopt' else '#7B1B3A'
            bars = ax.bar(x + offset, values, width,
                          color=COLOR[mil], edgecolor=edge_col, linewidth=0.6)
            for bar, v in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01, f'{v:.2f}',
                        ha='center', va='bottom',
                        fontsize=5.5, color='#333333' if mil == 'clam_noopt' else '#4A0020')
        ax.set_title(f'{TASK_DISPLAY.get(task, task)} — {dataset_label}', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_ylabel(METRIC_LABEL[task], fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='#E8A0B4')
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        for spine in ax.spines.values():
            spine.set_edgecolor('#C47A9A')

    # Fila 0: PAM50 (cols 0-2 TCGA, cols 3-5 CPTAC)
    if 'PAM50' in fd:
        _, task_rows, classes, cls_names, x_labels = fd['PAM50']
        ax_t = fig_all.add_subplot(gs[0, :4])
        ax_c = fig_all.add_subplot(gs[0, 4:])
        _fill_ax(ax_t, 'PAM50', task_rows, classes, cls_names, x_labels, 'none_tcga', 'TCGA')
        _fill_ax(ax_c, 'PAM50', task_rows, classes, cls_names, x_labels, 'none_cptac', 'CPTAC')

    # Fila 1: ER (cols 0-1), PR (cols 2-3), HER2/ERBB2 (cols 4-5)
    binary_tasks = [t for t in ['ER', 'PR', 'ERBB2'] if t in fd]
    binary_col_starts = [0, 3, 6]  # col 2 y 5 son separadores vacíos
    for bt_idx, bt in enumerate(binary_tasks):
        _, task_rows, classes, cls_names, x_labels = fd[bt]
        col_start = binary_col_starts[bt_idx]
        ax_t = fig_all.add_subplot(gs[1, col_start])
        ax_c = fig_all.add_subplot(gs[1, col_start + 1])
        _fill_ax(ax_t, bt, task_rows, classes, cls_names, x_labels, 'none_tcga', 'TCGA')
        _fill_ax(ax_c, bt, task_rows, classes, cls_names, x_labels, 'none_cptac', 'CPTAC')

    legend_patches = [mpatches.Patch(facecolor=COLOR[m],
                                     edgecolor='#555555' if m == 'clam_noopt' else '#7B1B3A',
                                     label=LABEL[m]) for m in ALL_MODELS]
    fig_all.legend(handles=legend_patches, loc='lower center', ncol=4,
                   fontsize=10, framealpha=0.8,
                   bbox_to_anchor=(0.5, -0.02), edgecolor='#C47A9A')
    for ext in ('eps', 'png'):
        out_all = results_dir / f'barplot_all.{ext}'
        fig_all.savefig(out_all, format=ext, bbox_inches='tight', facecolor='white')
        print(f'  Guardado: {out_all}', file=sys.stderr)
    plt.close(fig_all)


# ── Datos de prevalencia por clase ────────────────────────────────────────────
# Δp = p_CPTAC - p_TCGA  (positivo → más frecuente en CPTAC)

PREVALENCE_SHIFT: dict[tuple, float] = {
    # task_upper, cls_id → Δp
    ('ER',    0): +0.159,   # ER-negative
    ('ER',    1): -0.159,   # ER-positive
    ('PR',    0): +0.127,   # PR-negative
    ('PR',    1): -0.127,   # PR-positive
    ('ERBB2', 0): +0.049,   # HER2-negative
    ('ERBB2', 1): -0.049,   # HER2-positive
    ('PAM50', 0): +0.131,   # Basal
    ('PAM50', 1): +0.023,   # HER2-enriched
    ('PAM50', 2): -0.071,   # Luminal A
    ('PAM50', 3): +0.085,   # Luminal B (≈ 0.114-0.198 rounded)
    ('PAM50', 4): +0.001,   # Normal-like
}


# ── Carga de separabilidad morfológica ────────────────────────────────────────
# B̃_c = min_{d≠c} B_{c,d} − B_{c,c}   (ecuación del paper)

BIO_DIR = Path(__file__).resolve().parent.parent / 'results'

MORPH_SEP_MATRICES = {
    'PAM50': {
        'csv': 'biological_comparison_BASAL_HER2-ENRICHED_LUMINAL-A_LUMINAL-B_NORMAL-LIKE_matriz.csv',
        'labels': ['BASAL', 'HER2-ENRICHED', 'LUMINAL-A', 'LUMINAL-B', 'NORMAL-LIKE'],
        'cls_ids': [0, 1, 2, 3, 4],
    },
    'ER': {
        'csv': 'biological_comparison_ER-NEGATIVE_ER-POSITIVE_matriz.csv',
        'labels': ['ER-NEGATIVE', 'ER-POSITIVE'],
        'cls_ids': [0, 1],
    },
    'PR': {
        'csv': 'biological_comparison_PR-NEGATIVE_PR-POSITIVE_matriz.csv',
        'labels': ['PR-NEGATIVE', 'PR-POSITIVE'],
        'cls_ids': [0, 1],
    },
    'ERBB2': {
        'csv': 'biological_comparison_HER2-NEGATIVE_HER2-POSITIVE_matriz.csv',
        'labels': ['HER2-NEGATIVE', 'HER2-POSITIVE'],
        'cls_ids': [0, 1],
    },
}


def load_morphological_separability() -> dict:
    """Devuelve {(task_upper, cls_id): B̃_c}."""
    import csv
    result = {}
    for task, info in MORPH_SEP_MATRICES.items():
        p = BIO_DIR / info['csv']
        if not p.exists():
            print(f'[WARN] No encontrado: {p}', file=sys.stderr)
            continue
        with open(p) as f:
            rows_csv = list(csv.DictReader(f))
        n = len(info['labels'])
        mat = [[float(rows_csv[i][col]) for col in info['labels']] for i in range(n)]
        for i, cls_id in enumerate(info['cls_ids']):
            B_c = mat[i][i]
            off_diag = [mat[i][j] for j in range(n) if j != i]
            result[(task, cls_id)] = min(off_diag) - B_c
    return result


# ── Carga de distancias de centroides ─────────────────────────────────────────

PATCH_INTER_DIR = Path(__file__).resolve().parent.parent / 'results' / 'patch_intersection'


def load_centroid_distances(top_k: int = 8) -> dict:
    """Devuelve {(task_upper, cls_id): centroid_dist} como media de los 3 modelos
    (d_mean de patch_intersection_all/), coherente con Δn que también es media de 3."""
    dists = {}
    for task in TASK_ORDER:
        csv_path = PATCH_INTER_ALL_DIR / task / 'centroid_distances_all.csv'
        if not csv_path.exists():
            print(f'[WARN] No encontrado: {csv_path}', file=sys.stderr)
            continue
        import csv
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                dists[(task.upper(), int(row['class']))] = float(row['d_mean'])
    return dists



PATCH_INTER_ALL_DIR = Path(__file__).resolve().parent.parent / 'results' / 'patch_intersection_all'

MIL_DC_COL = {
    'clam_mil_mb': 'd_clam',
    'dsmil':       'd_dsmil',
    'transmil':    'd_transmil',
}

def load_centroid_distances_per_model() -> dict:
    """Devuelve {(mil, task_upper, cls_id): centroid_dist} desde patch_intersection_all/."""
    dists = {}
    for task in TASK_ORDER:
        csv_path = PATCH_INTER_ALL_DIR / task / 'centroid_distances_all.csv'
        if not csv_path.exists():
            continue
        import csv as _csv
        with open(csv_path) as f:
            for row in _csv.DictReader(f):
                cls_id = int(row['class'])
                for mil, col in MIL_DC_COL.items():
                    val = row.get(col)
                    if val:
                        dists[(mil, task.upper(), cls_id)] = float(val)
    return dists

# ── Tabla resumen: medias de los 3 modelos Opt- + distancia de centroide ──────

def generate_summary_latex(rows_mean: list, centroid_dists: dict,
                           morph_sep: dict = None) -> str:
    METRIC_LABEL = {'ER': 'PR-AUC', 'ERBB2': 'PR-AUC', 'PR': 'PR-AUC', 'PAM50': 'F1'}

    def _task_rows_sorted(task):
        return sorted([r for r in rows_mean if r['task'] == task],
                      key=lambda r: r['cls_id'])

    # ── Tabla 1: TCGA / CPTAC / RPD ──────────────────────────────────────────
    buf1 = []
    buf1.append(r'\begin{table}[ht]')
    buf1.append(r'\centering')
    buf1.append(
        r'\caption{Mean performance across CLAM-MB, DSMIL and TransMIL. '
        r'TCGA: MCCV score (train/val, no stain normalisation); '
        r'CPTAC: hold-out external test score (no stain normalisation). '
        r'RPD: relative performance drop, $\mathrm{RPD}=(\mathrm{CPTAC}-\mathrm{TCGA})\,/\,(-\mathrm{TCGA})$.}'
    )
    buf1.append(r'\label{tab:summary_performance}')
    buf1.append(r'\begin{tabular}{llcrrr}')
    buf1.append(r'\toprule')
    buf1.append(r'\textbf{Task} & \textbf{Class} & \textbf{Metric} & \textbf{TCGA} & \textbf{CPTAC} & \textbf{RPD} \\')
    buf1.append(r'\midrule')

    for task in [t.upper() for t in TASK_ORDER]:
        task_rows = _task_rows_sorted(task)
        if not task_rows:
            continue
        metric = METRIC_LABEL[task]
        n = len(task_rows)
        for i, r in enumerate(task_rows):
            task_col = f'\\multirow{{{n}}}{{*}}{{{TASK_DISPLAY.get(task, task)}}}' if i == 0 else ''
            met_col  = f'\\multirow{{{n}}}{{*}}{{{metric}}}' if i == 0 else ''
            tcga_val  = f"{r['none_tcga_mean']:.3f}"
            cptac_val = f"{r['none_cptac_mean']:.3f}"
            rpd_str   = '---' if np.isnan(r['rpd']) else f"${r['rpd']:+.3f}$"
            cls_name_ = r['cls_name']
            buf1.append(f'  {task_col} & {cls_name_} & {met_col} '
                        f'& {tcga_val} & {cptac_val} & {rpd_str} \\\\')
        buf1.append(r'\midrule' if task != 'PAM50' else r'\bottomrule')

    buf1.append(r'\end{tabular}')
    buf1.append(r'\end{table}')

    # ── Tabla 2: variables independientes + RPD ───────────────────────────────
    buf2 = []
    buf2.append(r'\begin{table}[ht]')
    buf2.append(r'\centering')
    buf2.append(
        r'\caption{Independent variables used in the statistical analysis and RPD (mean across models). '
        r'$\Delta n$: Macenko gain on CPTAC (Macenko$-$none). '
        r'$d_c$: cosine distance between TCGA and CPTAC class centroids in Virchow2 space '
        r'(top-8 attention patches, mean of three MIL models). '
        r'$\Delta p$: prevalence shift ($p_{\mathrm{CPTAC}}-p_{\mathrm{TCGA}}$). '
        r'$\tilde{B}_c$: morphological separability ($\min_{d\neq c}B_{c,d}-B_{c,c}$).}'
    )
    buf2.append(r'\label{tab:summary_predictors}')
    buf2.append(r'\begin{tabular}{llrrrrrr}')
    buf2.append(r'\toprule')
    buf2.append(r'\textbf{Task} & \textbf{Class} & $\Delta n$ & $d_c$ & $\Delta p$ & $\tilde{B}_c$ & \textbf{RPD} \\')
    buf2.append(r'\midrule')

    for task in [t.upper() for t in TASK_ORDER]:
        task_rows = _task_rows_sorted(task)
        if not task_rows:
            continue
        n = len(task_rows)
        for i, r in enumerate(task_rows):
            task_col = f'\\multirow{{{n}}}{{*}}{{{TASK_DISPLAY.get(task, task)}}}' if i == 0 else ''
            dn_str   = f"${r['delta_n']:+.3f}$"
            rpd_str  = '---' if np.isnan(r['rpd']) else f"${r['rpd']:+.3f}$"
            dist     = centroid_dists.get((task, r['cls_id']))
            dist_str = f"{dist:.3f}" if dist is not None else '---'
            dp       = PREVALENCE_SHIFT.get((task, r['cls_id']))
            dp_str   = f'${dp:+.3f}$' if dp is not None else '---'
            bs       = morph_sep.get((task, r['cls_id'])) if morph_sep else None
            bs_str   = f'${bs:+.3f}$' if bs is not None else '---'
            cls_name_ = r['cls_name']
            buf2.append(f'  {task_col} & {cls_name_} '
                        f'& {dn_str} & {dist_str} & {dp_str} & {bs_str} & {rpd_str} \\\\')
        buf2.append(r'\midrule' if task != 'PAM50' else r'\bottomrule')

    buf2.append(r'\end{tabular}')
    buf2.append(r'\end{table}')

    return '\n'.join(buf1) + '\n\n' + '\n'.join(buf2)


# ── Análisis univariante RPD ~ |Δp| ──────────────────────────────────────────

def univariate_prevalence(rows_mean: list) -> None:
    valid = []
    for r in rows_mean:
        dp = PREVALENCE_SHIFT.get((r['task'], r['cls_id']))
        if dp is None or np.isnan(r['rpd']):
            continue
        valid.append({'rpd': r['rpd'], 'dp': dp,
                      'task': r['task'], 'cls_id': r['cls_id']})

    if len(valid) < 4:
        print('[WARN] Insuficientes datos para regresión RPD ~ Δp',
              file=sys.stderr)
        return

    x = np.array([v['dp'] for v in valid])
    y = np.array([v['rpd']    for v in valid])
    n = len(x)

    slope, intercept, r_value, p_value, se_slope = stats.linregress(x, y)
    r2    = r_value ** 2
    y_hat = intercept + slope * x
    ss_res = np.sum((y - y_hat) ** 2)
    se_res = np.sqrt(ss_res / (n - 2))
    t_crit = stats.t.ppf(0.975, df=n - 2)
    ci_low, ci_high = slope - t_crit * se_slope, slope + t_crit * se_slope
    rho, p_spearman = stats.spearmanr(x, y)

    sep = '─' * 60
    print(sep)
    print('ANÁLISIS UNIVARIANTE:  RPD ~ Δp')
    print(sep)
    print(f'  N (filas válidas): {n}')
    print()
    print('  Regresión OLS')
    print(f'    Intercepto (β0): {intercept:+.4f}')
    print(f'    Pendiente  (β1): {slope:+.4f}  '
          f'(IC 95%: [{ci_low:+.4f}, {ci_high:+.4f}])')
    print(f'    SE residual    : {se_res:.4f}')
    print(f'    R²             : {r2:.4f}')
    print(f'    p-valor (β1=0) : {p_value:.4e}')
    print()
    print('  Correlación de Spearman')
    print(f'    ρ              : {rho:+.4f}')
    print(f'    p-valor        : {p_spearman:.4e}')
    print(sep)
    print()


# ── Análisis univariante RPD ~ d(centroide) ───────────────────────────────────

def univariate_centroid(rows_mean: list, centroid_dists: dict) -> None:
    valid = []
    for r in rows_mean:
        dist = centroid_dists.get((r['task'], r['cls_id']))
        if dist is None or np.isnan(r['rpd']):
            continue
        valid.append({'rpd': r['rpd'], 'dist': dist,
                      'task': r['task'], 'cls_id': r['cls_id']})

    if len(valid) < 4:
        print('[WARN] Insuficientes datos para regresión RPD ~ d(centroide)',
              file=sys.stderr)
        return

    x = np.array([v['dist'] for v in valid])
    y = np.array([v['rpd']  for v in valid])
    n = len(x)

    slope, intercept, r_value, p_value, se_slope = stats.linregress(x, y)
    r2    = r_value ** 2
    y_hat = intercept + slope * x
    ss_res = np.sum((y - y_hat) ** 2)
    se_res = np.sqrt(ss_res / (n - 2))
    t_crit = stats.t.ppf(0.975, df=n - 2)
    ci_low, ci_high = slope - t_crit * se_slope, slope + t_crit * se_slope
    rho, p_spearman = stats.spearmanr(x, y)

    sep = '─' * 60
    print(sep)
    print('ANÁLISIS UNIVARIANTE:  RPD ~ d(centroide)')
    print(sep)
    print(f'  N (filas válidas): {n}')
    print()
    print('  Regresión OLS')
    print(f'    Intercepto (β0): {intercept:+.4f}')
    print(f'    Pendiente  (β1): {slope:+.4f}  '
          f'(IC 95%: [{ci_low:+.4f}, {ci_high:+.4f}])')
    print(f'    SE residual    : {se_res:.4f}')
    print(f'    R²             : {r2:.4f}')
    print(f'    p-valor (β1=0) : {p_value:.4e}')
    print()
    print('  Correlación de Spearman')
    print(f'    ρ              : {rho:+.4f}')
    print(f'    p-valor        : {p_spearman:.4e}')
    print(sep)
    print()


# ── Análisis univariante d(centroide) ~ B̃_c ──────────────────────────────────

def univariate_d_vs_morph(rows_mean: list, centroid_dists: dict,
                           morph_sep: dict) -> None:
    valid = []
    for r in rows_mean:
        dist = centroid_dists.get((r['task'], r['cls_id']))
        bs   = morph_sep.get((r['task'], r['cls_id']))
        if dist is None or bs is None:
            continue
        valid.append({'dist': dist, 'bs': bs,
                      'task': r['task'], 'cls_id': r['cls_id']})

    if len(valid) < 4:
        print('[WARN] Insuficientes datos para regresión d ~ B̃_c', file=sys.stderr)
        return

    x = np.array([v['bs']   for v in valid])
    y = np.array([v['dist'] for v in valid])
    n = len(x)

    slope, intercept, r_value, p_value, se_slope = stats.linregress(x, y)
    r2    = r_value ** 2
    y_hat = intercept + slope * x
    se_res = np.sqrt(np.sum((y - y_hat) ** 2) / (n - 2))
    t_crit = stats.t.ppf(0.975, df=n - 2)
    ci_low, ci_high = slope - t_crit * se_slope, slope + t_crit * se_slope
    rho, p_spearman = stats.spearmanr(x, y)

    sep = '─' * 60
    print(sep)
    print('ANÁLISIS UNIVARIANTE:  d(centroide) ~ B̃_c')
    print(sep)
    print(f'  N (filas válidas): {n}')
    print()
    print('  Datos:')
    print(f'  {"Task":<8} {"Cls":<8} {"B̃_c":>7} {"d":>7}')
    for v in valid:
        print(f'    {v["task"]:<8} {v["cls_id"]:<8} {v["bs"]:>+7.3f} {v["dist"]:>7.3f}')
    print()
    print('  Regresión OLS  (X = B̃_c,  Y = d)')
    print(f'    Intercepto (β0): {intercept:+.4f}')
    print(f'    Pendiente  (β1): {slope:+.4f}  '
          f'(IC 95%: [{ci_low:+.4f}, {ci_high:+.4f}])')
    print(f'    SE residual    : {se_res:.4f}')
    print(f'    R²             : {r2:.4f}')
    print(f'    p-valor (β1=0) : {p_value:.4e}')
    print()
    print('  Correlación de Spearman')
    print(f'    ρ              : {rho:+.4f}')
    print(f'    p-valor        : {p_spearman:.4e}')
    print(sep)
    print()


# ── Análisis univariante RPD ~ B̃_c ───────────────────────────────────────────

def univariate_rpd_vs_morph(rows_mean: list, morph_sep: dict) -> None:
    valid = []
    for r in rows_mean:
        bs  = morph_sep.get((r['task'], r['cls_id']))
        rpd = r.get('rpd')
        if bs is None or rpd is None:
            continue
        valid.append({'bs': bs, 'rpd': rpd,
                      'task': r['task'], 'cls_id': r['cls_id']})

    if len(valid) < 4:
        print('[WARN] Insuficientes datos para regresión RPD ~ B̃_c', file=sys.stderr)
        return

    x = np.array([v['bs']  for v in valid])
    y = np.array([v['rpd'] for v in valid])
    n = len(x)

    slope, intercept, r_value, p_value, se_slope = stats.linregress(x, y)
    r2    = r_value ** 2
    y_hat = intercept + slope * x
    se_res = np.sqrt(np.sum((y - y_hat) ** 2) / (n - 2))
    t_crit = stats.t.ppf(0.975, df=n - 2)
    ci_low, ci_high = slope - t_crit * se_slope, slope + t_crit * se_slope
    rho, p_spearman = stats.spearmanr(x, y)

    sep = '─' * 60
    print(sep)
    print('ANÁLISIS UNIVARIANTE:  RPD ~ B̃_c')
    print(sep)
    print(f'  N (filas válidas): {n}')
    print()
    print('  Datos:')
    print(f'  {"Task":<8} {"Cls":<8} {"B̃_c":>7} {"RPD":>7}')
    for v in valid:
        print(f'    {v["task"]:<8} {v["cls_id"]:<8} {v["bs"]:>+7.3f} {v["rpd"]:>+7.3f}')
    print()
    print('  Regresión OLS  (X = B̃_c,  Y = RPD)')
    print(f'    Intercepto (β0): {intercept:+.4f}')
    print(f'    Pendiente  (β1): {slope:+.4f}  '
          f'(IC 95%: [{ci_low:+.4f}, {ci_high:+.4f}])')
    print(f'    SE residual    : {se_res:.4f}')
    print(f'    R²             : {r2:.4f}')
    print(f'    p-valor (β1=0) : {p_value:.4e}')
    print()
    print('  Correlación de Spearman')
    print(f'    ρ              : {rho:+.4f}')
    print(f'    p-valor        : {p_spearman:.4e}')
    print(sep)
    print()


# ── Análisis multivariante RPD ~ Δn + d + B̃_c ────────────────────────────────

def multivariate_analysis(rows_mean: list, centroid_dists: dict,
                           morph_sep: dict) -> None:
    """
    Regresión OLS múltiple: RPD = β0 + β1·Δn + β2·d + β3·B̃_c + ε
    Se ajustan también modelos parciales para valorar contribución individual.
    Se reporta VIF para detectar multicolinealidad.
    """
    valid = []
    for r in rows_mean:
        dist = centroid_dists.get((r["task"], r["cls_id"]))
        bs   = morph_sep.get((r["task"], r["cls_id"]))
        if dist is None or bs is None or np.isnan(r["rpd"]):
            continue
        valid.append({
            "rpd": r["rpd"], "dn": r["delta_n"],
            "dist": dist, "bs": bs,
            "task": r["task"], "cls_id": r["cls_id"],
        })

    if len(valid) < 5:
        print("[WARN] Insuficientes datos para análisis multivariante", file=sys.stderr)
        return

    from scipy.linalg import lstsq as sp_lstsq

    rpd = np.array([v["rpd"]  for v in valid])
    dn  = np.array([v["dn"]   for v in valid])
    d   = np.array([v["dist"] for v in valid])
    bs  = np.array([v["bs"]   for v in valid])
    n   = len(valid)

    def ols(X, y):
        X1 = np.column_stack([np.ones(len(y)), X])
        k  = X1.shape[1] - 1
        b, _, _, _ = sp_lstsq(X1, y)
        y_hat  = X1 @ b
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2     = 1 - ss_res / ss_tot
        r2_adj = 1 - (1 - r2) * (n - 1) / (n - k - 1)
        ms_res = ss_res / (n - k - 1) if n - k - 1 > 0 else np.nan
        f_stat = ((ss_tot - ss_res) / k) / ms_res if ms_res else np.nan
        p_f    = 1 - stats.f.cdf(f_stat, k, n - k - 1) if not np.isnan(f_stat) else np.nan
        cov_b  = ms_res * np.linalg.pinv(X1.T @ X1) if ms_res else np.full((k+1, k+1), np.nan)
        se_b   = np.sqrt(np.diag(cov_b))
        t_b    = b / se_b
        p_b    = 2 * (1 - stats.t.cdf(np.abs(t_b), df=n - k - 1))
        return b, r2, r2_adj, f_stat, p_f, se_b, t_b, p_b

    def vif(X):
        vifs = []
        for i in range(X.shape[1]):
            _, r2_i, *_ = ols(np.delete(X, i, axis=1), X[:, i])
            vifs.append(1 / (1 - r2_i) if r2_i < 1 else np.inf)
        return vifs

    sep = "═" * 60
    print(sep)
    print("ANÁLISIS MULTIVARIANTE:  RPD ~ Δn + d + B̃_c")
    print(sep)
    print(f"  N = {n}")

    MODELS = [
        ("Δn + d + B̃_c", np.column_stack([dn, d, bs]), ["Δn", "d", "B̃_c"]),
        ("Δn + B̃_c",     np.column_stack([dn, bs]),    ["Δn", "B̃_c"]),
        ("d  + B̃_c",     np.column_stack([d,  bs]),    ["d",  "B̃_c"]),
        ("Δn + d",        np.column_stack([dn, d]),     ["Δn", "d"]),
    ]

    for label, X, names in MODELS:
        b, r2, r2_adj, f_stat, p_f, se_b, t_b, p_b = ols(X, rpd)
        print(f"\n  ── Modelo: RPD ~ {label} ──")
        print(f"    R²={r2:.3f}   R²_adj={r2_adj:.3f}   "
              f"F={f_stat:.2f}   p(F)={p_f:.4e}")
        for nm, bi, si, ti, pi in zip(["Intercept"] + names, b, se_b, t_b, p_b):
            sig = "*" if pi < 0.05 else (" †" if pi < 0.10 else "")
            print(f"    {nm:<12}  β={bi:+.4f}  SE={si:.4f}  "
                  f"t={ti:+.3f}  p={pi:.4e}{sig}")
        if X.shape[1] == 3:
            vifs = vif(X)
            print("    VIF: " + "  ".join(f"{nm}={v:.2f}" for nm, v in zip(names, vifs)))

    print("\n  (* p<0.05   † p<0.10)")
    print(sep)

    # ── Correlaciones entre predictores (Pearson vs Spearman) ────────────────
    predictors = [("Δn", dn), ("d", d), ("B̃_c", bs)]
    print("\n  Correlaciones entre predictores (linealidad vs. monotonicidad)")
    print(f"  {'Par':<16}  {'Pearson r':>10}  {'Spearman ρ':>11}  {'p(Spearman)':>12}")
    print("  " + "─" * 56)
    for i in range(len(predictors)):
        for j in range(i + 1, len(predictors)):
            na, xa = predictors[i]
            nb, xb = predictors[j]
            pr = np.corrcoef(xa, xb)[0, 1]
            rho_s, p_s = stats.spearmanr(xa, xb)
            par = f"{na} ~ {nb}"
            sig = "*" if p_s < 0.05 else ""
            print(f"  {par:<16}  {pr:>+10.3f}  {rho_s:>+11.3f}  {p_s:>12.4e} {sig}")
    print("  " + "─" * 56)
    print("  (Si |Pearson r| ≈ |Spearman ρ|, la relación es aproximadamente lineal.)")
    print(sep)
    print()


# ── Análisis confusiones PAM50 vs. disimilitud biológica ─────────────────────

PAM50_CLS = {0: 'Basal', 1: 'HER2-enr', 2: 'LumA', 3: 'LumB', 4: 'Normal'}
PAM50_LABELS = [PAM50_CLS[i] for i in range(5)]
BIO_RENAME = {
    'BASAL': 'Basal', 'HER2-ENRICHED': 'HER2-enr',
    'LUMINAL-A': 'LumA', 'LUMINAL-B': 'LumB', 'NORMAL-LIKE': 'Normal',
}
BIO_MATRIX_CSV = (
    PATCH_INTER_DIR.parent
    / 'biological_comparison_BASAL_HER2-ENRICHED_LUMINAL-A_LUMINAL-B_NORMAL-LIKE_matriz.csv'
)


def _parse_pam50_cms(results_dir: Path) -> dict:
    """Parsea matrices de confusión (absolutas) de pam50_results.txt.
    Devuelve {(mil, norm, dataset): ndarray 5×5}."""
    import csv as _csv
    filepath = results_dir / 'pam50_results.txt'
    if not filepath.exists():
        return {}

    lines = filepath.read_text(encoding='utf-8').splitlines()
    cur_mil = cur_norm = cur_ds = None
    in_cm = False
    cm_rows = []
    cms = {}

    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r'MIL:\s+(\S+)\s+\|\s+Norm:\s+(\S+)', line)
        if m:
            cur_mil, cur_norm = m.group(1), m.group(2)
            cur_ds = None; in_cm = False; i += 1; continue
        m = re.match(r'Dataset:\s+(\S+)', line)
        if m:
            cur_ds = m.group(1).lower(); in_cm = False; i += 1; continue
        if 'Confusion Matrix (filas=real' in line:
            in_cm = True; cm_rows = []; i += 2; continue  # salta cabecera cols
        if in_cm:
            m2 = re.match(r'\d+\(\d+\)((?:\s+\d+)+)', line)
            if m2:
                cm_rows.append(list(map(int, m2.group(1).split())))
                if len(cm_rows) == 5:
                    cms[(cur_mil, cur_norm, cur_ds)] = np.array(cm_rows)
                    in_cm = False
            else:
                in_cm = False
        i += 1
    return cms


def analyse_confusions_vs_biology(results_dir: Path) -> None:
    cms = _parse_pam50_cms(results_dir)
    if not cms:
        print('[WARN] No se encontraron matrices de confusión PAM50.', file=sys.stderr)
        return

    # Suma de los 3 modelos Optuna / none / cptac
    total_cm = np.zeros((5, 5), dtype=int)
    for mil in MIL_ORDER:
        key = (mil, 'none', 'cptac')
        if key in cms:
            total_cm += cms[key]
        else:
            print(f'[WARN] CM no encontrada: {key}', file=sys.stderr)

    labels = PAM50_LABELS
    df_total = _cm_df(total_cm, labels)

    # Off-diagonal: errores absolutos y % por clase real
    err_cm = total_cm.copy()
    np.fill_diagonal(err_cm, 0)
    df_err = _cm_df(err_cm, labels)

    row_totals = total_cm.sum(axis=1)
    df_err_pct = _cm_df(
        (err_cm / row_totals[:, None] * 100).round(1), labels)

    sep = '─' * 60
    print(sep)
    print('CONFUSIONES PAM50  —  3 modelos Optuna × CPTAC × none')
    print(sep)
    print(f'  Total slides evaluados: {total_cm.sum()} (3 × 387)')
    print()
    print('  Predicciones absolutas (filas=real, columnas=predicho):')
    print(df_total.to_string(col_space=10))
    print()
    print('  Errores absolutos (off-diagonal):')
    print(df_err.to_string(col_space=10))
    print(f'  Total errores: {err_cm.sum()}')
    print()
    print('  Errores (% sobre total real por clase):')
    print(df_err_pct.to_string(col_space=10))
    print()

    # Matriz biológica
    if not BIO_MATRIX_CSV.exists():
        print(f'[WARN] Matriz biológica no encontrada: {BIO_MATRIX_CSV}', file=sys.stderr)
        print(sep)
        return

    import csv as _csv
    bio_raw = {}
    with open(BIO_MATRIX_CSV) as f:
        reader = _csv.DictReader(f)
        col_keys = [BIO_RENAME.get(c, c) for c in reader.fieldnames[1:]]
        for row in reader:
            row_key = BIO_RENAME.get(row[reader.fieldnames[0]], row[reader.fieldnames[0]])
            bio_raw[row_key] = [float(row[c]) for c in reader.fieldnames[1:]]

    bio_arr = np.array([[bio_raw[r][j] for j in range(5)] for r in labels])
    df_bio = _cm_df(bio_arr, labels)

    print('  Matriz biológica (Σ|r| comparaciones sig., filas=TCGA, cols=CPTAC):')
    print(df_bio.round(3).to_string(col_space=10))
    print()

    # Correlación Spearman off-diagonal: disimilitud bio vs. tasa de confusión
    mask = ~np.eye(5, dtype=bool)
    x_bio = bio_arr[mask]
    x_err = (err_cm / row_totals[:, None])[mask]
    rho, p_sp = stats.spearmanr(x_bio, x_err)
    print('  Correlación Spearman off-diagonal (disim. bio. ~ tasa confusión):')
    print(f'    ρ = {rho:+.3f}   p = {p_sp:.4e}')
    if p_sp < 0.05:
        direction = 'mayor disimilitud biológica → más confusión' if rho > 0 else \
                    'mayor disimilitud biológica → menos confusión'
        print(f'    Significativa: {direction}')
    else:
        print('    No significativa: los errores no siguen la disimilitud biológica.')
    print(sep)
    print()


def _cm_df(arr, labels):
    import pandas as pd
    return pd.DataFrame(arr, index=labels, columns=labels)



# ── Confusion matrices for CLAM-MB CPTAC ───────────────────────────────────────────

def _parse_binary_cms(results_dir: Path, task: str) -> dict:
    """Parse confusion matrices from binary task results files.
    Returns {(mil, norm, dataset): ndarray NxN}."""
    filepath = results_dir / f'{task}_results.txt'
    if not filepath.exists():
        return {}

    lines = filepath.read_text(encoding='utf-8').splitlines()
    cur_mil = cur_norm = cur_ds = None
    in_cm = False
    cm_rows = []
    cms = {}
    n_classes = 2

    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r'MIL:\s+(\S+)\s+\|\s+Norm:\s+(\S+)', line)
        if m:
            cur_mil, cur_norm = m.group(1), m.group(2)
            cur_ds = None; in_cm = False; i += 1; continue
        m = re.match(r'Dataset:\s+(\S+)', line)
        if m:
            cur_ds = m.group(1).lower(); in_cm = False; i += 1; continue
        if 'Confusion Matrix (filas=real' in line:
            in_cm = True; cm_rows = []; i += 2; continue  # skip header
        if in_cm:
            m2 = re.match(r'\d+\(\d+\)((?:\s+\d+)+)', line)
            if m2:
                cm_rows.append(list(map(int, m2.group(1).split())))
                if len(cm_rows) == n_classes:
                    cms[(cur_mil, cur_norm, cur_ds)] = np.array(cm_rows)
                    in_cm = False
            else:
                in_cm = False
        i += 1
    return cms


def plot_confusion_matrices_clam(results_dir: Path) -> None:
    """Plot CLAM-MB CPTAC confusion matrices for all tasks as heatmaps."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    figures_dir = results_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)

    TASK_LABELS = {
        'pam50': ['Basal', 'Her2-e', 'LumA', 'LumB', 'Normal'],
        'er':    ['neg', 'pos'],
        'pr':    ['neg', 'pos'],
        'erbb2': ['neg', 'pos'],
    }

    for task in TASK_ORDER:
        labels = TASK_LABELS[task]
        n_cls  = len(labels)

        if task == 'pam50':
            cms = _parse_pam50_cms(results_dir)
        else:
            cms = _parse_binary_cms(results_dir, task)

        key = ('clam_mil_mb', 'none', 'cptac')
        if key not in cms:
            print(f'[WARN] CM not found for {task} clam_mil_mb none cptac', file=sys.stderr)
            continue

        cm = cms[key].astype(float)
        row_totals = cm.sum(axis=1, keepdims=True)
        cm_norm = np.where(row_totals > 0, cm / row_totals, 0.0)

        fig, axes = plt.subplots(1, 2, figsize=(max(8, n_cls * 2), max(4, n_cls * 1.5)))
        fig.suptitle(f'CLAM-MB  —  {task.upper()}  —  CPTAC (none)',
                     fontsize=12, fontweight='bold')

        # Absolute counts
        sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues',
                    xticklabels=labels, yticklabels=labels,
                    ax=axes[0], cbar=True, linewidths=0.5)
        axes[0].set_title('Absolute counts')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('True')

        # Recall-normalised
        annot_norm = np.array([[f'{v:.2f}' for v in row] for row in cm_norm])
        sns.heatmap(cm_norm, annot=annot_norm, fmt='', cmap='Blues',
                    xticklabels=labels, yticklabels=labels,
                    vmin=0, vmax=1,
                    ax=axes[1], cbar=True, linewidths=0.5)
        axes[1].set_title('Recall-normalised')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('True')

        plt.tight_layout()
        for ext in ('pdf', 'png'):
            out = figures_dir / f'cm_clam_{task}_cptac.{ext}'
            fig.savefig(out, bbox_inches='tight', facecolor='white')
            print(f'  Guardado: {out}', file=sys.stderr)
        plt.close(fig)


# ── Biological B_{{c,d}} matrix heatmaps ─────────────────────────────────────────

def plot_biological_matrix(results_dir: Path) -> None:
    """Plot full B_{{c,d}} biological comparison matrices as heatmaps."""
    import csv as _csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    figures_dir = results_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)

    DISPLAY_LABELS = {
        'PAM50': ['Basal', 'Her2-e', 'LumA', 'LumB', 'Normal'],
        'ER':    ['neg', 'pos'],
        'PR':    ['neg', 'pos'],
        'ERBB2': ['neg', 'pos'],
    }

    for task_upper, info in MORPH_SEP_MATRICES.items():
        p = BIO_DIR / info['csv']
        if not p.exists():
            print(f'[WARN] No encontrado: {p}', file=sys.stderr)
            continue

        with open(p) as f:
            rows_csv = list(_csv.DictReader(f))

        raw_labels = info['labels']
        n = len(raw_labels)
        mat = np.array(
            [[float(rows_csv[i][col]) for col in raw_labels] for i in range(n)]
        )

        display_labels = DISPLAY_LABELS[task_upper]
        vmax = np.abs(mat).max()
        vmax = vmax if vmax > 0 else 1.0

        fig, ax = plt.subplots(figsize=(max(5, n * 1.4), max(4, n * 1.2)))
        annot = np.array([[f'{v:.2f}' for v in row] for row in mat])
        sns.heatmap(mat, annot=annot, fmt='', cmap='RdBu_r',
                    vmin=-vmax, vmax=vmax,
                    xticklabels=display_labels, yticklabels=display_labels,
                    ax=ax, cbar=True, linewidths=0.5)
        ax.set_title(
            f'Biological matrix B_{{c,d}} — {task_upper}\n'
            f'Rows=TCGA classes, Cols=CPTAC classes | Values=\u03a3|effect_r| sig.',
            fontsize=10)
        ax.set_xlabel('CPTAC class')
        ax.set_ylabel('TCGA class')
        plt.tight_layout()

        task_lower = task_upper.lower()
        for ext in ('pdf', 'png'):
            out = figures_dir / f'bio_matrix_{task_lower}.{ext}'
            fig.savefig(out, bbox_inches='tight', facecolor='white')
            print(f'  Guardado: {out}', file=sys.stderr)
        plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    _arg = sys.argv[1] if len(sys.argv) > 1 else None
    results_dir = Path(_arg) if (_arg and Path(_arg).is_dir()) else RESULTS_DIR

    rows = compute_rows(results_dir)
    if not rows:
        print('[ERROR] No se extrajeron datos. Revisa los ficheros de resultados.',
              file=sys.stderr)
        sys.exit(1)

    # Agregar por (task, cls_id): media de los tres modelos (CLAM-MB, DSMIL, TransMIL)
    from itertools import groupby
    key_fn = lambda r: (r['task'], r['cls_id'], r['cls_name'])
    rows_mean = []
    for (task, cls_id, cls_name), group in groupby(
            sorted(rows, key=key_fn), key=key_fn):
        group = list(group)
        valid_rpd = [r['rpd'] for r in group if not np.isnan(r['rpd'])]
        rows_mean.append({
            'task':           task,
            'mil':            'mean',
            'cls_id':         cls_id,
            'cls_name':       cls_name,
            'delta_n':        np.mean([r['delta_n'] for r in group]),
            'rpd':            np.mean(valid_rpd) if valid_rpd else float('nan'),
            'none_tcga_mean': np.mean([r['none_tcga']  for r in group]),
            'none_cptac_mean':np.mean([r['none_cptac'] for r in group]),
        })

    centroid_dists = load_centroid_distances(top_k=8)

    univariate_analysis(rows_mean)
    univariate_prevalence(rows_mean)
    univariate_centroid(rows_mean, centroid_dists)
    morph_sep_early = load_morphological_separability()
    univariate_d_vs_morph(rows_mean, centroid_dists, morph_sep_early)
    univariate_rpd_vs_morph(rows_mean, morph_sep_early)
    multivariate_analysis(rows_mean, centroid_dists, morph_sep_early)
    analyse_confusions_vs_biology(results_dir)

    print('Generando matrices de confusión CLAM-MB...', file=sys.stderr)
    plot_confusion_matrices_clam(results_dir)
    print('Generando heatmaps de matrices biológicas...', file=sys.stderr)
    plot_biological_matrix(results_dir)
    print('Generando morfología vs confusión...', file=sys.stderr)
    plot_morphology_vs_confusion(results_dir, show_diag=True)
    plot_morphology_vs_confusion(results_dir, show_diag=False)
    print('Generando diagramas de barras...', file=sys.stderr)
    plot_bar_charts(rows, results_dir)

    latex = generate_latex(rows)
    out_path = results_dir / 'results_table.tex'
    out_path.write_text(latex, encoding='utf-8')
    print(f'% LaTeX guardado en {out_path}', file=sys.stderr)

    morph_sep = load_morphological_separability()
    summary_latex = generate_summary_latex(rows_mean, centroid_dists, morph_sep)
    summary_out_path = results_dir / 'summary_table.tex'
    summary_out_path.write_text(summary_latex, encoding='utf-8')
    print(f'% LaTeX guardado en {summary_out_path}', file=sys.stderr)

    perf_mil_latex = generate_performance_per_mil_latex(rows)
    perf_mil_path = results_dir / 'performance_per_mil_table.tex'
    perf_mil_path.write_text(perf_mil_latex, encoding='utf-8')
    print(f'% LaTeX guardado en {perf_mil_path}', file=sys.stderr)

    centroid_latex = generate_centroid_latex()
    centroid_out_path = results_dir / 'centroid_distances_table.tex'
    centroid_out_path.write_text(centroid_latex, encoding='utf-8')
    print(f'% LaTeX guardado en {centroid_out_path}', file=sys.stderr)

    stats_latex = generate_stats_latex(rows_mean, centroid_dists, morph_sep)
    stats_out_path = results_dir / 'stats_table.tex'
    stats_out_path.write_text(stats_latex, encoding='utf-8')
    print(f'% LaTeX guardado en {stats_out_path}', file=sys.stderr)


def plot_morphology_vs_confusion(results_dir: Path) -> None:
    """Side-by-side heatmaps (bio matrix | confusion %) + scatter, tonos rosados."""
    import csv as _csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import seaborn as sns
    import numpy as np

    figures_dir = results_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)

    PINK_CMAP = mcolors.LinearSegmentedColormap.from_list(
        'pink_mono', ['#FFFFFF', '#AD1457'], N=256)
    PINK_DIV  = mcolors.LinearSegmentedColormap.from_list(
        'pink_div', ['#AD1457', '#FADADD', '#FFFFFF', '#FADADD', '#AD1457'], N=256)

    DISPLAY_LABELS = {
        'PAM50': ['Basal', 'Her2-e', 'LumA', 'LumB', 'Normal'],
        'ER':    ['neg', 'pos'],
        'PR':    ['neg', 'pos'],
        'ERBB2': ['neg', 'pos'],
    }

    # ── Cargar matrices de confusión (3 modelos, none, cptac) ─────────────────
    def _load_cms():
        """Devuelve {task: error_pct_matrix (ndarray)}."""
        cms = {}
        for task in TASK_ORDER:
            task_up = task.upper()
            n_cls = {'er': 2, 'erbb2': 2, 'pr': 2, 'pam50': 5}[task]
            total = np.zeros((n_cls, n_cls), dtype=float)
            for mil in MIL_ORDER:
                exp_dir = Path('/shared/home/jorgarcia/PathBench-MIL/experiments') / f'brca_virchow2_test_{task}_{mil}'
                pred_dir = exp_dir / 'mil_eval'
                none_dirs = sorted(pred_dir.glob('00001-*none*'))
                if not none_dirs:
                    continue
                pred_file = none_dirs[0] / '00000-*/predictions.csv'
                import glob as _glob
                files = _glob.glob(str(pred_file))
                if not files:
                    continue
                import pandas as _pd
                df = _pd.read_csv(files[0])
                if 'y_true' not in df.columns or 'y_pred_class' not in df.columns:
                    continue
                from sklearn.metrics import confusion_matrix as _cm
                cm = _cm(df['y_true'], df['y_pred_class'],
                         labels=list(range(n_cls)))
                total += cm
            # error % per row
            row_tot = total.sum(axis=1, keepdims=True)
            row_tot[row_tot == 0] = 1
            err = total.copy()
            np.fill_diagonal(err, 0)
            cms[task_up] = (err / row_tot * 100)
        return cms

    conf_mats = _load_cms()

    # ── Per-task: side-by-side heatmaps ──────────────────────────────────────
    for task_upper, info in MORPH_SEP_MATRICES.items():
        p = BIO_DIR / info['csv']
        if not p.exists():
            continue
        with open(p) as f:
            rows_csv = list(_csv.DictReader(f))
        raw_labels = info['labels']
        n = len(raw_labels)
        bio_mat = np.array(
            [[float(rows_csv[i][col]) for col in raw_labels] for i in range(n)]
        )
        disp = DISPLAY_LABELS[task_upper]
        conf_mat = conf_mats.get(task_upper)

        fig, axes = plt.subplots(1, 2, figsize=(max(10, n * 2.8), max(4, n * 1.4)))
        fig.subplots_adjust(wspace=0.4)

        # Bio matrix
        vmax = max(np.abs(bio_mat).max(), 0.1)
        sns.heatmap(bio_mat, annot=True, fmt='.2f', cmap=PINK_CMAP,
                    vmin=0, vmax=vmax,
                    xticklabels=disp, yticklabels=disp,
                    ax=axes[0], linewidths=0.5, linecolor='#E8A0B4',
                    cbar_kws={'label': r'$B_{c,d}$'})
        axes[0].set_title(r'Morphological distance $B_{c,d}$', fontsize=11)
        axes[0].set_xlabel('CPTAC class', fontsize=9)
        axes[0].set_ylabel('TCGA class', fontsize=9)

        # Confusion % matrix
        if conf_mat is not None:
            vmax_c = max(conf_mat.max(), 1.0)
            sns.heatmap(conf_mat, annot=True, fmt='.1f', cmap=PINK_CMAP,
                        vmin=0, vmax=vmax_c,
                        xticklabels=disp, yticklabels=disp,
                        ax=axes[1], linewidths=0.5, linecolor='#E8A0B4',
                        cbar_kws={'label': 'Error rate (%)'})
            axes[1].set_title('Confusion rate (%, mean 3 models, CPTAC)', fontsize=11)
            axes[1].set_xlabel('Predicted class', fontsize=9)
            axes[1].set_ylabel('True class', fontsize=9)
        else:
            axes[1].axis('off')

        task_disp = TASK_DISPLAY.get(task_upper, task_upper)
        fig.suptitle(f'{task_disp}', fontsize=13, fontweight='bold', color='#6D1B3A')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        for ext in ('pdf', 'png'):
            out = figures_dir / f'morph_vs_conf_{task_upper.lower()}.{ext}'
            fig.savefig(out, bbox_inches='tight', dpi=200, facecolor='white')
            print(f'  Guardado: {out}', file=sys.stderr)
        plt.close(fig)

    # ── Scatter global: B_{c,d} off-diagonal vs confusion % ──────────────────
    x_all, y_all, c_all, t_all = [], [], [], []
    TASK_COLORS = {'ER': '#F06292', 'ERBB2': '#9b59b6',
                   'PR': '#AD1457', 'PAM50': '#FADADD'}

    for task_upper, info in MORPH_SEP_MATRICES.items():
        p = BIO_DIR / info['csv']
        if not p.exists():
            continue
        with open(p) as f:
            rows_csv = list(_csv.DictReader(f))
        raw_labels = info['labels']
        n = len(raw_labels)
        bio_mat = np.array(
            [[float(rows_csv[i][col]) for col in raw_labels] for i in range(n)]
        )
        conf_mat = conf_mats.get(task_upper)
        if conf_mat is None:
            continue
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                x_all.append(bio_mat[i, j])
                y_all.append(conf_mat[i, j])
                c_all.append(TASK_COLORS.get(task_upper, '#F06292'))
                t_all.append(task_upper)

    if x_all:
        from scipy import stats as _stats
        x_arr, y_arr = np.array(x_all), np.array(y_all)
        rho, p_rho = _stats.spearmanr(x_arr, y_arr)
        slope, intercept, r, p_ols, _ = _stats.linregress(x_arr, y_arr)

        fig, ax = plt.subplots(figsize=(6, 5))
        for task_upper in MORPH_SEP_MATRICES:
            idx = [i for i, t in enumerate(t_all) if t == task_upper]
            ax.scatter([x_all[i] for i in idx], [y_all[i] for i in idx],
                       color=TASK_COLORS.get(task_upper, '#F06292'),
                       s=60, edgecolors='#4A0020', linewidth=0.6,
                       label=TASK_DISPLAY.get(task_upper, task_upper), zorder=3)

        x_line = np.linspace(x_arr.min(), x_arr.max(), 200)
        ax.plot(x_line, intercept + slope * x_line,
                color='#AD1457', linewidth=1.5, linestyle='--')

        ax.set_xlabel(r'$B_{c,d}$ (morphological distance)', fontsize=11)
        ax.set_ylabel('Confusion rate (%, mean 3 models)', fontsize=11)
        ax.set_title(
            (fr'$\rho={rho:+.3f}$, $p={p_rho:.3f}$'
             ' — off-diagonal pairs'),
            fontsize=10, color='#6D1B3A')
        ax.legend(title='Task', fontsize=8, loc='upper right')
        ax.spines[['top', 'right']].set_visible(False)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4, color='#E8A0B4')
        ax.set_axisbelow(True)
        plt.tight_layout()
        for ext in ('pdf', 'png'):
            out = figures_dir / f'morph_vs_conf_scatter.{ext}'
            fig.savefig(out, bbox_inches='tight', dpi=200, facecolor='white')
            print(f'  Guardado: {out}', file=sys.stderr)
        plt.close(fig)


    print("Generando matrices de confusion CLAM-MB...", file=sys.stderr)
    plot_confusion_matrices_clam(results_dir)

    print("Generando heatmaps de matrices biologicas...", file=sys.stderr)
    plot_biological_matrix(results_dir)

    print("Generando morfología vs confusión...", file=sys.stderr)
    plot_morphology_vs_confusion(results_dir, show_diag=True)
    plot_morphology_vs_confusion(results_dir, show_diag=False)

    print("Generando diagramas de barras...", file=sys.stderr)
    plot_bar_charts(rows, results_dir)

    latex    = generate_latex(rows)
    out_path = results_dir / 'results_table.tex'
    out_path.write_text(latex, encoding='utf-8')
    print(latex)
    print(f'\n% LaTeX guardado en {out_path}', file=sys.stderr)

    morph_sep = load_morphological_separability()
    summary_latex    = generate_summary_latex(rows_mean, centroid_dists, morph_sep)
    summary_out_path = results_dir / 'summary_table.tex'
    summary_out_path.write_text(summary_latex, encoding='utf-8')
    print(summary_latex)
    print(f'\n% Tabla resumen guardada en {summary_out_path}', file=sys.stderr)