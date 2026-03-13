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
TASK_ORDER = ['er', 'erbb2', 'pr', 'pam50']


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
        r'PR-AUC is reported for binary tasks (ER, ERBB2, PR) and F1 for PAM50. '
        r'$\delta_n$: difference between Macenko-normalised and unnormalised '
        r'performance on the external test set (CPTAC). '
        r'RPD (Relative Performance Drop): normalised performance gap between '
        r'CPTAC and TCGA without stain normalisation, '
        r'$\mathrm{RPD}=(\mathrm{none}_\mathrm{CPTAC}-\mathrm{none}_\mathrm{TCGA})'
        r'\,/\,(-\,\mathrm{none}_\mathrm{TCGA})$.}'
    )
    buf.append(r'\label{tab:results_summary}')
    buf.append(r'\begin{tabular}{lllrr}')
    buf.append(r'\toprule')
    buf.append(r'\textbf{MIL} & \textbf{Task} & \textbf{Class} '
               r'& $\delta_n$ & \textbf{RPD} \\')
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
                delta_str = _fmt_delta(row['delta_n'])
                rpd_str   = _fmt_rpd(row['rpd'])

                mil_col  = (f'\\multirow{{{n_mil_rows}}}{{*}}{{{mil_disp}}}'
                            if first_mil else '')
                task_col = (f'\\multirow{{{n_task}}}{{*}}{{{task}}}'
                            if first_task else '')

                buf.append(f' {mil_col} & {task_col} & {cls_label}'
                           f' & {delta_str} & {rpd_str} \\\\')

                first_mil  = False
                first_task = False

            if task_idx < len(tasks_present) - 1:
                buf.append(r'\cmidrule{2-5}')

        if mil_idx < len(MIL_ORDER) - 1:
            buf.append(r'\midrule')

    buf.append(r'\bottomrule')
    buf.append(r'\end{tabular}')
    buf.append(r'\end{table}')

    return '\n'.join(buf)


# ── Análisis univariante RPD ~ δn ────────────────────────────────────────────

def univariate_analysis(rows: list) -> None:
    """
    Regresión lineal simple OLS:  RPD = β0 + β1·δn + ε

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
    print('ANÁLISIS UNIVARIANTE:  RPD ~ δn')
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
        print(f'    Pendiente {direction}: mayor δn → '
              + ('menor' if slope < 0 else 'mayor') + ' RPD.')
        print(f'    δn explica el {r2 * 100:.1f}% de la varianza en RPD (R²).')
        print(f'    Tamaño del efecto: r={r_pearson:+.3f} ({_label_r(r_pearson)}), '
              f'f²={f2:.3f} ({_label_f2(f2)}).')
    else:
        print(f'    Relación NO significativa (p={p_value:.3e}, α=0.05).')
        print(f'    δn no predice linealmente el RPD en esta muestra.')
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

    for task in tasks:
        task_rows = [r for r in rows if r['task'] == task]
        if not task_rows:
            continue

        classes   = sorted(set(r['cls_id'] for r in task_rows))
        cls_names = {r['cls_id']: r['cls_name'] for r in task_rows}
        x_labels  = [f"{cls_names[c]}\n({c})" for c in classes]
        n_cls     = len(classes)
        n_all     = len(ALL_MODELS)
        width     = 0.18
        x         = np.arange(n_cls)

        fig, axes = plt.subplots(1, 2, figsize=(max(9, n_cls * 2.4), 5),
                                 sharey=False)
        fig.suptitle(f'{task}  —  {METRIC_LABEL[task]} por clase y MIL  (none)',
                     fontsize=13, fontweight='bold', color='#6D1B3A')

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


# ── Carga de distancias de centroides ─────────────────────────────────────────

PATCH_INTER_DIR = Path(__file__).resolve().parent.parent / 'results' / 'patch_intersection'


def load_centroid_distances(top_k: int = 8) -> dict:
    """Devuelve {(task_upper, cls_id): centroid_dist} desde los CSV generados."""
    dists = {}
    for task in TASK_ORDER:
        csv_path = PATCH_INTER_DIR / task / f'centroid_distances_top{top_k}.csv'
        if not csv_path.exists():
            print(f'[WARN] No encontrado: {csv_path}', file=sys.stderr)
            continue
        import csv
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                dists[(task.upper(), int(row['class']))] = float(row['centroid_dist'])
    return dists


# ── Tabla resumen: medias de los 3 modelos Opt- + distancia de centroide ──────

def generate_summary_latex(rows_mean: list, centroid_dists: dict) -> str:
    CLASS_LABEL_MAP = {
        'er':    {0: 'neg', 1: 'pos'},
        'erbb2': {0: 'neg', 1: 'pos'},
        'pr':    {0: 'neg', 1: 'pos'},
        'pam50': {0: 'Basal', 1: 'Her2', 2: 'LumA', 3: 'LumB', 4: 'Normal'},
    }
    METRIC_LABEL = {'ER': 'PR-AUC', 'ERBB2': 'PR-AUC', 'PR': 'PR-AUC', 'PAM50': 'F1'}

    buf = []
    buf.append(r'\begin{table}[ht]')
    buf.append(r'\centering')
    buf.append(
        r'\caption{Summary of results (mean across three Optuna-optimised MIL models: '
        r'Opt-CLAM, Opt-DSMIL, Opt-TransMIL). '
        r'TCGA: MCCV score (train/val); CPTAC: hold-out score (external test). '
        r'$\delta_n$: Macenko minus none on CPTAC. '
        r'RPD: relative performance drop CPTAC vs TCGA. '
        r'$d$: Euclidean distance between TCGA and CPTAC class centroids '
        r'in the embedding space (top-8 attention patches).}'
    )
    buf.append(r'\label{tab:summary_centroid}')
    buf.append(r'\begin{tabular}{llcrrrrrr}')
    buf.append(r'\toprule')
    buf.append(r'\textbf{Task} & \textbf{Class} & \textbf{Metric} '
               r'& \textbf{TCGA} & \textbf{CPTAC} '
               r'& $\delta_n$ & $d$ & $\Delta p$ & \textbf{RPD} \\')
    buf.append(r'\midrule')

    for task in [t.upper() for t in TASK_ORDER]:
        task_rows = sorted([r for r in rows_mean if r['task'] == task],
                           key=lambda r: r['cls_id'])
        if not task_rows:
            continue
        metric = METRIC_LABEL[task]
        n = len(task_rows)

        for i, r in enumerate(task_rows):
            task_col = f'\\multirow{{{n}}}{{*}}{{{task}}}' if i == 0 else ''
            cls_name = r['cls_name']
            tcga_val = f"{r['none_tcga_mean']:.3f}"
            cptac_val = f"{r['none_cptac_mean']:.3f}"
            dn_val   = f"{r['delta_n']:+.3f}"
            rpd_str  = ('---' if np.isnan(r['rpd'])
                        else f"{r['rpd']:+.3f}")
            dist = centroid_dists.get((task, r['cls_id']))
            dist_str = f"{dist:.2f}" if dist is not None else '---'

            met_col = f'\\multirow{{{n}}}{{*}}{{{metric}}}' if i == 0 else ''

            dp = PREVALENCE_SHIFT.get((task, r['cls_id']))
            dp_str = f'${dp:+.3f}$' if dp is not None else '---'

            buf.append(f'  {task_col} & {cls_name} & {met_col} '
                       f'& {tcga_val} & {cptac_val} & ${dn_val}$ '
                       f'& {dist_str} & {dp_str} & ${rpd_str}$ \\\\')

        buf.append(r'\midrule' if task != 'PAM50' else r'\bottomrule')

    buf.append(r'\end{tabular}')
    buf.append(r'\end{table}')
    return '\n'.join(buf)


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


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS_DIR

    rows = compute_rows(results_dir)
    if not rows:
        print('[ERROR] No se extrajeron datos. Revisa los ficheros de resultados.',
              file=sys.stderr)
        sys.exit(1)

    # Agregar por (task, cls_id): media de δn, RPD y métricas sobre los 3 MIL Opt-
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

    print("Generando diagramas de barras...", file=sys.stderr)
    plot_bar_charts(rows, results_dir)

    latex    = generate_latex(rows)
    out_path = results_dir / 'results_table.tex'
    out_path.write_text(latex, encoding='utf-8')
    print(latex)
    print(f'\n% LaTeX guardado en {out_path}', file=sys.stderr)

    summary_latex    = generate_summary_latex(rows_mean, centroid_dists)
    summary_out_path = results_dir / 'summary_table.tex'
    summary_out_path.write_text(summary_latex, encoding='utf-8')
    print(summary_latex)
    print(f'\n% Tabla resumen guardada en {summary_out_path}', file=sys.stderr)