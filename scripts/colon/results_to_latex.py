#!/usr/bin/env python3
"""
results_to_latex.py

Genera tablas LaTeX desde el CSV de eval_final_warmstart.py.
Produce una tabla por tarea (MSS/MSI, MLH1, MSH2, MSH6, PMS2).

Uso:
  python scripts/colon/results_to_latex.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[2]
CSV     = ROOT / 'results/colon/eval_final_warmstart/results_pca128_cv10.csv'
OUT_DIR = ROOT / 'results/colon/eval_final_warmstart'

EVAL_ORDER = [
    'Macarena 10-fold CV',
    'Sin calibrar',
    'Reajuste N=5/clase',
    'Reajuste N=10/clase',
    'Reajuste N=20/clase',
]

EVAL_LABELS = {
    'Macarena 10-fold CV': r'10-fold CV (Macarena)',
    'Sin calibrar':        r'Sin calibrar',
    'Reajuste N=5/clase':  r'Reajuste $N=5$',
    'Reajuste N=10/clase': r'Reajuste $N=10$',
    'Reajuste N=20/clase': r'Reajuste $N=20$',
}

DATASET_ORDER = ['macarena', 'CPTAC-COAD', 'TCGA-COAD']

TASK_LABELS = {
    'MSS/MSI': 'MSS/MSI',
    'MLH1': 'MLH1',
    'MSH2': 'MSH2',
    'MSH6': 'MSH6',
    'PMS2': 'PMS2',
}

TASKS = {
    'MSS/MSI': ROOT / 'config/annotations/annotations_colon_mss_msi_all.csv',
    'MLH1':    ROOT / 'config/annotations/annotations_colon_mlh1_all.csv',
    'MSH2':    ROOT / 'config/annotations/annotations_colon_msh2_all.csv',
    'MSH6':    ROOT / 'config/annotations/annotations_colon_msh6_all.csv',
    'PMS2':    ROOT / 'config/annotations/annotations_colon_pms2_all.csv',
}

BAGS_DIR = Path('/home/PARADIS/colon/tfrecords/shared_bags/256_128_none_h_optimus_0')

def get_counts(ann_path):
    """Devuelve {dataset: (n_wt, n_mut)} para los slides con bag disponible."""
    import torch
    ann = pd.read_csv(ann_path)
    counts = {}
    for ds, grp in ann.groupby('dataset'):
        available = grp[grp['slide'].apply(
            lambda s: (BAGS_DIR / f'{s}.pt').exists())]
        n0 = (available['category'] == 0).sum()
        n1 = (available['category'] == 1).sum()
        counts[ds] = (n0, n1)
    return counts


def fmt(val, std=None):
    if pd.isna(val):
        return '---'
    if std is None or pd.isna(std):
        return f'{val:.3f}'
    return f'${val:.3f} \\pm {std:.3f}$'


def bold(s):
    return r'\textbf{' + s + '}'


def make_task_table(df_task, task_name, counts):
    lines = []
    lines.append(r'\begin{table}[ht]')
    lines.append(r'\centering')
    lines.append(r'\small')
    lines.append(r'\begin{tabular}{llccc}')
    lines.append(r'\toprule')
    lines.append(r'\textbf{Conjunto} & \textbf{Método} & \textbf{Acc(wt)} & \textbf{Acc(mut)} & \textbf{Bal. Acc} \\')
    lines.append(r'\midrule')

    for ds in DATASET_ORDER:
        df_ds = df_task[df_task['dataset'].str.lower().str.replace('-', '_') == ds.lower().replace('-', '_')]
        if ds == 'macarena':
            df_ds = df_task[df_task['dataset'] == 'macarena']
        else:
            df_ds = df_task[df_task['dataset'] == ds]

        if df_ds.empty:
            continue

        ds_key     = {'macarena': 'macarena', 'CPTAC-COAD': 'cptac_coad', 'TCGA-COAD': 'tcga_coad'}[ds]
        base_label = {'macarena': 'Macarena', 'CPTAC-COAD': 'CPTAC-COAD', 'TCGA-COAD': 'TCGA-COAD'}[ds]
        if ds_key in counts:
            n0, n1 = counts[ds_key]
            ds_label = f'{base_label} ({n0}/{n1})'
        else:
            ds_label = base_label

        # Find best bal for bolding (excluding CV row when in cross-domain)
        if ds == 'macarena':
            best_bal = df_ds['bal'].max()
        else:
            best_bal = df_ds['bal'].max()

        first_row = True
        n_rows = len([e for e in EVAL_ORDER
                      if not df_ds[df_ds['eval'] == e].empty])

        for eval_name in EVAL_ORDER:
            row = df_ds[df_ds['eval'] == eval_name]
            if row.empty:
                continue
            row = row.iloc[0]

            acc0_str = fmt(row['acc0'], row.get('acc0_std'))
            acc1_str = fmt(row['acc1'], row.get('acc1_std'))
            bal_str  = fmt(row['bal'],  row.get('bal_std'))

            # Bold best bal per dataset
            if abs(row['bal'] - best_bal) < 1e-9:
                bal_str = bold(bal_str)

            method_label = EVAL_LABELS[eval_name]

            if first_row:
                ds_col = r'\multirow{' + str(n_rows) + r'}{*}{' + ds_label + r'}'
                first_row = False
            else:
                ds_col = ''

            lines.append(f'{ds_col} & {method_label} & {acc0_str} & {acc1_str} & {bal_str} \\\\')

        lines.append(r'\midrule')

    # Remove last \midrule and add \bottomrule
    lines[-1] = r'\bottomrule'
    lines.append(r'\end{tabular}')
    lines.append(r'\caption{Resultados para ' + task_name + r'. '
                 r'Representación: media de los patches (slide mean) + PCA(128) + LR. '
                 r'El reajuste se realiza mediante warm-start sobre $N$ slides etiquetados del dominio target. '
                 r'Media $\pm$ desviación estándar sobre 10 semillas.}')
    lines.append(r'\label{tab:results_' + task_name.lower().replace('/', '_') + r'}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)


def main():
    df = pd.read_csv(CSV)

    all_tables = []
    for task, ann_path in TASKS.items():
        df_task = df[df['task'] == task].copy()
        counts  = get_counts(ann_path)
        table   = make_task_table(df_task, TASK_LABELS[task], counts)
        all_tables.append(table)
        print(table)
        print()

    out_path = OUT_DIR / 'tables.tex'
    with open(out_path, 'w') as f:
        f.write('\n\n'.join(all_tables) + '\n')
    print(f'% Tablas guardadas en: {out_path}')


if __name__ == '__main__':
    main()