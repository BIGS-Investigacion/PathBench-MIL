#!/usr/bin/env python3
"""
summary_coral.py — Tabla resumen y figuras de todos los experimentos CORAL.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT    = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / 'results/colon/coral_summary'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Cargar resultados ─────────────────────────────────────────────────────────

geo  = pd.read_csv(ROOT / 'results/colon/coral_geometric/results_pca128_s42.csv')
orth = pd.read_csv(ROOT / 'results/colon/coral_orthogonal/results_pca128_s42.csv')
ps   = pd.read_csv(ROOT / 'results/colon/coral_pseudolabel/results_pca128.csv')
mil  = pd.read_csv(ROOT / 'results/colon/coral_mil/results_pca128.csv')
fs   = pd.read_csv(ROOT / 'results/colon/coral_fewshot/fewshot_pca128.csv')

# Función auxiliar: extraer fila de un dataframe por tag y dataset
def get(df, tag, ds):
    row = df[(df['tag'] == tag) & (df['dataset'] == ds)]
    if row.empty:
        return None
    return row.iloc[0]

# ── Construir tabla resumen ───────────────────────────────────────────────────

rows = []

def add(label, cat, cptac_mss, cptac_msi, cptac_bal, tcga_mss, tcga_msi, tcga_bal,
        cptac_std=None, tcga_std=None):
    rows.append({
        'Método': label, 'Categoría': cat,
        'CPTAC MSS': cptac_mss, 'CPTAC MSI': cptac_msi, 'CPTAC bal': cptac_bal,
        'TCGA MSS':  tcga_mss,  'TCGA MSI':  tcga_msi,  'TCGA bal':  tcga_bal,
        'CPTAC std': cptac_std, 'TCGA std':  tcga_std,
    })

# MIL sin CORAL (modelo DSMIL entrenado en Macarena)
r = get(mil, 'Sin CORAL', 'Macarena')
add('MIL Macarena (train)',    'MIL',
    None, None, None, None, None, None)

r_c = get(mil, 'Sin CORAL', 'CPTAC-COAD')
r_t = get(mil, 'Sin CORAL', 'TCGA-COAD')
add('MIL sin CORAL',           'MIL',
    r_c['acc_mss'], r_c['acc_msi'], r_c['bal_acc'],
    r_t['acc_mss'], r_t['acc_msi'], r_t['bal_acc'])

# LR — variantes sobre slide means
for tag, label, cat in [
    ('A) Sin alineación',                   'LR sin alineación',              'LR'),
    ('B) CORAL estándar',                   'LR + CORAL estándar',            'LR'),
    ('C) CORAL cond. — labels verdaderas',  'LR + CORAL cond. (oracle)',      'LR'),
    ('D) CORAL cond. — pseudo-labels LR',   'LR + CORAL pseudo-labels',       'LR'),
]:
    r_c = get(geo, tag, 'CPTAC-COAD')
    if r_c is None:
        r_c = get(ps, tag, 'CPTAC-COAD')
    r_t = get(geo, tag, 'TCGA-COAD')
    if r_t is None:
        r_t = get(ps, tag, 'TCGA-COAD')
    if r_c is not None and r_t is not None:
        add(label, cat,
            r_c['acc_mss'], r_c['acc_msi'], r_c['bal_acc'],
            r_t['acc_mss'], r_t['acc_msi'], r_t['bal_acc'])

r_c = get(geo,  'E) Corrección geométrica',       'CPTAC-COAD')
r_t = get(geo,  'E) Corrección geométrica',       'TCGA-COAD')
add('LR + Corrección geométrica', 'LR',
    r_c['acc_mss'], r_c['acc_msi'], r_c['bal_acc'],
    r_t['acc_mss'], r_t['acc_msi'], r_t['bal_acc'])

r_c = get(orth, 'G) CORAL ortogonal a la clase',  'CPTAC-COAD')
r_t = get(orth, 'G) CORAL ortogonal a la clase',  'TCGA-COAD')
add('LR + CORAL ortogonal', 'LR',
    r_c['acc_mss'], r_c['acc_msi'], r_c['bal_acc'],
    r_t['acc_mss'], r_t['acc_msi'], r_t['bal_acc'])

# Few-shot
for N in [5, 10, 20, 30]:
    rc = fs[(fs['dataset'] == 'CPTAC-COAD') & (fs['N'] == N)]
    rt = fs[(fs['dataset'] == 'TCGA-COAD')  & (fs['N'] == N)]
    if rc.empty or rt.empty:
        continue
    add(f'LR + CORAL cond. few-shot (N={N}/clase)', 'Few-shot',
        None, None, rc.iloc[0]['mean'],
        None, None, rt.iloc[0]['mean'],
        cptac_std=rc.iloc[0]['std'], tcga_std=rt.iloc[0]['std'])

table = pd.DataFrame(rows)
table.to_csv(OUT_DIR / 'summary_table.csv', index=False)

# Imprimir tabla
print('\n' + '='*95)
print(f'{"Método":<45} {"CPTAC bal":>10} {"TCGA bal":>10}')
print('='*95)
last_cat = None
for _, r in table.iterrows():
    if r['Categoría'] != last_cat:
        if last_cat is not None:
            print('-'*95)
        last_cat = r['Categoría']
    c_bal = f'{r["CPTAC bal"]:.3f}' if pd.notna(r["CPTAC bal"]) else '—'
    t_bal = f'{r["TCGA bal"]:.3f}'  if pd.notna(r["TCGA bal"])  else '—'
    c_std = f' ±{r["CPTAC std"]:.3f}' if pd.notna(r.get("CPTAC std") or np.nan) else ''
    t_std = f' ±{r["TCGA std"]:.3f}'  if pd.notna(r.get("TCGA std") or np.nan) else ''
    print(f'  {r["Método"]:<43} {c_bal+c_std:>14} {t_bal+t_std:>14}')
print('='*95)


# ── Figura 1: comparativa de métodos (bar chart) ──────────────────────────────

# Seleccionar filas con bal-acc para comparar
compare = table[table['CPTAC bal'].notna() & ~table['Método'].str.contains('train')].copy()

CAT_COLORS = {'MIL': '#78909C', 'LR': '#1565C0', 'Few-shot': '#2E7D32'}
HATCH = {'MIL': '', 'LR': '', 'Few-shot': '//'}

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
x = np.arange(len(compare))
width = 0.6

for ax, ds_col, ds_std, title in [
        (axes[0], 'CPTAC bal', 'CPTAC std', 'CPTAC-COAD'),
        (axes[1], 'TCGA bal',  'TCGA std',  'TCGA-COAD')]:

    vals  = compare[ds_col].values.astype(float)
    stds  = compare[ds_std].values
    stds  = np.where(pd.isna(stds), 0, stds.astype(float))
    colors = [CAT_COLORS[c] for c in compare['Categoría']]
    hatches = [HATCH[c] for c in compare['Categoría']]

    bars = ax.bar(x, vals, width, yerr=stds, capsize=4,
                  color=colors, edgecolor='white', linewidth=0.5,
                  error_kw={'elinewidth': 1.5, 'ecolor': '#333'})
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)

    ax.set_xticks(x)
    ax.set_xticklabels(compare['Método'], rotation=40, ha='right', fontsize=8)
    ax.set_ylabel('Balanced accuracy', fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.set_ylim(0.4, 1.08)
    ax.axhline(1.0, ls=':', color='#999', lw=1)
    ax.grid(axis='y', alpha=0.3)

    for i, (v, s) in enumerate(zip(vals, stds)):
        ax.text(i, v + s + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=7)

legend_patches = [mpatches.Patch(color=CAT_COLORS[c], label=c) for c in CAT_COLORS]
fig.legend(handles=legend_patches, loc='upper center', ncol=3,
           fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, 1.01))

fig.suptitle('Comparativa de métodos CORAL — balanced accuracy cross-domain',
             fontsize=13, y=1.05)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(OUT_DIR / f'comparison.{ext}', dpi=150, bbox_inches='tight')
plt.close(fig)
print('\nGuardado: comparison')


# ── Figura 2: curva few-shot ──────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, ds_name in zip(axes, ['CPTAC-COAD', 'TCGA-COAD']):
    sub = fs[fs['dataset'] == ds_name].sort_values('N')
    if sub.empty:
        continue

    ax.plot(sub['N'], sub['mean'], 'o-', color='#1565C0', lw=2.5,
            markersize=7, label='CORAL cond. few-shot')
    ax.fill_between(sub['N'],
                    sub['mean'] - sub['std'],
                    sub['mean'] + sub['std'],
                    alpha=0.2, color='#1565C0', label='±1 std (20 semillas)')
    ax.axhline(sub['baseline'].iloc[0], ls='--', color='#757575', lw=1.5,
               label=f'Sin alineación  ({sub["baseline"].iloc[0]:.3f})')
    ax.axhline(sub['oracle'].iloc[0], ls='--', color='#E53935', lw=1.5,
               label=f'Oracle — labels completas  ({sub["oracle"].iloc[0]:.3f})')

    ax.set_xlabel('N slides etiquetados por clase', fontsize=12)
    ax.set_ylabel('Balanced accuracy', fontsize=12)
    ax.set_title(ds_name, fontsize=13)
    ax.set_xticks(sub['N'].tolist())
    ax.legend(fontsize=9, framealpha=0.9)
    ax.set_ylim(0.55, 1.05)
    ax.grid(True, alpha=0.3)

    for _, r in sub.iterrows():
        ax.annotate(f'{r["mean"]:.3f}', (r['N'], r['mean']),
                    textcoords='offset points', xytext=(0, 8),
                    ha='center', fontsize=8)

fig.suptitle('CORAL condicional few-shot — curva de aprendizaje (20 semillas)',
             fontsize=13)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(OUT_DIR / f'fewshot_curve.{ext}', dpi=150, bbox_inches='tight')
plt.close(fig)
print('Guardado: fewshot_curve')

print(f'\nListo. Resultados en: {OUT_DIR}')