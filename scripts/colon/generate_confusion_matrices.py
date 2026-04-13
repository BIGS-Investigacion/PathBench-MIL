"""
Genera matrices de confusión (texto + LaTeX) para los resultados de predict_slides
del modelo MSS/MSI, mergeando predicciones con anotaciones ground truth.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, roc_auc_score

ROOT    = Path(__file__).resolve().parents[2]
RES_DIR = ROOT / 'results/colon'
ANN_DIR = ROOT / 'config/annotations'

EXPERIMENTS = [
    {
        'label': 'None — Macarena',
        'pred':  RES_DIR / 'predictions_macarena.csv',
        'ann':   ANN_DIR / 'annotations_colon_mss_msi.csv',
    },
    {
        'label': 'None — CPTAC-COAD',
        'pred':  RES_DIR / 'predictions_cptac.csv',
        'ann':   ANN_DIR / 'annotations_cptac_coad_mss_msi.csv',
    },
    {
        'label': 'None — TCGA-COAD',
        'pred':  RES_DIR / 'predictions_tcga.csv',
        'ann':   ANN_DIR / 'annotations_tcga_coad_mss_msi.csv',
    },
    {
        'label': 'Macenko — Macarena',
        'pred':  RES_DIR / 'predictions_macarena_macenko.csv',
        'ann':   ANN_DIR / 'annotations_colon_mss_msi.csv',
    },
    {
        'label': 'Macenko — CPTAC-COAD',
        'pred':  RES_DIR / 'predictions_cptac_macenko.csv',
        'ann':   ANN_DIR / 'annotations_cptac_coad_mss_msi.csv',
    },
    {
        'label': 'Macenko — TCGA-COAD',
        'pred':  RES_DIR / 'predictions_tcga_macenko.csv',
        'ann':   ANN_DIR / 'annotations_tcga_coad_mss_msi.csv',
    },
]

CLASS_NAMES = {0: 'MSS', 1: 'MSI'}


def cm_to_latex(cm, label):
    tn, fp, fn, tp = cm.ravel()
    return (
        f"\\begin{{table}}[h]\n"
        f"\\centering\n"
        f"\\caption{{{label}}}\n"
        f"\\begin{{tabular}}{{lcc}}\n"
        f"\\toprule\n"
        f" & \\textbf{{Pred MSS}} & \\textbf{{Pred MSI}} \\\\\n"
        f"\\midrule\n"
        f"\\textbf{{True MSS}} & {tn} & {fp} \\\\\n"
        f"\\textbf{{True MSI}} & {fn} & {tp} \\\\\n"
        f"\\bottomrule\n"
        f"\\end{{tabular}}\n"
        f"\\end{{table}}\n"
    )


latex_blocks = []
summary_rows = []

for exp in EXPERIMENTS:
    if not exp['pred'].exists():
        print(f"[SKIP] No existe: {exp['pred'].name}")
        continue

    pred = pd.read_csv(exp['pred'])
    ann  = pd.read_csv(exp['ann'])[['slide', 'category']]

    df = pred.merge(ann, on='slide', how='inner')
    if df.empty:
        print(f"[WARN] Sin matches para {exp['label']}")
        continue

    y_true = df['category'].values
    y_pred = df['y_pred_class'].values
    y_prob = df['y_pred1'].values  # probabilidad de clase MSI

    cm  = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    bac = balanced_accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float('nan')

    print(f"\n{exp['label']}  (n={len(df)})")
    print(f"  ACC={acc:.3f}  BAcc={bac:.3f}  AUC={auc:.3f}")
    print(f"  {cm}")

    latex_blocks.append(cm_to_latex(cm, exp['label']))
    summary_rows.append({
        'experiment': exp['label'],
        'n': len(df),
        'ACC': round(acc, 4),
        'BAcc': round(bac, 4),
        'AUC': round(auc, 4),
    })

# Guardar LaTeX
latex_path = RES_DIR / 'confusion_matrices.tex'
with open(latex_path, 'w') as f:
    f.write("% Matrices de confusión MSS/MSI\n\n")
    f.write("\n".join(latex_blocks))
print(f"\nGuardado: {latex_path}")

# Guardar resumen CSV
summary_df = pd.DataFrame(summary_rows)
summary_path = RES_DIR / 'summary_metrics.csv'
summary_df.to_csv(summary_path, index=False)
print(f"Guardado: {summary_path}")
print(summary_df.to_string(index=False))