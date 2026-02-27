"""
Genera matrices de confusión (validación y test) para el mejor trial por
arquitectura MIL en una optimización PathBench.

Lee directamente los ficheros val_result.csv (mil/) y test_result.csv (mil_eval/)
y usa sklearn para calcular la CM.

Uso:
    source /shared/home/PARADIS/PathBench-MIL/pathbench_env/bin/activate
    python3 claude/confusion_matrix_best_trials.py [--exp EXP_DIR] [--study STUDY_NAME]

Valores por defecto:
    --exp    experiments/brca_pam50_virchow2_prueba_13_optimizado
    --study  Exp2
"""

import argparse
import os
import sys
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    print("ERROR: optuna no instalado.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Argumentos
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--exp', default='experiments/brca_pam50_virchow2_prueba_13_optimizado',
                    help='Directorio del experimento')
parser.add_argument('--study', default='Exp2',
                    help='Nombre del estudio Optuna')
parser.add_argument('--out', default='claude',
                    help='Directorio de salida para las imágenes')
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)

# ---------------------------------------------------------------------------
# Clases PAM50 (índice → nombre)
# ---------------------------------------------------------------------------
CLASS_NAMES = ['Basal', 'Her2', 'LumA', 'LumB', 'Normal']

# ---------------------------------------------------------------------------
# Cargar estudio Optuna
# ---------------------------------------------------------------------------
db_path = os.path.join(args.exp, 'optimization', 'optuna_study.db')
storage = f'sqlite:///{db_path}'

try:
    study = optuna.load_study(study_name=args.study, storage=storage)
except Exception:
    available = optuna.get_all_study_names(storage=storage)
    if len(available) == 1:
        study = optuna.load_study(study_name=available[0], storage=storage)
    else:
        print(f"ERROR: estudio '{args.study}' no encontrado. Disponibles: {available}")
        sys.exit(1)

completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

# Mejor trial por arquitectura
mil_bests: dict = {}
for t in completed:
    mil = t.params.get('mil', 'unknown')
    if mil not in mil_bests or t.value > mil_bests[mil].value:
        mil_bests[mil] = t

print(f"Arquitecturas encontradas: {list(mil_bests.keys())}")

# ---------------------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------------------
def find_trial_dir(base_dir: str, trial_number: int, suffix: str = '') -> str | None:
    """Busca el directorio con prefijo NNNNN- correspondiente al trial."""
    pattern = os.path.join(base_dir, f'{trial_number:05d}-*{suffix}')
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def load_predictions(csv_path: str):
    """Devuelve (y_true, y_pred) como arrays numpy."""
    df = pd.read_csv(csv_path)
    y_true = df['y_true'].values
    pred_cols = [c for c in df.columns if c.startswith('y_pred')]
    y_pred = df[pred_cols].values.argmax(axis=1)
    return y_true, y_pred


def plot_cm(cm: np.ndarray, class_names: list, title: str, out_path: str, value: float):
    """Dibuja y guarda una matriz de confusión normalizada."""
    n = len(class_names)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, data, fmt, label in [
        (axes[0], cm,      'd',    'Counts'),
        (axes[1], cm_norm, '.2f',  'Normalized'),
    ]:
        im = ax.imshow(data, interpolation='nearest',
                       cmap='Blues', vmin=0, vmax=(1 if fmt == '.2f' else None))
        plt.colorbar(im, ax=ax)
        ax.set(xticks=range(n), yticks=range(n),
               xticklabels=class_names, yticklabels=class_names)
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('True', fontsize=11)
        ax.set_title(label, fontsize=11)
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
        thresh = data.max() / 2.0
        for i in range(n):
            for j in range(n):
                ax.text(j, i, format(data[i, j], fmt),
                        ha='center', va='center', fontsize=9,
                        color='white' if data[i, j] > thresh else 'black')

    fig.suptitle(f'{title}\nmean_f1 = {value:.4f}', fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Guardado: {out_path}")


# ---------------------------------------------------------------------------
# Generar CMs
# ---------------------------------------------------------------------------
mil_dir     = os.path.join(args.exp, 'mil')
mil_eval_dir = os.path.join(args.exp, 'mil_eval')

for mil_name, trial in sorted(mil_bests.items(), key=lambda x: -x[1].value):
    print(f"\n--- {mil_name}  (trial #{trial.number:03d}, value={trial.value:.4f}) ---")

    # Validación
    val_dir = find_trial_dir(mil_dir, trial.number)
    if val_dir:
        val_csv = os.path.join(val_dir, 'val_result.csv')
        if os.path.exists(val_csv):
            y_true, y_pred = load_predictions(val_csv)
            cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
            out = os.path.join(args.out, f'cm_{mil_name}_val_trial{trial.number:03d}.png')
            plot_cm(cm, CLASS_NAMES,
                    f'{mil_name} — Validación (TCGA) — trial {trial.number:03d}',
                    out, trial.value)
        else:
            print(f"  WARN: no encontrado {val_csv}")
    else:
        print(f"  WARN: no encontrado directorio mil/ para trial {trial.number}")

    # Test (CPTAC)
    test_dir = find_trial_dir(mil_eval_dir, trial.number, suffix='_cptac')
    if test_dir:
        test_csv = os.path.join(test_dir, 'test_result.csv')
        if os.path.exists(test_csv):
            y_true, y_pred = load_predictions(test_csv)
            cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
            out = os.path.join(args.out, f'cm_{mil_name}_test_trial{trial.number:03d}.png')
            plot_cm(cm, CLASS_NAMES,
                    f'{mil_name} — Test (CPTAC) — trial {trial.number:03d}',
                    out, trial.value)
        else:
            print(f"  WARN: no encontrado {test_csv}")
    else:
        print(f"  WARN: no encontrado directorio mil_eval/ para trial {trial.number}")

print("\nHecho.")