"""
plot_learning_curves.py

Genera curvas de entrenamiento (train/val loss y métricas opcionales) a partir
de los ficheros metrics.csv generados por PyTorch Lightning CSVLogger.

Uso básico (un experimento, todos sus splits):
    python pathbench/utils/plot_learning_curves.py \
        --experiment_dir experiments/brca_pam50_virchow2

Comparar varios experimentos:
    python pathbench/utils/plot_learning_curves.py \
        --mil_dirs experiments/brca_pam50_virchow2/mil/00000-... \
                   experiments/brca_pam50_virchow2/mil/00001-...

Opciones:
    --metrics   Métricas a graficar además de loss (ej. val/acc val/auc).
    --output    Ruta de salida del PNG (por defecto: learning_curves.png).
    --title     Título del gráfico.
"""

import argparse
import os
import glob
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


TRAIN_COLOR = '#90CAF9'   # azul claro
VAL_COLOR   = '#1565C0'   # azul oscuro
PALETTE = [
    ('#EF9A9A', '#C62828'),  # rojo
    ('#A5D6A7', '#2E7D32'),  # verde
    ('#FFF59D', '#F57F17'),  # amarillo
    ('#CE93D8', '#6A1B9A'),  # morado
]


def load_metrics(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def extract_per_epoch(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Devuelve DataFrame con columnas [epoch, col] sin NaN, sin duplicados."""
    if col not in df.columns:
        return pd.DataFrame(columns=['epoch', col])
    sub = df[df[col].notna()][['epoch', col]].drop_duplicates('epoch').sort_values('epoch')
    return sub.reset_index(drop=True)


def find_mil_dirs(experiment_dir: str):
    pattern = os.path.join(experiment_dir, 'mil', '*', 'logs', 'version_0', 'metrics.csv')
    csvs = sorted(glob.glob(pattern))
    # también saved_models
    pattern2 = os.path.join(experiment_dir, 'saved_models', '*', 'logs', 'version_0', 'metrics.csv')
    csvs += sorted(glob.glob(pattern2))
    return csvs


def short_label(csv_path: str) -> str:
    """Extrae un label legible del path."""
    parts = csv_path.split(os.sep)
    # buscar la parte que contiene el nombre del run (00000-...)
    for p in reversed(parts):
        if p.startswith('0') and '-' in p:
            return p
    return os.path.dirname(csv_path)


def plot_curves(csv_paths: list, metrics: list, output: str, title: str):
    n_metrics = 1 + len(metrics)  # loss + extra métricas
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5), squeeze=False)
    axes = axes[0]

    for i, csv_path in enumerate(csv_paths):
        df = load_metrics(csv_path)
        label = short_label(csv_path)
        train_c, val_c = (TRAIN_COLOR, VAL_COLOR) if len(csv_paths) == 1 else PALETTE[i % len(PALETTE)]

        # --- Loss ---
        ax = axes[0]
        train_loss = extract_per_epoch(df, 'train/loss')
        val_loss   = extract_per_epoch(df, 'val/loss')
        if not train_loss.empty:
            ax.plot(train_loss['epoch'], train_loss['train/loss'],
                    '--', color=train_c, linewidth=1.5,
                    label=f'train {label}' if len(csv_paths) > 1 else 'train loss')
        if not val_loss.empty:
            ax.plot(val_loss['epoch'], val_loss['val/loss'],
                    '-o', color=val_c, linewidth=2,
                    label=f'val {label}' if len(csv_paths) > 1 else 'val loss')
        ax.set_title('Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

        # --- Métricas adicionales ---
        for j, metric in enumerate(metrics):
            ax = axes[j + 1]
            series = extract_per_epoch(df, metric)
            if not series.empty:
                ax.plot(series['epoch'], series[metric],
                        '-o', color=val_c, linewidth=2,
                        label=label if len(csv_paths) > 1 else metric)
            ax.set_title(metric)
            ax.set_xlabel('Epoch')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)

    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold')

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    plt.savefig(output, dpi=150)
    print(f"Gráfica guardada en: {output}")


def main():
    parser = argparse.ArgumentParser(description="Genera curvas de aprendizaje MIL")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--experiment_dir', type=str,
                       help='Directorio del experimento (busca todos los runs MIL)')
    group.add_argument('--mil_dirs', nargs='+',
                       help='Rutas directas a directorios MIL o a ficheros metrics.csv')
    parser.add_argument('--metrics', nargs='*', default=[],
                        help='Métricas adicionales a graficar (ej. val/acc val/auc val/macro_f1)')
    parser.add_argument('--output', default='learning_curves.png',
                        help='Fichero PNG de salida (por defecto: learning_curves.png)')
    parser.add_argument('--title', default='',
                        help='Título del gráfico')
    args = parser.parse_args()

    if args.experiment_dir:
        csv_paths = find_mil_dirs(args.experiment_dir)
        if not csv_paths:
            print(f"No se encontraron metrics.csv en {args.experiment_dir}/mil/*/logs/version_0/")
            return
        if not args.title:
            args.title = os.path.basename(args.experiment_dir)
    else:
        csv_paths = []
        for p in args.mil_dirs:
            if p.endswith('.csv'):
                csv_paths.append(p)
            else:
                candidate = os.path.join(p, 'logs', 'version_0', 'metrics.csv')
                if os.path.exists(candidate):
                    csv_paths.append(candidate)
                else:
                    print(f"[WARN] No se encontró metrics.csv en {p}")

    if not csv_paths:
        print("Sin datos que graficar.")
        return

    print(f"Graficando {len(csv_paths)} run(s)...")
    plot_curves(csv_paths, args.metrics, args.output, args.title)


if __name__ == '__main__':
    main()