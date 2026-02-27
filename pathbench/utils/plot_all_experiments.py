"""
plot_all_experiments.py

Genera curvas de aprendizaje para todos los experimentos encontrados en el
directorio `experiments/`. Por cada experimento crea un PNG con todos sus
runs MIL superpuestos.

Uso:
    python pathbench/utils/plot_all_experiments.py
    python pathbench/utils/plot_all_experiments.py --experiments_dir experiments
    python pathbench/utils/plot_all_experiments.py --metrics val/acc val/macro_f1
    python pathbench/utils/plot_all_experiments.py --output_dir /tmp/curves
"""

import argparse
import os
import glob
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PALETTE = [
    ('#EF9A9A', '#C62828'),
    ('#90CAF9', '#1565C0'),
    ('#A5D6A7', '#2E7D32'),
    ('#FFF59D', '#F9A825'),
    ('#CE93D8', '#6A1B9A'),
    ('#FFCC80', '#E65100'),
    ('#80DEEA', '#00838F'),
    ('#F48FB1', '#880E4F'),
]


def load_metrics(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def extract_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame(columns=['epoch', col])
    return (df[df[col].notna()][['epoch', col]]
            .drop_duplicates('epoch')
            .sort_values('epoch')
            .reset_index(drop=True))


def find_all_experiments(experiments_dir: str):
    """Devuelve dict {experiment_name: [metrics_csv_path, ...]}."""
    result = {}
    skip = {'shared_bags'}  # directorios que no son experimentos MIL

    for entry in sorted(os.scandir(experiments_dir), key=lambda e: e.name):
        if not entry.is_dir() or entry.name in skip:
            continue
        csvs = sorted(glob.glob(
            os.path.join(entry.path, 'mil', '*', 'logs', 'version_0', 'metrics.csv')
        ))
        if csvs:
            result[entry.name] = csvs
    return result


def run_label(csv_path: str) -> str:
    """Extrae label del run desde el path (ej. 00000-256_128_macenko_virchow2_clam_mil_mb_...)."""
    parts = csv_path.split(os.sep)
    for p in reversed(parts):
        if p[:5].isdigit() and '-' in p:
            return p
    return os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(csv_path))))


def plot_experiment(exp_name: str, csv_paths: list, metrics: list, output_path: str):
    cols = ['val/loss'] + metrics
    n_plots = 1 + len(metrics)  # loss siempre + métricas extra

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5), squeeze=False)
    axes = axes[0]

    any_data = False
    for i, csv_path in enumerate(csv_paths):
        df = load_metrics(csv_path)
        train_c, val_c = PALETTE[i % len(PALETTE)]
        label = run_label(csv_path)

        # --- train/val loss ---
        ax = axes[0]
        train_df = extract_col(df, 'train/loss')
        val_df   = extract_col(df, 'val/loss')
        lbl = label if len(csv_paths) > 1 else ''
        if not train_df.empty:
            ax.plot(train_df['epoch'], train_df['train/loss'],
                    '--', color=train_c, linewidth=1.5,
                    label=f'train {lbl}' if lbl else 'train loss')
            any_data = True
        if not val_df.empty:
            ax.plot(val_df['epoch'], val_df['val/loss'],
                    '-o', color=val_c, linewidth=2, markersize=4,
                    label=f'val {lbl}' if lbl else 'val loss')
            any_data = True

        # --- métricas adicionales ---
        for j, metric in enumerate(metrics):
            ax = axes[j + 1]
            mdf = extract_col(df, metric)
            if not mdf.empty:
                ax.plot(mdf['epoch'], mdf[metric],
                        '-o', color=val_c, linewidth=2, markersize=4,
                        label=lbl if lbl else metric)
                any_data = True

    if not any_data:
        plt.close(fig)
        return

    # Formatear ejes
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7)

    for j, metric in enumerate(metrics):
        axes[j + 1].set_title(metric)
        axes[j + 1].set_xlabel('Epoch')
        axes[j + 1].grid(True, alpha=0.3)
        axes[j + 1].legend(fontsize=7)

    fig.suptitle(exp_name, fontsize=13, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  -> {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Genera learning curves para todos los experimentos MIL'
    )
    parser.add_argument('--experiments_dir', default='experiments',
                        help='Directorio raíz de experimentos (por defecto: experiments)')
    parser.add_argument('--metrics', nargs='*', default=['val/macro_f1'],
                        help='Métricas adicionales a graficar (por defecto: val/macro_f1)')
    parser.add_argument('--output_dir', default=None,
                        help='Directorio de salida. Si no se indica, guarda dentro de cada experimento.')
    args = parser.parse_args()

    experiments = find_all_experiments(args.experiments_dir)

    if not experiments:
        print(f"No se encontraron experimentos MIL en '{args.experiments_dir}'")
        return

    print(f"Encontrados {len(experiments)} experimento(s):")
    for exp_name, csvs in experiments.items():
        print(f"\n[{exp_name}] — {len(csvs)} run(s)")
        if args.output_dir:
            output_path = os.path.join(args.output_dir, f"{exp_name}_learning_curves.png")
        else:
            output_path = os.path.join(args.experiments_dir, exp_name,
                                       'visualizations', 'learning_curves.png')
        plot_experiment(exp_name, csvs, args.metrics, output_path)

    print("\nListo.")


if __name__ == '__main__':
    main()