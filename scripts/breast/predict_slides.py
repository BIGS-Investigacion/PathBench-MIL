#!/usr/bin/env python
"""
Genera un CSV con las predicciones MIL para cada bag (.pt) en un directorio.
Si se proporciona --annotations, genera además una matriz de confusión por dataset.

Carga el modelo directamente desde el checkpoint de Lightning sin pasar por
slideflow.mil ni pathbench.__init__ (evita el circular import).

Uso:
    python scripts/predict_slides.py \
        --model_dir experiments/brca_pam50_virchow2_ho_tcga_tcga/mil/00000-256_128_macenko_virchow2_clam_mil_mb_CrossEntropyLoss_ReLU_AdamW_1 \
        --config config/conf_brca_pam50_virchow2_test.yaml \
        --bags_dir /home/PARADIS/mama/tfrecords/shared_bags/256_128_macenko_virchow2 \
        --annotations config/annotations/annotations_brca_pam50.csv \
        --output predictions_per_slide.csv
"""

import argparse
import os
import sys
import logging
import json
import glob
import importlib.util

import yaml
import numpy as np
import pandas as pd
import torch

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Modelos que slideflow implementa de forma nativa (no en aggregators.py)
_SLIDEFLOW_MIL_FILES = {
    'transmil': os.path.join(ROOT, 'slideflow_fork', 'slideflow', 'mil', 'models', 'transmil.py'),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Predicciones MIL desde bags (.pt) → CSV")
    parser.add_argument('--model_dir', required=True,
                        help="Directorio del modelo entrenado (contiene mil_params.json)")
    parser.add_argument('--config', required=True,
                        help="Fichero de configuración PathBench (.yaml)")
    parser.add_argument('--bags_dir', required=True,
                        help="Directorio con los ficheros .pt de features")
    parser.add_argument('--annotations', default=None,
                        help="CSV de anotaciones (columnas: slide, dataset, category). "
                             "Si se proporciona, genera matrices de confusión por dataset.")
    parser.add_argument('--output', default='predictions_per_slide.csv',
                        help="Ruta del CSV de salida (default: predictions_per_slide.csv)")
    return parser.parse_args()



def find_checkpoint(model_dir: str) -> str:
    """Busca el fichero .ckpt del mejor epoch en el directorio del modelo."""
    pattern = os.path.join(model_dir, 'checkpoints', '**', 'best-epoch*.ckpt')
    ckpts = glob.glob(pattern, recursive=True)
    if not ckpts:
        raise FileNotFoundError(f"No se encontró ningún checkpoint en {model_dir}")
    return sorted(ckpts)[-1]


def _load_module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _get_model_cls(model_name: str):
    """Resuelve la clase del modelo sin triggear imports circulares."""
    key = model_name.lower()
    if key in _SLIDEFLOW_MIL_FILES:
        mod = _load_module_from_file(f'sf_mil_{key}', _SLIDEFLOW_MIL_FILES[key])
        # El nombre de clase en slideflow es CamelCase (ej. transmil → TransMIL)
        class_name = {'transmil': 'TransMIL'}.get(key, model_name)
        return getattr(mod, class_name)
    agg = _load_module_from_file('aggregators',
                                 os.path.join(ROOT, 'pathbench', 'models', 'aggregators.py'))
    return getattr(agg, model_name)


def load_model(model_dir: str):
    """Construye el modelo desde mil_params.json y carga los pesos del checkpoint."""
    with open(os.path.join(model_dir, 'mil_params.json')) as f:
        mil_params = json.load(f)

    params = mil_params['params']
    model_cls = _get_model_cls(params['model'])
    model = model_cls(
        n_feats=mil_params['input_shape'],
        n_out=mil_params['output_shape'],
        z_dim=params.get('z_dim', 256),
        dropout_p=params.get('dropout_p', 0.1),
        activation_function=params.get('activation_function', 'ReLU'),
        encoder_layers=params.get('encoder_layers', 1),
    )

    ckpt_path = find_checkpoint(model_dir)
    logging.info(f"Cargando checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')

    # Lightning guarda pesos con prefijo "model." → strippearlo
    state_dict = {
        k.removeprefix('model.'): v
        for k, v in ckpt['state_dict'].items()
        if k.startswith('model.')
    }
    model.load_state_dict(state_dict, strict=True)
    return model, mil_params


def collect_bags(bags_dir: str):
    bags = sorted([
        os.path.join(bags_dir, f)
        for f in os.listdir(bags_dir)
        if f.endswith('.pt')
    ])
    logging.info(f"Encontrados {len(bags)} bags en {bags_dir}")
    return bags


def predict_bag(model, bag_path: str, device: torch.device,
                apply_softmax: bool, outcome_labels: dict) -> dict:
    slide_name = os.path.splitext(os.path.basename(bag_path))[0]

    features = torch.load(bag_path, map_location=device).float()
    if features.dim() == 1:
        features = features.unsqueeze(0)  # (1, n_features)

    with torch.no_grad():
        logits = model(features.unsqueeze(0))  # (1, n_tiles, n_feats) → (1, n_out)

    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    logits = logits.squeeze()  # (n_out,)

    probs = torch.softmax(logits, dim=-1).cpu().numpy() if apply_softmax else logits.cpu().numpy()
    predicted_idx = int(np.argmax(probs))

    row = {'slide': slide_name}
    for i, p in enumerate(probs):
        row[f'y_pred{i}'] = float(p)
    row['y_pred_class'] = predicted_idx
    row['y_pred_label'] = outcome_labels.get(str(predicted_idx), str(predicted_idx))
    return row


def main():
    args = parse_args()

    for path, label in [(args.model_dir, 'model_dir'), (args.bags_dir, 'bags_dir')]:
        if not os.path.isdir(path):
            logging.error(f"{label} no encontrado: {path}")
            sys.exit(1)

    with open(args.config) as f:
        config = yaml.safe_load(f)
    token = config.get('hf_key')
    if token and str(token) != 'None':
        from huggingface_hub import login
        login(token=token)
    weights_dir = config['weights_dir']
    for var in ('TORCH_HOME', 'HF_HOME', 'XDG_CACHE_HOME', 'HF_DATASETS_CACHE', 'WEIGHTS_DIR'):
        os.environ[var] = weights_dir

    model, mil_params = load_model(args.model_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    logging.info(f"Modelo en {device}")

    outcome_labels = mil_params.get('outcome_labels', {})
    apply_softmax = mil_params['params'].get('apply_softmax', True)
    logging.info(f"Clases: {outcome_labels}  |  apply_softmax: {apply_softmax}")

    bags = collect_bags(args.bags_dir)
    if not bags:
        logging.error("No se encontraron ficheros .pt.")
        sys.exit(1)

    rows, errors = [], []
    for bag_path in bags:
        try:
            row = predict_bag(model, bag_path, device, apply_softmax, outcome_labels)
            rows.append(row)
            logging.info(f"  {row['slide']} → clase {row['y_pred_label']}")
        except Exception as e:
            name = os.path.splitext(os.path.basename(bag_path))[0]
            logging.warning(f"  Error en {name}: {e}")
            errors.append(name)

    if not rows:
        logging.error("No se generaron predicciones.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    logging.info(f"\nCSV guardado: {args.output}  ({len(df)} slides)")
    if errors:
        logging.warning(f"{len(errors)} slides con error: {errors}")

    print("\n=== Distribución de predicciones ===")
    print(df['y_pred_label'].value_counts().sort_index().to_string())
    print("\nPrimeras filas:")
    print(df.head(10).to_string(index=False))

    if args.annotations:
        print_confusion_matrices(df, args.annotations, outcome_labels)


def print_confusion_matrices(df: pd.DataFrame, annotations_path: str, outcome_labels: dict):
    from sklearn.metrics import confusion_matrix, classification_report, average_precision_score

    annot = pd.read_csv(annotations_path)
    merged = df.merge(annot[['slide', 'dataset', 'category']], on='slide', how='inner')

    n_classes = len(outcome_labels)
    labels = list(range(n_classes))
    class_names = [f"{outcome_labels.get(str(i), str(i))}({i})" for i in labels]
    prob_cols = [f'y_pred{i}' for i in labels]

    datasets = sorted(merged['dataset'].unique())
    for ds in datasets:
        sub = merged[merged['dataset'] == ds]
        mask = sub['category'].isin(labels)
        if not mask.all():
            n_dropped = (~mask).sum()
            logging.warning(f"  [{ds}] {n_dropped} slides con categoria fuera de {labels} — excluidos de métricas")
            sub = sub[mask]
        y_true = sub['category'].values.astype(int)
        y_pred = sub['y_pred_class'].values.astype(int)

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        cm_norm = (cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)).round(2)
        cm_norm_df = pd.DataFrame(cm_norm, index=class_names, columns=class_names)

        acc = (y_true == y_pred).mean()
        print(f"\n{'='*60}")
        print(f"Dataset: {ds}  |  slides: {len(sub)}  |  accuracy: {acc:.3f}")
        print(f"{'='*60}")
        print("\nConfusion Matrix (filas=real, columnas=predicho):")
        print(cm_df.to_string())
        print("\nNormalizada por fila (recall):")
        print(cm_norm_df.to_string())
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, labels=labels,
                                    target_names=class_names, digits=3, zero_division=0))

        if all(c in sub.columns for c in prob_cols):
            y_probs = sub[prob_cols].values
            ap_per_class = [
                average_precision_score((y_true == i).astype(int), y_probs[:, i])
                for i in labels
            ]
            macro_ap = float(np.mean(ap_per_class))
            print("PR-AUC por clase:")
            for i, ap in enumerate(ap_per_class):
                print(f"  {class_names[i]}: {ap:.3f}")
            print(f"PR-AUC macro: {macro_ap:.3f}")


if __name__ == '__main__':
    main()