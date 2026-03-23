"""
Script de prueba para replicar la generación de predictions.parquet de validación.

La diferencia con test:
  - Val predictions las genera train_mil() internamente al final del entrenamiento
    usando val_dataset (el split 15% de TCGA sin solapamiento de paciente)
  - Test predictions las genera eval_mil() explícitamente sobre el dataset de test completo

Este script carga el split guardado (fixed_classification.json), reconstruye el
val_dataset y llama a eval_mil() con el modelo entrenado — equivalente a lo que
slideflow hace internamente al final de train_mil().

Uso:
    python scripts/test_eval_val.py

Ajusta las variables de la sección CONFIG si es necesario.
"""

import os
import sys
import glob
import logging
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import slideflow as sf
from slideflow.mil import eval_mil

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# ─── CONFIG ────────────────────────────────────────────────────────────────────
PROJECT_NAME    = "brca_pam50_virchow2_ho_tcga_tcga"
ANNOTATION_FILE = "./config/annotations/annotations_brca_pam50.csv"
OUTCOMES        = "category"

# Modelo entrenado (el mismo que se usa para test)
MODEL_DIR = (
    f"experiments/{PROJECT_NAME}/mil/"
    "00000-256_128_macenko_virchow2_clam_mil_mb_CrossEntropyLoss_ReLU_AdamW_1"
)

# Bags (macenko)
BAGS_PATH = "/home/PARADIS/mama/tfrecords/shared_bags/256_128_macenko_virchow2"

# Fichero de splits guardado por benchmark.py
SPLITS_FILE = f"experiments/{PROJECT_NAME}/fixed_classification.json"

# Dónde guardar la salida de este script
OUT_DIR = f"experiments/{PROJECT_NAME}/val_eval_manual/macenko"
# ───────────────────────────────────────────────────────────────────────────────


def main():
    project_path = f"experiments/{PROJECT_NAME}"
    if not os.path.exists(project_path):
        raise FileNotFoundError(f"Proyecto no encontrado en {project_path}")

    # 1. Cargar proyecto
    project = sf.Project(project_path, annotations=ANNOTATION_FILE)

    # 2. Obtener el dataset completo de training (fuente TCGA usada para entrenar)
    full_dataset = project.dataset(tile_px=256, tile_um=128)
    logging.info(f"Dataset completo: {len(full_dataset.slides())} slides")

    # 3. Reconstruir el mismo split que usó benchmark.py
    #    read_only=True garantiza que no se modifique el fichero de splits
    train_ds, val_ds = full_dataset.split(
        labels=OUTCOMES,
        model_type="categorical",
        val_strategy="fixed",
        val_fraction=0.15,
        splits=SPLITS_FILE,
        read_only=True,
    )
    logging.info(f"Train slides: {len(train_ds.slides())}")
    logging.info(f"Val slides  : {len(val_ds.slides())}")

    # 4. Evaluar sobre val_dataset con el modelo entrenado
    os.makedirs(OUT_DIR, exist_ok=True)
    logging.info(f"Evaluando modelo: {MODEL_DIR}")
    logging.info(f"Salida: {OUT_DIR}")

    result = eval_mil(
        weights=MODEL_DIR,
        outcomes=OUTCOMES,
        dataset=val_ds,
        bags=BAGS_PATH,
        outdir=OUT_DIR,
    )

    # 5. Comparar con las predictions.parquet originales de val
    original_val = f"experiments/{PROJECT_NAME}/mil/00000-256_128_macenko_virchow2_clam_mil_mb_CrossEntropyLoss_ReLU_AdamW_1/predictions.parquet"

    import pandas as pd
    pq_files = glob.glob(os.path.join(OUT_DIR, "*/predictions.parquet"))
    if pq_files:
        df_new = pd.read_parquet(pq_files[0])
        csv_path = pq_files[0].replace(".parquet", ".csv")
        df_new.to_csv(csv_path, index=False)
        logging.info(f"Predictions guardadas en: {csv_path}")

        print("\n=== Predicciones val (reconstruidas) ===")
        print(df_new.head(10).to_string())
        y_pred_cols = [c for c in df_new.columns if c.startswith("y_pred")]
        y_pred_class = np.argmax(df_new[y_pred_cols].values, axis=1)
        print(f"\ny_true distribución: {df_new['y_true'].value_counts().sort_index().to_dict()}")
        print(f"y_pred distribución: {dict(zip(*np.unique(y_pred_class, return_counts=True)))}")

    if os.path.exists(original_val):
        df_orig = pd.read_parquet(original_val)
        print("\n=== Predicciones val (originales de train_mil) ===")
        print(df_orig.head(10).to_string())
        y_pred_cols_o = [c for c in df_orig.columns if c.startswith("y_pred")]
        y_pred_class_o = np.argmax(df_orig[y_pred_cols_o].values, axis=1)
        print(f"\ny_true distribución: {df_orig['y_true'].value_counts().sort_index().to_dict()}")
        print(f"y_pred distribución: {dict(zip(*np.unique(y_pred_class_o, return_counts=True)))}")
    else:
        logging.warning(f"No se encontró el parquet original de val en: {original_val}")


if __name__ == "__main__":
    main()