"""
Script de prueba para replicar la generación de predictions.csv de mil_eval.
Llama a eval_mil directamente con el modelo entrenado y el dataset de test.

Uso:
    python scripts/test_eval_mil.py

Ajusta las variables de la sección CONFIG antes de ejecutar.
"""

import os
import sys
import logging

# Añadir el root del proyecto al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import slideflow as sf
from slideflow.mil import eval_mil, mil_config

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# ─── CONFIG ────────────────────────────────────────────────────────────────────
PROJECT_NAME    = "brca_pam50_virchow2_ho_tcga_tcga"
ANNOTATION_FILE = "./config/annotations/annotations_brca_pam50.csv"
OUTCOMES        = "category"

# Modelo a evaluar (directorio con mil_params.json y best_epoch_*.pt)
MODEL_DIR = (
    f"experiments/{PROJECT_NAME}/mil/"
    "00000-256_128_macenko_virchow2_clam_mil_mb_CrossEntropyLoss_ReLU_AdamW_1"
)

# Bags del test set (macenko)
BAGS_PATH = "/home/PARADIS/mama/tfrecords/shared_bags/256_128_macenko_virchow2"

# Dónde guardar resultados
OUT_DIR = f"experiments/{PROJECT_NAME}/mil_eval_test/macenko_manual"

# Dataset: slides y tfrecords de test
SLIDE_PATH   = "/home/PARADIS/mama/wsi/TCGA-BRCA"
TFRECORD_PATH = "/home/PARADIS/mama/tfrecords/TCGA-BRCA"
# ───────────────────────────────────────────────────────────────────────────────


def main():
    os.makedirs("experiments", exist_ok=True)

    # 1. Cargar o crear el proyecto
    project_path = f"experiments/{PROJECT_NAME}"
    if os.path.exists(project_path):
        project = sf.Project(project_path, annotations=ANNOTATION_FILE)
    else:
        raise FileNotFoundError(f"Proyecto no encontrado en {project_path}")

    # 2. Construir el dataset de test (equivalente a individual_test_set en benchmark.py)
    # Filtramos únicamente los slides que pertenecen al split de test.
    # Para replicar benchmark.py exactamente usamos el dataset completo y dejamos
    # que slideflow filtre por los bags disponibles.
    test_dataset = project.dataset(
        tile_px=256,
        tile_um=128,
    )
    logging.info(f"Dataset de test: {len(test_dataset.slides())} slides")

    # 3. Llamar a eval_mil (replica lo que hace benchmark.py en la línea ~541)
    os.makedirs(OUT_DIR, exist_ok=True)
    logging.info(f"Evaluando modelo: {MODEL_DIR}")
    logging.info(f"Bags: {BAGS_PATH}")
    logging.info(f"Salida: {OUT_DIR}")

    result = eval_mil(
        weights=MODEL_DIR,
        outcomes=OUTCOMES,
        dataset=test_dataset,
        bags=BAGS_PATH,
        outdir=OUT_DIR,
    )

    # 4. Guardar CSV además del parquet
    predictions_parquet = os.path.join(OUT_DIR, "00000-*/predictions.parquet")
    import glob
    pq_files = glob.glob(predictions_parquet)
    if pq_files:
        import pandas as pd
        df = pd.read_parquet(pq_files[0])
        csv_path = pq_files[0].replace(".parquet", ".csv")
        df.to_csv(csv_path, index=False)
        logging.info(f"Predictions guardadas en: {csv_path}")
        print("\nPrimeras filas:")
        print(df.head(10).to_string())
        print(f"\nDistribución y_true: {df['y_true'].value_counts().sort_index().to_dict()}")
        import numpy as np
        y_pred_cols = [c for c in df.columns if c.startswith("y_pred")]
        if y_pred_cols:
            y_pred_class = np.argmax(df[y_pred_cols].values, axis=1)
            print(f"Distribución y_pred (argmax): {dict(zip(*np.unique(y_pred_class, return_counts=True)))}")
    else:
        logging.warning("No se encontró predictions.parquet en la salida.")
        if result is not None:
            print(result.head(10))


if __name__ == "__main__":
    main()