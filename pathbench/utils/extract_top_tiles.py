"""
extract_top_tiles.py

Genera patches JPG de los tiles más y menos atendidos a partir de ficheros _att.npz
ya existentes, sin necesidad de re-ejecutar el benchmark.

Uso básico:
    python pathbench/utils/extract_top_tiles.py \
        --attention_dir experiments/CPTAC_COAD/mil_eval/00001-.../00000-attention_mil/attention \
        --tfrecord_dir /home/PARADIS/colon/wsi/CPTAC-COAD_v1/tfrecords/256px_10x \
        --output_dir experiments/CPTAC_COAD/visualizations/top_tiles \
        --annotation_file config/annotations/annotation_file.csv \
        --target category \
        --n 5

Argumentos opcionales:
    --slides   Lista de nombres de slides a procesar (por defecto todos los del directorio).
    --n        Número de tiles top/least a guardar por slide (por defecto 5).
    --dataset  Etiqueta del dataset (ej. test, val). Solo afecta al nombre de fichero.
    --split    Etiqueta del split (ej. 0, 1). Solo afecta al nombre de fichero.
"""

import argparse
import os
import numpy as np
from PIL import Image
import pandas as pd
import slideflow as sf


def extract_top_tiles(
    attention_path: str,
    tfrecord_dir: str,
    output_dir: str,
    annotation_df: pd.DataFrame,
    target: str,
    n: int = 5,
    dataset: str = "test",
    split: str = "0",
    save_string: str = "",
):
    slide_name = os.path.basename(attention_path).replace("_att.npz", "")

    tfr_path = os.path.join(tfrecord_dir, f"{slide_name}.tfrecords")
    index_path = os.path.join(tfrecord_dir, f"{slide_name}.index.npz")

    if not os.path.exists(tfr_path):
        print(f"[SKIP] TFRecord no encontrado: {tfr_path}")
        return
    if not os.path.exists(index_path):
        print(f"[SKIP] Index no encontrado: {index_path}")
        return

    label_row = annotation_df.loc[annotation_df["slide"] == slide_name]
    label = label_row[target].values[0] if not label_row.empty else "unknown"

    attention = np.load(attention_path)["arr_0"]

    tfr = sf.TFRecord(tfr_path)
    bag_index = np.load(index_path)
    tile_coordinates = bag_index["locations"]

    n = min(n, len(attention))
    top_indices = np.argsort(attention)[::-1][:n]
    least_indices = np.argsort(attention)[:n]

    os.makedirs(output_dir, exist_ok=True)

    prefix = f"{dataset}_{split}_{label}_{slide_name}"
    if save_string:
        prefix = f"{prefix}_{save_string}"

    for rank, idx in enumerate(top_indices):
        coord = tile_coordinates[idx]
        _, img_arr = tfr.get_record_by_xy(int(coord[0]), int(coord[1]), decode=True)
        if hasattr(img_arr, "cpu"):
            img_arr = img_arr.cpu().numpy()
        img = Image.fromarray(img_arr.astype("uint8"))
        fname = f"{prefix}_top{rank+1}_score{attention[idx]:.4f}.jpg"
        img.save(os.path.join(output_dir, fname), format="JPEG", quality=95)

    for rank, idx in enumerate(least_indices):
        coord = tile_coordinates[idx]
        _, img_arr = tfr.get_record_by_xy(int(coord[0]), int(coord[1]), decode=True)
        if hasattr(img_arr, "cpu"):
            img_arr = img_arr.cpu().numpy()
        img = Image.fromarray(img_arr.astype("uint8"))
        fname = f"{prefix}_least{rank+1}_score{attention[idx]:.4f}.jpg"
        img.save(os.path.join(output_dir, fname), format="JPEG", quality=95)

    print(f"[OK] {slide_name} — {n} top + {n} least tiles guardados")


def main():
    parser = argparse.ArgumentParser(description="Extrae top/least tiles como JPG desde ficheros _att.npz")
    parser.add_argument("--attention_dir", required=True, help="Directorio con los ficheros _att.npz")
    parser.add_argument("--tfrecord_dir", required=True, help="Directorio con los .tfrecords e .index.npz")
    parser.add_argument("--output_dir", required=True, help="Directorio de salida para los JPGs")
    parser.add_argument("--annotation_file", required=True, help="CSV de anotaciones (slide, patient, dataset, ...)")
    parser.add_argument("--target", required=True, help="Nombre de la columna objetivo (ej. category)")
    parser.add_argument("--n", type=int, default=5, help="Número de tiles top/least a guardar (por defecto 5)")
    parser.add_argument("--dataset", default="test", help="Etiqueta del dataset para el nombre de fichero")
    parser.add_argument("--split", default="0", help="Etiqueta del split para el nombre de fichero")
    parser.add_argument("--save_string", default="", help="Sufijo adicional para el nombre de fichero")
    parser.add_argument("--slides", nargs="*", help="Lista de slides a procesar (por defecto todos)")
    args = parser.parse_args()

    annotation_df = pd.read_csv(args.annotation_file)

    npz_files = sorted(f for f in os.listdir(args.attention_dir) if f.endswith("_att.npz"))

    if args.slides:
        npz_files = [f for f in npz_files if f.replace("_att.npz", "") in args.slides]

    print(f"Procesando {len(npz_files)} slides...")
    for fname in npz_files:
        extract_top_tiles(
            attention_path=os.path.join(args.attention_dir, fname),
            tfrecord_dir=args.tfrecord_dir,
            output_dir=args.output_dir,
            annotation_df=annotation_df,
            target=args.target,
            n=args.n,
            dataset=args.dataset,
            split=args.split,
            save_string=args.save_string,
        )


if __name__ == "__main__":
    main()