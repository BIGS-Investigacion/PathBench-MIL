"""
extract_top_tiles.py

Genera patches JPG de los tiles más y menos atendidos a partir de ficheros _att.npz
ya existentes, sin necesidad de re-ejecutar el benchmark.

Modos de extracción:
  1. Desde tfrecords (por defecto): extrae a la resolución de entrenamiento.
  2. Desde WSI (--wsi_dir):        extrae a resolución nativa (40x por defecto).

Uso básico (desde tfrecords, 256px):
    python pathbench/utils/extract_top_tiles.py \
        --attention_dir experiments/brca_pam50/mil_eval/.../attention \
        --tfrecord_dir /home/PARADIS/mama/tfrecords/CPTAC-BRCA/256px_128um \
        --output_dir experiments/brca_pam50/visualizations/top_tiles \
        --annotation_file config/annotations/annotations_brca_pam50.csv \
        --target category

Uso desde WSI (512px @ 40x para patólogo):
    python pathbench/utils/extract_top_tiles.py \
        --attention_dir experiments/brca_pam50/mil_eval/.../attention \
        --tfrecord_dir /home/PARADIS/mama/tfrecords/CPTAC-BRCA/256px_128um \
        --wsi_dir /home/PARADIS/mama/wsi/CPTAC-BRCA \
        --output_dir experiments/brca_pam50/visualizations/top_tiles_512 \
        --annotation_file config/annotations/annotations_brca_pam50.csv \
        --target category \
        --output_px 512 \
        --tile_um 128

Argumentos opcionales:
    --slides      Lista de nombres de slides a procesar (por defecto todos).
    --n           Número de tiles top/least a guardar por slide (por defecto 5).
    --output_px   Tamaño de salida en píxeles (solo con --wsi_dir, por defecto 512).
    --tile_um     Tamaño físico del tile en µm (solo con --wsi_dir, por defecto 128).
    --wsi_ext     Extensiones WSI a buscar, separadas por comas (por defecto svs,tif,tiff,ndpi).
    --dataset     Etiqueta del dataset para el nombre de fichero.
    --split       Etiqueta del split para el nombre de fichero.
"""

import argparse
import os
import re
import numpy as np
from PIL import Image
import pandas as pd
import slideflow as sf


WSI_EXTENSIONS = ['svs', 'tif', 'tiff', 'ndpi', 'mrxs', 'scn']


# ---------------------------------------------------------------------------
# WSI helpers
# ---------------------------------------------------------------------------

def _get_mpp(tif) -> float:
    """Extract µm/pixel from a tifffile.TiffFile object (Aperio/generic TIFF)."""
    import tifffile
    page = tif.pages[0]
    # Aperio / SVS: MPP in ImageDescription
    desc = page.tags.get('ImageDescription')
    if desc is not None:
        m = re.search(r'MPP\s*=\s*([\d.]+)', desc.value)
        if m:
            return float(m.group(1))
    # Generic TIFF: XResolution tag (pixels per cm or per inch)
    xres = page.tags.get('XResolution')
    unit = page.tags.get('ResolutionUnit')
    if xres is not None:
        val = xres.value
        ratio = val[0] / val[1] if isinstance(val, tuple) else float(val)
        if unit is not None and unit.value == 3:   # centimeter
            return 1e4 / ratio  # µm/px
        elif unit is not None and unit.value == 2:  # inch
            return 25400.0 / ratio
    raise ValueError("No se pudo determinar MPP del slide.")


def _find_wsi(wsi_dir: str, slide_name: str, extensions: list) -> str | None:
    """Busca el fichero WSI correspondiente a slide_name en wsi_dir."""
    for ext in extensions:
        path = os.path.join(wsi_dir, f"{slide_name}.{ext}")
        if os.path.exists(path):
            return path
    # Búsqueda parcial (el nombre del slide puede ser prefijo del fichero)
    for fname in os.listdir(wsi_dir):
        name, ext = os.path.splitext(fname)
        if ext.lstrip('.').lower() in extensions and name.startswith(slide_name):
            return os.path.join(wsi_dir, fname)
    return None


def _extract_from_wsi(wsi_path: str, coord, tile_um: float, output_px: int) -> Image.Image:
    """
    Extrae un patch desde el WSI a resolución nativa y redimensiona a output_px.

    coord: (x, y) en píxeles de nivel 0 (esquina superior izquierda del tile).
    tile_um: tamaño físico del tile en µm.
    output_px: tamaño de salida en píxeles.
    """
    import tifffile
    import zarr

    with tifffile.TiffFile(wsi_path) as tif:
        mpp = _get_mpp(tif)
        native_px = int(round(tile_um / mpp))

        store = tif.aszarr()
        z = zarr.open(store, mode='r')
        # zarr puede ser un Group (pirámide) o un Array (imagen plana)
        level0 = z[0] if isinstance(z, zarr.hierarchy.Group) else z

        x, y = int(coord[0]), int(coord[1])
        h, w = level0.shape[:2]
        # Recortar sin salirse de los bordes
        x2 = min(x + native_px, w)
        y2 = min(y + native_px, h)
        region = level0[y:y2, x:x2, :3]  # solo RGB

    img = Image.fromarray(region.astype('uint8'))
    if img.size != (output_px, output_px):
        img = img.resize((output_px, output_px), Image.LANCZOS)
    return img


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

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
    wsi_dir: str = None,
    output_px: int = 512,
    tile_um: float = 128.0,
    wsi_extensions: list = None,
):
    if wsi_extensions is None:
        wsi_extensions = WSI_EXTENSIONS

    slide_name = os.path.basename(attention_path).replace("_att.npz", "")

    index_path = os.path.join(tfrecord_dir, f"{slide_name}.index.npz")
    if not os.path.exists(index_path):
        print(f"[SKIP] Index no encontrado: {index_path}")
        return

    label_row = annotation_df.loc[annotation_df["slide"] == slide_name]
    label = label_row[target].values[0] if not label_row.empty else "unknown"

    attention = np.load(attention_path)["arr_0"]
    bag_index = np.load(index_path)
    tile_coordinates = bag_index["locations"]

    n = min(n, len(attention))
    top_indices = np.argsort(attention)[::-1][:n]
    least_indices = np.argsort(attention)[:n]

    os.makedirs(output_dir, exist_ok=True)

    prefix = f"{dataset}_{split}_{label}_{slide_name}"
    if save_string:
        prefix = f"{prefix}_{save_string}"

    # --- seleccionar fuente de imagen ---
    use_wsi = wsi_dir is not None
    wsi_path = None
    tfr = None

    if use_wsi:
        wsi_path = _find_wsi(wsi_dir, slide_name, wsi_extensions)
        if wsi_path is None:
            print(f"[WARN] WSI no encontrado para {slide_name}, usando tfrecord.")
            use_wsi = False

    if not use_wsi:
        tfr_path = os.path.join(tfrecord_dir, f"{slide_name}.tfrecords")
        if not os.path.exists(tfr_path):
            print(f"[SKIP] TFRecord no encontrado: {tfr_path}")
            return
        tfr = sf.TFRecord(tfr_path)

    def get_image(idx):
        coord = tile_coordinates[idx]
        if use_wsi:
            return _extract_from_wsi(wsi_path, coord, tile_um, output_px)
        else:
            _, img_arr = tfr.get_record_by_xy(int(coord[0]), int(coord[1]), decode=True)
            if hasattr(img_arr, "cpu"):
                img_arr = img_arr.cpu().numpy()
            return Image.fromarray(img_arr.astype("uint8"))

    for rank, idx in enumerate(top_indices):
        img = get_image(idx)
        fname = f"{prefix}_top{rank+1}_score{attention[idx]:.4f}.jpg"
        img.save(os.path.join(output_dir, fname), format="JPEG", quality=95)

    for rank, idx in enumerate(least_indices):
        img = get_image(idx)
        fname = f"{prefix}_least{rank+1}_score{attention[idx]:.4f}.jpg"
        img.save(os.path.join(output_dir, fname), format="JPEG", quality=95)

    src = f"WSI ({output_px}px)" if use_wsi else "tfrecord"
    print(f"[OK] {slide_name} — {n} top + {n} least tiles [{src}]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extrae top/least tiles como JPG desde ficheros _att.npz"
    )
    parser.add_argument("--attention_dir", required=True,
                        help="Directorio con los ficheros _att.npz")
    parser.add_argument("--tfrecord_dir", required=True,
                        help="Directorio con los .tfrecords e .index.npz")
    parser.add_argument("--output_dir", required=True,
                        help="Directorio de salida para los JPGs")
    parser.add_argument("--annotation_file", required=True,
                        help="CSV de anotaciones (slide, patient, dataset, ...)")
    parser.add_argument("--target", required=True,
                        help="Nombre de la columna objetivo (ej. category)")
    parser.add_argument("--n", type=int, default=5,
                        help="Número de tiles top/least a guardar (por defecto 5)")
    parser.add_argument("--dataset", default="test",
                        help="Etiqueta del dataset para el nombre de fichero")
    parser.add_argument("--split", default="0",
                        help="Etiqueta del split para el nombre de fichero")
    parser.add_argument("--save_string", default="",
                        help="Sufijo adicional para el nombre de fichero")
    parser.add_argument("--slides", nargs="*",
                        help="Lista de slides a procesar (por defecto todos)")
    # WSI mode
    parser.add_argument("--wsi_dir", default=None,
                        help="Directorio con los WSI originales. "
                             "Si se indica, extrae desde el WSI en lugar del tfrecord.")
    parser.add_argument("--output_px", type=int, default=512,
                        help="Tamaño de salida en píxeles cuando se extrae desde WSI (por defecto 512)")
    parser.add_argument("--tile_um", type=float, default=128.0,
                        help="Tamaño físico del tile en µm (por defecto 128)")
    parser.add_argument("--wsi_ext", default="svs,tif,tiff,ndpi,mrxs,scn",
                        help="Extensiones WSI separadas por comas (por defecto svs,tif,tiff,ndpi,mrxs,scn)")
    args = parser.parse_args()

    annotation_df = pd.read_csv(args.annotation_file)
    wsi_extensions = [e.strip().lower() for e in args.wsi_ext.split(',')]

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
            wsi_dir=args.wsi_dir,
            output_px=args.output_px,
            tile_um=args.tile_um,
            wsi_extensions=wsi_extensions,
        )


if __name__ == "__main__":
    main()