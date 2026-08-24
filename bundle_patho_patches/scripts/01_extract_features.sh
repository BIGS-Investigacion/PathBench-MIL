#!/bin/bash
# ============================================================================
# 01_extract_features.sh
# ----------------------
# Paso 1 del bundle: extrae los features h-optimus-0 de una carpeta de WSIs,
# reutilizando el stack de Slideflow/PathBench (mismo tiling/QC que el estudio).
#
# Produce, para cada slide:
#   <BAGS_DIR>/256_128_none_h_optimus_0/<slide>.pt         (features, N x 1536)
#   <BAGS_DIR>/256_128_none_h_optimus_0/<slide>.index.npz  (coordenadas x,y)
#
# Uso:
#   ./01_extract_features.sh <WSI_DIR> <WORK_DIR> <HF_KEY>
#
#   WSI_DIR : carpeta con las WSIs de entrada
#   WORK_DIR: carpeta de trabajo (tiles, tfrecords y bags se crean dentro)
#   HF_KEY  : token de HuggingFace (para descargar los pesos de h-optimus-0)
# ============================================================================
set -euo pipefail

WSI_DIR="${1:?Falta WSI_DIR}"
WORK_DIR="${2:?Falta WORK_DIR}"
HF_KEY="${3:?Falta HF_KEY (token HuggingFace)}"

# Resolver rutas ABSOLUTAS: main.py se ejecuta desde el repo (cd más abajo),
# así que las rutas relativas romperían. Las normalizamos aquí.
mkdir -p "$WORK_DIR"
WSI_DIR="$(cd "$WSI_DIR" && pwd)"
WORK_DIR="$(cd "$WORK_DIR" && pwd)"

# Rutas del bundle y del repo PathBench (necesario para slideflow + h-optimus)
BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="${PATHBENCH_REPO:-$(cd "$BUNDLE_DIR/.." && pwd)}"   # por defecto, el repo padre

mkdir -p "$WORK_DIR"
ANNOT="$WORK_DIR/annotation.csv"
BAGS_DIR="$WORK_DIR/bags"
TFREC_DIR="$WORK_DIR/tfrecords"
CONF="$WORK_DIR/conf_extract.yaml"

echo "==> [1/3] Generando anotación (lista de slides)"
python3 "$BUNDLE_DIR/scripts/00_make_annotation.py" "$WSI_DIR" "$ANNOT"

echo "==> [2/3] Generando config de extracción desde la plantilla"
sed -e "s#@WSI_DIR@#$WSI_DIR#g" \
    -e "s#@TFREC_DIR@#$TFREC_DIR#g" \
    -e "s#@BAGS_DIR@#$BAGS_DIR#g" \
    -e "s#@ANNOT@#$ANNOT#g" \
    -e "s#@HF_KEY@#$HF_KEY#g" \
    "$BUNDLE_DIR/configs/extract_template.yaml" > "$CONF"

echo "==> [3/3] Ejecutando extracción de features (Slideflow/PathBench)"
# Entorno: en una instalación sana bastaría 'source pathbench_env/bin/activate'.
# Aquí usamos el lanzador del repo que ya resuelve el entorno recuperado.
cd "$REPO_DIR"
export SF_SLIDE_BACKEND=cucim
export SF_BACKEND=torch
python3 main.py --config "$CONF"

echo "==> Extracción terminada. Bags en: $BAGS_DIR/256_128_none_h_optimus_0/"
