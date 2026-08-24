#!/bin/bash
# ============================================================================
# run_all.sh  —  Orquestador del bundle (extracción + selección de patches)
#
# Ejecuta de principio a fin, dado una carpeta de WSIs:
#   Paso 1  -> extrae features h-optimus-0            (01_extract_features.sh)
#   Paso 2  -> selecciona los patches para patólogos  (02_select_patches.py)
#
# Uso:
#   ./run_all.sh <WSI_DIR> <WORK_DIR> <HF_KEY>
#
# Ejemplo:
#   ./run_all.sh /datos/mis_wsis  ./salida  hf_xxxxxxxxxxxxxxxxx
# ============================================================================
set -euo pipefail

WSI_DIR="${1:?Uso: ./run_all.sh <WSI_DIR> <WORK_DIR> <HF_KEY>}"
WORK_DIR="${2:?Falta WORK_DIR}"
HF_KEY="${3:?Falta HF_KEY (token HuggingFace)}"

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Rutas absolutas (los sub-scripts cambian de directorio internamente)
mkdir -p "$WORK_DIR"
WSI_DIR="$(cd "$WSI_DIR" && pwd)"
WORK_DIR="$(cd "$WORK_DIR" && pwd)"

echo "########## PASO 1: extracción de features h-optimus-0 ##########"
bash "$BUNDLE_DIR/scripts/01_extract_features.sh" "$WSI_DIR" "$WORK_DIR" "$HF_KEY"

echo ""
echo "########## PASO 2: selección de patches para patólogos ##########"
python3 "$BUNDLE_DIR/scripts/02_select_patches.py" \
    --bags_dir "$WORK_DIR/bags/256_128_none_h_optimus_0" \
    --wsi_dir  "$WSI_DIR" \
    --out_dir  "$WORK_DIR/patches" \
    --top_k 8 --top_n 100

echo ""
echo "########## FIN ##########"
echo "Patches finales en: $WORK_DIR/patches/{MSS,MSI}/"
