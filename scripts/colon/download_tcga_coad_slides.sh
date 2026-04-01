#!/bin/bash
# Download labeled TCGA-COAD diagnostic slides from GDC.
# Uses the GDC REST API directly (no gdc-client required).
#
# Input:  config/gdc_manifest_tcga_coad_labeled.txt  (tab-separated GDC manifest)
# Output: /home/PARADIS/colon/wsi/TCGA-COAD/  (SVS files)

set -euo pipefail

MANIFEST="/shared/home/jorgarcia/PathBench-MIL/config/gdc_manifest_tcga_coad_labeled.txt"
OUT_DIR="/home/PARADIS/colon/wsi/TCGA-COAD"
GDC_API="https://api.gdc.cancer.gov/data"
LOG_FILE="$OUT_DIR/download.log"

mkdir -p "$OUT_DIR"

# Count total
TOTAL=$(tail -n +2 "$MANIFEST" | wc -l)
echo "[$(date)] Starting download of $TOTAL slides to $OUT_DIR" | tee -a "$LOG_FILE"

DONE=0
SKIP=0
FAIL=0

while IFS=$'\t' read -r file_id filename md5 size state; do
    # Skip header
    [[ "$file_id" == "id" ]] && continue

    dest="$OUT_DIR/$filename"

    # Skip if already downloaded and size matches
    if [[ -f "$dest" ]]; then
        actual=$(stat -c%s "$dest" 2>/dev/null || echo 0)
        if [[ "$actual" -eq "$size" ]]; then
            ((SKIP++)) || true
            continue
        fi
    fi

    # Download
    http_code=$(curl -s -o "$dest" -w "%{http_code}" \
        -X POST "$GDC_API" \
        -H "Content-Type: application/json" \
        -d "{\"ids\":[\"$file_id\"]}")

    if [[ "$http_code" == "200" ]]; then
        ((DONE++)) || true
        echo "[$(date)] OK ($DONE/$TOTAL) $filename" | tee -a "$LOG_FILE"
    else
        ((FAIL++)) || true
        echo "[$(date)] FAIL (HTTP $http_code) $filename" | tee -a "$LOG_FILE"
        rm -f "$dest"
    fi

done < "$MANIFEST"

echo "[$(date)] Done. Downloaded=$DONE  Skipped=$SKIP  Failed=$FAIL" | tee -a "$LOG_FILE"