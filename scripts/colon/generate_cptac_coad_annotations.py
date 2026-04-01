"""
Generate annotation CSV for CPTAC-COAD dataset.

Task: MSS vs MSI-H classification.

MSI labels source:
  Vasaikar et al. 2019 (Cell, doi:10.1016/j.cell.2019.07.003), mmc2.xlsx,
  sheet "B-Summary-total-msi-pcr", column "MSMutect-Fisher decision".
  This is the final per-patient MSI call from the paper, combining MSMuTect
  microsatellite-locus analysis with PCR validation where available.

  Results: 24 MSI-H, 82 MSS, 3 without call (excluded).
  Note: a TMB > 200 proxy gives 32 "MSI-H" because POLE-hypermutated tumours
  also have high TMB but are NOT microsatellite-unstable; MSMuTect correctly
  excludes them.

  cat 0 = MSS
  cat 1 = MSI-H
  (MSI-low cases are classified as MSS by MSMuTect-Fisher and excluded from
  the MSI-H group)

Coverage:
  372 SVS files, 178 unique patients.
  106 patients have MSI calls; 72 have slides but no MSI data → excluded.

Input files (expected at project root):
  mmc2.xlsx  — Vasaikar 2019 supplementary table 2

Output (saved to config/annotations/):
  annotations_cptac_coad_mss_msi.csv  [slide, patient, dataset, category]
"""

import os
from pathlib import Path

import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────

SLIDE_DIR       = Path("/home/PARADIS/colon/wsi/CPTAC-COAD_v1/COAD")
PROJECT_ROOT    = Path(__file__).resolve().parents[2]
ANNOTATIONS_DIR = PROJECT_ROOT / "config" / "annotations"
MMC2_PATH       = PROJECT_ROOT / "mmc2.xlsx"
DATASET_NAME    = "cptac_coad"

# ── Load MSI labels from Vasaikar 2019 mmc2 ───────────────────────────────────

print("Reading MSI labels from Vasaikar 2019 (mmc2.xlsx)...")
raw = pd.read_excel(MMC2_PATH, sheet_name="B-Summary-total-msi-pcr")
# Row 0 contains the real column names
raw.columns = raw.iloc[0]
raw = raw.iloc[1:].reset_index(drop=True)
raw.columns.name = None

msi_raw = raw[["idx", "MSMutect-Fisher decision"]].rename(
    columns={"idx": "patient", "MSMutect-Fisher decision": "msi_call"}
).dropna(subset=["patient", "msi_call"])

msi_map = {"MSI-high": 1, "MSS": 0}
msi_raw["category"] = msi_raw["msi_call"].map(msi_map)
msi_labels = msi_raw.dropna(subset=["category"]).set_index("patient")["category"].astype(int)

n_mss = (msi_labels == 0).sum()
n_msi = (msi_labels == 1).sum()
print(f"  Labeled patients: {len(msi_labels)}  (MSS={n_mss}, MSI-H={n_msi})")

# ── Build slide → patient mapping ─────────────────────────────────────────────

svs_files = [f for f in os.listdir(SLIDE_DIR) if f.endswith(".svs")]
slide_df = pd.DataFrame({"slide_file": svs_files})
slide_df["patient"] = slide_df["slide_file"].str.extract(r"^([0-9A-Z0-9]+CO[0-9]+)")
slide_df["slide"]   = slide_df["slide_file"].str.replace(".svs", "", regex=False)
slide_df = slide_df.drop(columns="slide_file")

# ── Merge and save ────────────────────────────────────────────────────────────

ann = (
    slide_df
    .merge(msi_labels.rename("category").reset_index(), on="patient", how="inner")
    .assign(dataset=DATASET_NAME)
    [["slide", "patient", "dataset", "category"]]
    .sort_values(["patient", "slide"])
    .reset_index(drop=True)
)

out = ANNOTATIONS_DIR / "annotations_cptac_coad_mss_msi.csv"
ann.to_csv(out, index=False)

n_mss_s = (ann["category"] == 0).sum()
n_msi_s = (ann["category"] == 1).sum()
p_mss   = ann[ann["category"] == 0]["patient"].nunique()
p_msi   = ann[ann["category"] == 1]["patient"].nunique()

print(f"\nSaved: {out.name}")
print(f"  MSS   (cat 0): {n_mss_s} slides, {p_mss} patients")
print(f"  MSI-H (cat 1): {n_msi_s} slides, {p_msi} patients")
print(f"  Total: {len(ann)} slides, {ann['patient'].nunique()} patients")
print(f"  Excluded: {len(slide_df)-len(ann)} slides (no MSI call in Vasaikar 2019)")