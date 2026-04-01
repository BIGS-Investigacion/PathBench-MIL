"""
Generate annotation CSV for TCGA-COAD dataset.

Task: MSS vs MSI-H classification.

MSI labels source:
  UCSC Xena TCGA-COAD clinical matrix (TCGA.COAD.sampleMap/COAD_clinicalMatrix).
  Two columns are used (one complements the other):
    - MSI_updated_Oct62011: MSI-H / MSS / MSI-L  (from the TCGA COAD 2012 paper)
    - microsatellite_instability: YES / NO        (from the TCGA BCR)
  Priority: MSI_updated_Oct62011 if available, else microsatellite_instability.
  MSI-L is treated as MSS (cat 0): clinically closer to MSS in CRC.

Coverage:
  459 diagnostic SVS slides, 451 unique patients.
  ~242 slides have MSI labels (53%); 217 are cases added in the PanCancer Atlas
  2018 that were not systematically tested for MSI in the original cohort.

Output (saved to config/annotations/):
  annotations_tcga_coad_mss_msi.csv  [slide, patient, dataset, category]
    cat 0 = MSS (includes MSI-L)
    cat 1 = MSI-H
"""

import json
from pathlib import Path

import pandas as pd
import requests
from io import StringIO

# ── Configuration ──────────────────────────────────────────────────────────────

ANNOTATIONS_DIR = Path(__file__).resolve().parents[2] / "config" / "annotations"
DATASET_NAME    = "tcga_coad"

XENA_URL = (
    "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/"
    "TCGA.COAD.sampleMap%2FCOAD_clinicalMatrix"
)
GDC_FILES_URL = "https://api.gdc.cancer.gov/files"

# ── Load MSI labels from UCSC Xena ────────────────────────────────────────────

print("Downloading TCGA-COAD clinical matrix from UCSC Xena...")
r = requests.get(XENA_URL, timeout=120)
r.raise_for_status()
clin = pd.read_csv(StringIO(r.text), sep="\t", index_col=0)
clin["patient"] = clin.index.str[:12]   # TCGA-XX-XXXX-01A → TCGA-XX-XXXX


def _msi_label(row) -> int | None:
    """Return 1 (MSI-H), 0 (MSS/MSI-L), or None (unknown)."""
    v1 = row["MSI_updated_Oct62011"]
    v2 = row["microsatellite_instability"]
    if pd.notna(v1):
        return {"MSI-H": 1, "MSS": 0, "MSI-L": 0}.get(v1)
    if pd.notna(v2):
        return {"YES": 1, "NO": 0}.get(v2)
    return None


clin["msi_label"] = clin.apply(_msi_label, axis=1)
patient_msi = (
    clin.groupby("patient")["msi_label"]
    .first()
    .dropna()
    .astype(int)
    .rename("category")
)

n_mss = (patient_msi == 0).sum()
n_msi = (patient_msi == 1).sum()
print(f"  Labeled patients: {len(patient_msi)}  (MSS={n_mss}, MSI-H={n_msi})")

# ── Fetch SVS slide manifest from GDC ─────────────────────────────────────────

print("Fetching TCGA-COAD diagnostic slide manifest from GDC...")
params = {
    "filters": json.dumps({
        "op": "and", "content": [
            {"op": "=", "content": {"field": "cases.project.project_id",   "value": "TCGA-COAD"}},
            {"op": "=", "content": {"field": "data_format",                "value": "SVS"}},
            {"op": "=", "content": {"field": "experimental_strategy",      "value": "Diagnostic Slide"}},
        ]
    }),
    "fields": "file_id,file_name,cases.submitter_id",
    "expand": "cases",
    "size": 1000,
    "format": "json",
}
r = requests.get(GDC_FILES_URL, params=params, timeout=60)
r.raise_for_status()
hits = r.json()["data"]["hits"]

slides = pd.DataFrame([
    {
        "file_id": h["file_id"],
        "slide":   h["file_name"].replace(".svs", ""),
        "patient": h["cases"][0]["submitter_id"],
    }
    for h in hits
])
print(f"  Total slides: {len(slides)}, unique patients: {slides['patient'].nunique()}")

# ── Merge and save ────────────────────────────────────────────────────────────

ann = (
    slides
    .merge(patient_msi.reset_index(), on="patient", how="inner")
    .assign(dataset=DATASET_NAME)
    [["slide", "patient", "dataset", "category"]]
    .sort_values(["patient", "slide"])
    .reset_index(drop=True)
)

out = ANNOTATIONS_DIR / "annotations_tcga_coad_mss_msi.csv"
ann.to_csv(out, index=False)

n_mss_s = (ann["category"] == 0).sum()
n_msi_s = (ann["category"] == 1).sum()
p_mss   = ann[ann["category"] == 0]["patient"].nunique()
p_msi   = ann[ann["category"] == 1]["patient"].nunique()

print(f"\nSaved: {out.name}")
print(f"  MSS   (cat 0): {n_mss_s} slides, {p_mss} patients")
print(f"  MSI-H (cat 1): {n_msi_s} slides, {p_msi} patients")
print(f"  Total labeled: {len(ann)} slides  "
      f"({len(ann)/len(slides)*100:.0f}% of {len(slides)} available)")
print(f"  Unlabeled (excluded): {len(slides)-len(ann)} slides  "
      f"(PanCancer Atlas cases without MSI testing)")