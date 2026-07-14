# FUSION.xlsx — Generation of the patches annotated by the pathologists

This document describes the full pipeline used to select, export and annotate the
histological patches reviewed by the pathologists, whose final consensus is stored
in `FUSION.xlsx`.

## Summary

600 patches (200 per cohort: **Macarena**, **TCGA-COAD**, **CPTAC-COAD**;
100 MSS + 100 MSI in each) automatically selected as the most representative of
each class, exported as images, and independently annotated by two pathologists
(LMG and MPP) over 7 binary morphological features, followed by consensus.

## Input data

| Cohort      | WSI format | # slides | MSI/MSS label source |
|-------------|------------|----------|-----------------------|
| Macarena    | `.tiff`    | 243      | (to be confirmed)     |
| TCGA-COAD   | `.svs`     | 242      | UCSC Xena clinical matrix |
| CPTAC-COAD  | `.svs`     | 223      | Vasaikar et al. 2019 (mmc2) |

## Pipeline

### 1. Quality control and tiling (Slideflow)
- Automatic QC: **GaussianV2 + Otsu-CLAHE**.
- Filters: `grayspace_threshold=0.05`, `grayspace_fraction=0.6`,
  `whitespace_threshold=230`, `whitespace_fraction=1.0`.
- Tiling at **256 px / 128 µm** per tile → **0.5 µm/px (20x)**.

### 2. Feature extraction
- **H-Optimus-0** extractor (1536 dimensions per patch).
- Bags stored under `256_128_none_h_optimus_0` (no stain normalization).

### 3. MIL model training
- **DSMIL** model trained on Macarena for the **MSS vs MSI** task
  (`experiments/colon_mss_msi_benchmark/mil/00000-256_128_none_h_optimus_0_dsmil_...`).

### 4. Attention-based patch selection
`scripts/colon/extract_attention_patches.py`
- For each slide, DSMIL attention is computed per patch.
- The **top-8 most attended patches** per slide are extracted → `top8_patches_{cohort}.csv`.
- The **per-class feature centroid** (mean of the top-K) is computed →
  `centroids_{cohort}.csv`.

### 5. Selection of the 100 closest to the centroid
`scripts/colon/select_closest_to_centroid.py`
- The H-Optimus-0 feature of each top-8 patch is loaded.
- The **cosine distance to its class centroid** is computed.
- The **100 closest patches per class** (MSS and MSI) are selected →
  `results/colon/closest100_{cohort}/{MSS,MSI}/`.
- Result: 200 patches per cohort (100 MSS + 100 MSI), 600 in total.

### 6. Image export for review
- The images shown to the pathologists were exported at **512 px at 20x**
  (256 µm field of view; MPP stays at 0.5 µm/px).
- Original naming: `{rank:03d}_{slide}_d{cosine_distance}.png`
  (e.g. `001_143-4_d0.2217.png`), where `d` is the cosine distance to the centroid.
- Anonymized published name: `{COHORT}_{NNNNNN}.png` (e.g. `MAC_000001.png`).

### 7. Pathologist annotation and consensus
- Two pathologists (**LMG** and **MPP**) independently annotated 7 binary
  morphological features per patch:
  `Arquitectura glandular`, `Diferenciación tumoral`, `Serración glandular`,
  `Mucina`, `TILs intraepiteliales`, `Inflamación peritumoral`,
  `Necrosis intraluminal`. → `LMG.xlsx`, `MPP.xlsx`.
- The **final consensus** was built (`FUSION.xlsx`, sheet `consensus_final`):
  - `consensus_status`: `auto_agreement` (direct match) or
    `consensus_resolved` (adjudicated disagreement).
  - `n_disagreement_tags` / `disagreement_tags`: number and name of the features
    with an initial disagreement.

## Reproduction — commands

Run from the repository root, with the environment active:

```bash
source /shared/home/jorgarcia/PathBench-MIL/pathbench_env/bin/activate
cd /shared/home/jorgarcia/PathBench-MIL
```

### Step 0 — MSI/MSS annotations (if not already generated)
```bash
python scripts/colon/generate_tcga_coad_annotations.py
python scripts/colon/generate_cptac_coad_annotations.py
# Macarena labels: provided externally (method to be confirmed)
```

### Steps 1–3 — QC, tiling, features and DSMIL training
Handled by the PathBench extraction/benchmark config
(`conf_colon_h_optimus_0_extract.yaml` and the `colon_mss_msi_benchmark`
experiment). The trained DSMIL checkpoint already exists under
`experiments/colon_mss_msi_benchmark/mil/00000-256_128_none_h_optimus_0_dsmil_.../`.
Re-run only if the bags or the model need to be regenerated.

### Step 4 — attention patches + class centroids (per cohort)
```bash
python scripts/colon/extract_attention_patches.py --top_k 8            # macarena (default)
# add --force to overwrite existing patches
```
Produces `top8_patches_{cohort}.csv`, `centroids_{cohort}.csv` and the patch
images under `results/colon/top_patches/{cohort}/`.

### Step 5 — 100 closest to centroid (run once per cohort)
```bash
python scripts/colon/select_closest_to_centroid.py --dataset macarena   --top_n 100
python scripts/colon/select_closest_to_centroid.py --dataset tcga_coad  --top_n 100
python scripts/colon/select_closest_to_centroid.py --dataset cptac_coad --top_n 100
```
Produces `results/colon/closest100_{cohort}/{MSS,MSI}/` (200 patches per cohort).

### Step 6–7 — export, anonymization and consensus
Image export at 512 px, filename anonymization and pathologist adjudication were
done manually / externally; the results are `LMG.xlsx`, `MPP.xlsx` and
`FUSION.xlsx`.

### Downstream analysis (already available)
```bash
# Reformat the consensus to the LMG.xlsx schema
python -c "import scripts"   # (see FUSION_formato_LMG.xlsx generator, if kept as a script)

# Inter-observer agreement (Cohen's kappa)
python scripts/colon/pathologist_agreement.py

# Feature ↔ MSI association, per cohort (Fisher + FDR)
python scripts/colon/feature_significance.py
```

## Structure of `FUSION.xlsx` (sheet `consensus_final`)

| Column | Description |
|--------|-------------|
| `image_id` | Anonymized identifier (e.g. `MAC_000001`) |
| `cohort` | `macarena` / `tcga_coad` / `cptac_coad` |
| `original_filename` | Original name including the distance to the centroid |
| `published_filename` | Anonymized published name (`.png`) |
| `msi_status` | `MSS` / `MSI` (reference molecular label) |
| 7 feature columns | Consensus value (0/1) |
| `n_disagreement_tags` | # of features with an initial disagreement |
| `disagreement_tags` | Features in disagreement (`;`-separated) |
| `consensus_status` | `auto_agreement` / `consensus_resolved` |

## Related files

- `LMG.xlsx`, `MPP.xlsx` — individual annotations of each pathologist.
- `FUSION_formato_LMG.xlsx` — consensus reformatted to the `LMG.xlsx` schema.
- `scripts/colon/pathologist_agreement.py` — inter-observer agreement (kappa).
- `scripts/colon/feature_significance.py` — feature ↔ MSI association per cohort.

## References

- The Cancer Genome Atlas Network. *Comprehensive molecular characterization of
  human colon and rectal cancer.* Nature 2012;487(7407):330–337.
  doi:10.1038/nature11252
- Vasaikar S, et al. *Proteogenomic Analysis of Human Colon Cancer Reveals New
  Therapeutic Opportunities.* Cell 2019;177(4):1035–1049.e19.
  doi:10.1016/j.cell.2019.03.030
- Goldman MJ, et al. *Visualizing and interpreting cancer genomics data via the
  Xena platform.* Nat Biotechnol 2020;38(6):675–678. doi:10.1038/s41587-020-0546-8