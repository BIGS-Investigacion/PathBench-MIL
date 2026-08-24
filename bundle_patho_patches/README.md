# Bundle: patch extraction for pathologist review

Self-contained pipeline that, **given a folder of WSIs**, reproduces the patch
selection shown to the pathologists in the colon MSS/MSI study:

1. **Extracts features** with **h-optimus-0** (tiling 256px @ 20x, QC GaussianV2 +
   Otsu-CLAHE) reusing Slideflow/PathBench.
2. **Loads the optimal DSMIL model** (the Macarena benchmark model).
3. **Selects the patches**: top-8 by attention per slide and, out of all of them,
   the **100 closest** (cosine distance) to each **reference class centroid**
   (Macarena MSS and MSI, shipped inside the bundle).

The output is two image folders, `MSS/` and `MSI/`, with 100 patches each, plus
CSVs with the distances.

---

## Bundle contents

```
bundle_patho_patches/
├── README.md                     ← this file
├── env.sh                        ← activates the environment (edit paths if needed)
├── run_all.sh                    ← runs both steps end to end
├── model/
│   ├── best_dsmil.ckpt           ← optimal DSMIL checkpoint
│   └── mil_params.json           ← model hyperparameters
├── reference/
│   └── centroids_macarena.csv    ← reference MSS/MSI centroids (1536 dims)
├── configs/
│   └── extract_template.yaml     ← feature-extraction config template
└── scripts/
    ├── 00_make_annotation.py     ← lists the WSIs in the folder
    ├── 01_extract_features.sh    ← step 1: h-optimus-0 features
    └── 02_select_patches.py      ← step 2: patch selection
```

---

## Requirements

- Access to the **PathBench-MIL** repository (for `slideflow` and the DSMIL model
  definition). By default the bundle is assumed to live **inside** the repo; if
  not, export `PATHBENCH_REPO=/path/to/repo`.
- An environment with: `torch`, `pytorch_lightning`, `slideflow`,
  `openslide-python`, `pandas`, `numpy`, `Pillow`. (See `env.sh`.)
- A **HuggingFace token** with access to the `h-optimus-0` weights.
- A GPU is recommended (extraction and attention are much faster on GPU).

---

## Quick start (all at once)

```bash
cd bundle_patho_patches
source env.sh                     # activate the environment
./run_all.sh  <WSI_DIR>  <WORK_DIR>  <HF_KEY>
```

- `WSI_DIR` : folder with the WSIs (`.svs`, `.tiff`, `.ndpi`, …).
- `WORK_DIR`: work/output folder (created if missing).
- `HF_KEY`  : your HuggingFace token.

Result: `WORK_DIR/patches/MSS/` and `WORK_DIR/patches/MSI/` (100 PNGs each).

---

## Step-by-step usage (recommended to understand the flow)

### Step 0 — activate the environment
```bash
cd bundle_patho_patches
source env.sh
```
If your PathBench install is healthy, edit `env.sh` to simply use
`source .../pathbench_env/bin/activate` (Option A).

### Step 1 — extract h-optimus-0 features
```bash
./scripts/01_extract_features.sh  <WSI_DIR>  <WORK_DIR>  <HF_KEY>
```
What it does:
1. `00_make_annotation.py` scans `WSI_DIR` and creates `WORK_DIR/annotation.csv`
   (list of slides; the class is a *dummy* = 0, unused here).
2. Fills `configs/extract_template.yaml` with your paths → `WORK_DIR/conf_extract.yaml`.
3. Runs the extraction (`python main.py --config …`).

Output:
```
WORK_DIR/bags/256_128_none_h_optimus_0/<slide>.pt         # features N x 1536
WORK_DIR/bags/256_128_none_h_optimus_0/<slide>.index.npz  # coordinates (x,y)
```
> FIXED parameters (do not change, to match the study): tile 256px / 128 µm
> (= 0.5 µm/px, 20x), normalization `none`, QC GaussianV2 + Otsu-CLAHE.

### Step 2 — select the pathologist patches
```bash
python3 scripts/02_select_patches.py \
    --bags_dir WORK_DIR/bags/256_128_none_h_optimus_0 \
    --wsi_dir  WSI_DIR \
    --out_dir  WORK_DIR/patches \
    --top_k 8 --top_n 100
```
What it does:
1. Loads the DSMIL (`model/`) and the reference centroids (`reference/`).
2. For each slide: attention → **top-8** patches → saves the image (3×3 region
   centered on the patch, context for the pathologist) → cosine distance to the
   MSS and MSI centroids.
3. Out of all top-8: the **100 closest** to MSS → `patches/MSS/`; the 100 closest
   to MSI → `patches/MSI/`.

Output:
```
WORK_DIR/patches/MSS/001_<slide>_d0.2217.png   # rank_slide_distance
WORK_DIR/patches/MSI/001_<slide>_d0.1985.png
WORK_DIR/patches/all_topk_patches.csv          # all top-8 with distances
WORK_DIR/patches/closest100_MSS.csv
WORK_DIR/patches/closest100_MSI.csv
```

---

## Running on the cluster (SLURM)

```bash
sbatch --gpus=1 --cpus-per-task=8 --mem=48G --time=04:00:00 \
  --wrap "cd bundle_patho_patches && source env.sh && \
          ./run_all.sh /data/my_wsis ./output hf_xxxxxxxx"
```

---

## Notes and design decisions

- **No input labels.** The class of each patch is unknown; it is decided by
  **proximity to the Macarena centroids** (shipped in `reference/`). This keeps
  the selection criterion identical to the study without annotating the WSIs.
- **Model.** `model/best_dsmil.ckpt` is the DSMIL from the Macarena benchmark
  (z_dim 270, encoder_layers 1, dropout 0.3966), the one used to select the
  original patches.
- **Reproducibility.** Tiling/QC/extractor parameters are fixed in the template
  to match the study's bags.
