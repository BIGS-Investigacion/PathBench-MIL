"""
Centroid distance analysis between two datasets in the FM embedding space.

Algorithm:
  1. For each WSI, run the MIL model and select the top-K patches by attention
     weight (default K=8).
  2. Collect all selected patches per (dataset, class) using ground-truth labels.
     Compute the class centroid as the mean of those patches.
  3. From the global pool of top-K patches per class, find the N patches
     closest to the centroid (default N=25) and save them as JPEG images.
  4. Compute d_c = ||μ_c^A − μ_c^B||₂ between the two datasets.

Usage:
    python scripts/centroid_distance.py \
        --model_dir experiments/.../mil/00000-... \
        --bags_dir /home/PARADIS/mama/tfrecords/shared_bags/256_128_macenko_virchow2 \
        --annotations config/annotations/annotations_brca_pam50.csv \
        --labels 0:Basal 1:Her2 2:LumA 3:LumB 4:Normal \
        --top_k_attn 8 \
        --top_n_images 25 \
        --output centroid_distances.csv \
        --patches_dir top_patches/ \
        --tfrecord_dirs /home/PARADIS/mama/tfrecords/TCGA-BRCA/256px_128um \
                        /home/PARADIS/mama/tfrecords/CPTAC-BRCA/256px_128um
"""

import argparse
import importlib.util
import io
import json
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_aggregators_module():
    agg_path = os.path.join(ROOT, "pathbench", "models", "aggregators.py")
    spec = importlib.util.spec_from_file_location("aggregators", agg_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_checkpoint(model_dir):
    for root, _, files in os.walk(model_dir):
        for f in files:
            if f.endswith(".ckpt"):
                return os.path.join(root, f)
    raise FileNotFoundError(f"No .ckpt found under {model_dir}")


def load_model(model_dir, aggregators_module):
    with open(os.path.join(model_dir, "mil_params.json")) as f:
        mil_params = json.load(f)
    params = mil_params["params"]
    model_cls = getattr(aggregators_module, params["model"])
    model = model_cls(
        n_feats=mil_params["input_shape"],
        n_out=mil_params["output_shape"],
        z_dim=params.get("z_dim", 256),
        dropout_p=params.get("dropout_p", 0.1),
        activation_function=params.get("activation_function", "ReLU"),
        encoder_layers=params.get("encoder_layers", 1),
    )
    ckpt_path = find_checkpoint(model_dir)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = {
        k.removeprefix("model."): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    has_attn = hasattr(model, "calculate_attention")
    print(f"Loaded '{params['model']}' from {ckpt_path}")
    print(f"  Attention support: {'yes' if has_attn else 'no — uniform weights used'}")
    return model, has_attn


# ---------------------------------------------------------------------------
# Attention-based top-K patch selection
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_topk_indices(model, bag: torch.Tensor, k: int, has_attn: bool) -> np.ndarray:
    """
    Return indices of the top-k patches by attention weight.

    Handles output shapes:
      [B, N, 1]       : CLAM-family
      [B, n_out*N, 1] : clam_mil_mb (cat-on-dim1 bug → average over classes)
      [B, N]          : TransMIL, VarMIL, …
    Falls back to uniform random for models without calculate_attention.
    """
    n = bag.shape[0]
    k = min(k, n)

    if not has_attn:
        return np.random.choice(n, k, replace=False)

    attn = model.calculate_attention(bag.unsqueeze(0), apply_softmax=True)
    attn = attn.squeeze()
    if attn.ndim == 2:
        attn = attn.squeeze(-1)
    if attn.ndim == 1 and attn.shape[0] != n and attn.shape[0] % n == 0:
        attn = attn.view(attn.shape[0] // n, n).mean(dim=0)

    return attn.topk(k).indices.numpy()


# ---------------------------------------------------------------------------
# TFRecord image extraction
# ---------------------------------------------------------------------------

def build_tfrecord_index(tfrecord_dirs: list) -> dict:
    """slide_id → (tfrecord_path, index_array [N, 2]) with (offset, length)."""
    index = {}
    for d in tfrecord_dirs:
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            if not fname.endswith(".tfrecords"):
                continue
            slide_id = fname[: -len(".tfrecords")]
            idx_path = os.path.join(d, slide_id + ".index.npz")
            if os.path.exists(idx_path):
                index[slide_id] = (
                    os.path.join(d, fname),
                    np.load(idx_path)["arr_0"],
                )
    return index


def read_jpeg_from_tfrecord(tfr_path: str, byte_offset: int, record_length: int) -> bytes:
    """Extract JPEG bytes from a SlideFlow TFRecord record using byte offsets."""
    with open(tfr_path, "rb") as fh:
        fh.seek(byte_offset + 12)           # skip uint64 length + CRC
        payload = fh.read(record_length - 16)
    jpeg_start = payload.find(b"\xff\xd8")
    if jpeg_start == -1:
        raise ValueError(f"No JPEG marker at offset {byte_offset}")
    return payload[jpeg_start:]


def save_patch_image(jpeg_bytes: bytes, out_path: str) -> None:
    Image.open(io.BytesIO(jpeg_bytes)).convert("RGB").save(out_path, format="JPEG", quality=95)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Centroid distance analysis between datasets.")
    p.add_argument("--model_dir",     required=True, help="Model dir (mil_params.json + *.ckpt).")
    p.add_argument("--bags_dir",      required=True, help="Directory with .pt bag files.")
    p.add_argument("--annotations",   required=True, help="CSV: slide, dataset, category.")
    p.add_argument("--labels",        nargs="*", default=None,
                   help="Class labels as 'id:name', e.g. 0:Basal 1:Her2.")
    p.add_argument("--datasets",      nargs=2, default=["tcga", "cptac"])
    p.add_argument("--top_k_attn",    type=int, default=8,
                   help="Patches selected per WSI by attention weight (default: 8).")
    p.add_argument("--top_n_images",  type=int, default=25,
                   help="Patches closest to centroid to save as JPEG (default: 25).")
    p.add_argument("--output",        default=None, help="CSV output path.")
    p.add_argument("--patches_dir",   default=None,
                   help="Output directory for JPEG images. Requires --tfrecord_dirs.")
    p.add_argument("--tfrecord_dirs", nargs="*", default=None,
                   help="Directories with .tfrecords + .index.npz files.")
    return p.parse_args()


def parse_labels(label_args):
    if not label_args:
        return None
    return {int(k): v for item in label_args for k, v in [item.split(":")]}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    labels = parse_labels(args.labels)
    ds_a, ds_b = args.datasets

    save_images = args.patches_dir is not None
    if save_images and not args.tfrecord_dirs:
        raise ValueError("--tfrecord_dirs is required when --patches_dir is set.")

    aggregators_module = load_aggregators_module()
    model, has_attn = load_model(args.model_dir, aggregators_module)

    ann = pd.read_csv(args.annotations)
    ann = ann[ann["dataset"].isin([ds_a, ds_b])].copy()
    ann["slide_id"] = ann["slide"].apply(lambda s: os.path.splitext(s)[0])
    slide_info = ann.set_index("slide_id")[["dataset", "category"]].to_dict("index")
    print(f"\nAnnotations — {ds_a}: {(ann['dataset']==ds_a).sum()}, "
          f"{ds_b}: {(ann['dataset']==ds_b).sum()}")

    bag_files = sorted(f for f in os.listdir(args.bags_dir) if f.endswith(".pt"))
    print(f"Found {len(bag_files)} bags")

    tfr_index = {}
    if save_images:
        os.makedirs(args.patches_dir, exist_ok=True)
        tfr_index = build_tfrecord_index(args.tfrecord_dirs)
        print(f"TFRecord index: {len(tfr_index)} slides")

    # ------------------------------------------------------------------
    # Step 1: select top-K patches per WSI by attention weight
    # ------------------------------------------------------------------
    print(f"\nStep 1: selecting top-{args.top_k_attn} patches per WSI by attention...")

    # pool[(ds, cls)] = list of dicts: {emb, slide_id, patch_idx}
    pool: dict = {}
    processed, skipped = 0, 0

    for fname in bag_files:
        slide_id = os.path.splitext(fname)[0]
        if slide_id not in slide_info:
            skipped += 1
            continue
        info = slide_info[slide_id]
        ds, cls = info["dataset"], int(info["category"])

        bag = torch.load(os.path.join(args.bags_dir, fname), map_location="cpu")
        if bag.ndim == 1:
            bag = bag.unsqueeze(0)

        topk_idx = get_topk_indices(model, bag, args.top_k_attn, has_attn)  # [k]
        embs = bag.numpy()[topk_idx]                                          # [k, n_feats]

        key = (ds, cls)
        if key not in pool:
            pool[key] = {"embs": [], "slide_ids": [], "patch_idxs": []}
        pool[key]["embs"].append(embs)
        pool[key]["slide_ids"].extend([slide_id] * len(topk_idx))
        pool[key]["patch_idxs"].extend(topk_idx.tolist())
        processed += 1

    print(f"  Processed {processed} slides ({skipped} skipped).")

    # Consolidate into arrays
    for key in pool:
        pool[key]["embs"]       = np.concatenate(pool[key]["embs"], axis=0)  # [M, n_feats]
        pool[key]["slide_ids"]  = np.array(pool[key]["slide_ids"])
        pool[key]["patch_idxs"] = np.array(pool[key]["patch_idxs"])

    # ------------------------------------------------------------------
    # Step 2: compute class centroids
    # ------------------------------------------------------------------
    print("\nStep 2: computing class centroids from top-K patches...")
    centroids: dict = {}
    for key, data in pool.items():
        ds, cls = key
        name = labels[cls] if labels else str(cls)
        centroids[key] = data["embs"].mean(axis=0)
        print(f"  μ {ds}/{name}: {len(data['embs'])} patches")

    # ------------------------------------------------------------------
    # Step 3: select top-N patches globally closest to centroid per class
    # ------------------------------------------------------------------
    print(f"\nStep 3: selecting top-{args.top_n_images} patches closest to centroid...")

    selected: dict = {}   # (ds, cls) → indices into pool arrays
    for key, data in pool.items():
        centroid = centroids[key]
        dists    = np.linalg.norm(data["embs"] - centroid, axis=1)   # [M]
        n        = min(args.top_n_images, len(dists))
        selected[key] = np.argsort(dists)[:n]

    # ------------------------------------------------------------------
    # Step 4: save images and compute final centroid distances
    # ------------------------------------------------------------------
    class_ids = sorted({cls for _, cls in pool})
    rows = []

    print(f"\n{'Class':<12} {'n_'+ds_a:<12} {'n_'+ds_b:<12} {'Distance':>12}")
    print("-" * 52)

    for cls in class_ids:
        class_name = labels[cls] if labels else str(cls)

        pool_a = pool.get((ds_a, cls))
        pool_b = pool.get((ds_b, cls))
        if pool_a is None or pool_b is None:
            print(f"{class_name:<12} {'—':>11}  {'—':>11}  {'N/A':>12}")
            continue

        sel_a = selected[(ds_a, cls)]
        sel_b = selected[(ds_b, cls)]

        mu_a = pool_a["embs"][sel_a].mean(axis=0)
        mu_b = pool_b["embs"][sel_b].mean(axis=0)
        dist = float(np.linalg.norm(mu_a - mu_b))

        print(f"{class_name:<12} {len(sel_a):>11}  {len(sel_b):>11}  {dist:>12.4f}")
        rows.append({
            "class_id":          cls,
            "class_name":        class_name,
            f"n_{ds_a}":         len(sel_a),
            f"n_{ds_b}":         len(sel_b),
            "centroid_distance": dist,
        })

        # Save images
        if save_images:
            for ds, cur_pool, cur_sel in [(ds_a, pool_a, sel_a), (ds_b, pool_b, sel_b)]:
                out_subdir = os.path.join(args.patches_dir, ds, class_name)
                os.makedirs(out_subdir, exist_ok=True)
                for rank, idx in enumerate(cur_sel):
                    slide_id  = cur_pool["slide_ids"][idx]
                    patch_idx = int(cur_pool["patch_idxs"][idx])
                    if slide_id not in tfr_index:
                        continue
                    tfr_path, idx_data = tfr_index[slide_id]
                    try:
                        offset = int(idx_data[patch_idx, 0])
                        length = int(idx_data[patch_idx, 1])
                        jpeg   = read_jpeg_from_tfrecord(tfr_path, offset, length)
                        fname_img = f"{slide_id}_rank{rank:02d}.jpg"
                        save_patch_image(jpeg, os.path.join(out_subdir, fname_img))
                    except Exception as e:
                        print(f"  Warning: {slide_id} patch {patch_idx}: {e}")

    if args.output and rows:
        pd.DataFrame(rows).to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()