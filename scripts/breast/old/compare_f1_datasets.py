"""
Compare F1 scores between datasets (e.g. TCGA vs CPTAC) for one or more
prediction CSV files, and generate confusion matrix images.

Usage:
    python scripts/compare_f1_datasets.py \
        --annotations config/annotations/annotations_brca_pam50.csv \
        --predictions predictions_per_slide.csv predictions_per_slide_macenko_imbalance.csv \
        --labels 0:Basal 1:Her2 2:LumA 3:LumB 4:Normal \
        --datasets tcga cptac \
        --figures_dir figures/
"""

import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score


# Rose/pink colormap: white → deep rose
ROSE_CMAP = plt.cm.colors.LinearSegmentedColormap.from_list(
    "rose",
    ["#ffffff", "#fce4ec", "#f48fb1", "#e91e8c", "#880e4f"],
)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare F1 per dataset across prediction files.")
    parser.add_argument("--annotations", required=True,
                        help="Path to annotations CSV (must have 'slide', 'dataset', 'category' columns).")
    parser.add_argument("--predictions", nargs="+", required=True,
                        help="One or more prediction CSV files (must have 'slide' and 'y_pred_class' columns).")
    parser.add_argument("--labels", nargs="*", default=None,
                        help="Class labels in 'id:name' format, e.g. 0:Basal 1:Her2. "
                             "If omitted, numeric class IDs are used.")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Datasets to compare. If omitted, all datasets in annotations are used.")
    parser.add_argument("--output", default=None,
                        help="Optional path to save results as CSV.")
    parser.add_argument("--figures_dir", default=None,
                        help="Directory to save confusion matrix images. If omitted, figures are not saved.")
    return parser.parse_args()


def parse_labels(label_args):
    if not label_args:
        return None
    labels = {}
    for item in label_args:
        idx, name = item.split(":")
        labels[int(idx)] = name
    return labels


def compute_f1(df, class_ids, labels):
    macro = f1_score(df["category"], df["y_pred_class"], average="macro", labels=class_ids, zero_division=0)
    per_class = f1_score(df["category"], df["y_pred_class"], average=None, labels=class_ids, zero_division=0)
    result = {"macro_f1": macro}
    for i, cid in enumerate(class_ids):
        name = labels[cid] if labels else str(cid)
        result[name] = per_class[i]
    return result


def plot_confusion_matrix(y_true, y_pred, class_ids, labels, title, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=class_ids)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    tick_labels = [labels[c] if labels else str(c) for c in class_ids]
    n = len(class_ids)

    fig, ax = plt.subplots(figsize=(n + 2, n + 1.5))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=ROSE_CMAP, vmin=0, vmax=1)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Proportion", fontsize=10)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(tick_labels, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(title, fontsize=12, pad=12)

    # Annotate cells with count and proportion
    for i in range(n):
        for j in range(n):
            color = "white" if cm_norm[i, j] > 0.55 else "#3d0020"
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.2f})",
                    ha="center", va="center", fontsize=8, color=color)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def safe_filename(text):
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", text)


def main():
    args = parse_args()
    labels = parse_labels(args.labels)

    ann = pd.read_csv(args.annotations)
    datasets = args.datasets if args.datasets else sorted(ann["dataset"].unique())

    if args.figures_dir:
        os.makedirs(args.figures_dir, exist_ok=True)

    rows = []

    for pred_path in args.predictions:
        pred = pd.read_csv(pred_path)
        merged = pred.merge(ann[["slide", "dataset", "category"]], on="slide", how="inner")

        class_ids = sorted(merged["category"].unique())
        if labels:
            class_ids = sorted(labels.keys())

        pred_tag = safe_filename(os.path.splitext(os.path.basename(pred_path))[0])

        print(f"\n{'='*60}")
        print(f"File: {pred_path}  (n={len(merged)})")
        print(f"{'='*60}")

        per_ds = {}
        for ds in datasets:
            sub = merged[merged["dataset"] == ds]
            if sub.empty:
                print(f"  {ds}: no data")
                continue
            metrics = compute_f1(sub, class_ids, labels)
            per_ds[ds] = metrics

            print(f"\n  {ds} (n={len(sub)}):")
            print(f"    macro F1 = {metrics['macro_f1']:.4f}")
            for key, val in metrics.items():
                if key != "macro_f1":
                    print(f"    {key}: {val:.4f}")

            row = {"file": pred_path, "dataset": ds, **metrics}
            rows.append(row)

            if args.figures_dir:
                title = f"{pred_tag} — {ds.upper()}  (macro F1={metrics['macro_f1']:.3f})"
                fname = f"cm_{pred_tag}_{safe_filename(ds)}.png"
                plot_confusion_matrix(
                    sub["category"], sub["y_pred_class"],
                    class_ids, labels,
                    title=title,
                    save_path=os.path.join(args.figures_dir, fname),
                )

        # Delta between first two datasets
        if len(datasets) >= 2:
            ds0, ds1 = datasets[0], datasets[1]
            if ds0 in per_ds and ds1 in per_ds:
                print(f"\n  Delta ({ds0} - {ds1}):")
                for key in per_ds[ds0]:
                    delta = per_ds[ds0][key] - per_ds[ds1][key]
                    print(f"    {key}: {delta:+.4f}")

    if args.output and rows:
        out_df = pd.DataFrame(rows)
        out_df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()