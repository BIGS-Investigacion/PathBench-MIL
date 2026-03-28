#!/usr/bin/env python3
"""
scripts/biological_separability.py

Computes two biological separability matrices per task group from pathologist
morphological annotations (mean of two raters):

  B_{c,d}^{TT}  (TCGA × TCGA):   Σ|r| of Mann-Whitney features that survive
                                    BH correction when comparing TCGA class c
                                    against TCGA class d.

  B_{c,d}^{TC}  (TCGA × CPTAC):  Same but comparing TCGA class c against
                                    CPTAC class d.

From the TCGA×CPTAC matrix, derives:
  B̃_c = min_{d≠c} B_{c,d}^{TC} − B_{c,c}^{TC}

Input:
  results/representative_images_annotation_mean_input.xlsx
    Sheet 'TCGA'  — 275 patches, columns: ETIQUETA + 6 feature columns
    Sheet 'CPTAC' — 275 patches, same structure

Output (per task):
  results/bio_tcga_tcga_{task}_matriz.csv
  results/bio_tcga_cptac_{task}_matriz.csv

Summary:
  results/bio_b_tilde.csv   — B̃_c per (task, class)

Usage:
  python scripts/biological_separability.py [--input PATH] [--output_dir DIR] [--alpha FLOAT]
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# ── Constants ─────────────────────────────────────────────────────────────────

FEATURES = [
    "ESTRUCTURA GLANDULAR",
    "ATIPIA NUCLEAR",
    "MITOSIS",
    "NECROSIS",
    "INFILTRADO_LI",
    "INFILTRADO_PMN",
]

# Task groups — labels must match after uppercasing the ETIQUETA column
TASK_GROUPS = {
    "pam50": ["BASAL", "HER2-ENRICHED", "LUMINAL-A", "LUMINAL-B", "NORMAL-LIKE"],
    "er":    ["ER-NEGATIVE", "ER-POSITIVE"],
    "pr":    ["PR-NEGATIVE", "PR-POSITIVE"],
    "erbb2": ["HER2-NEGATIVE", "HER2-POSITIVE"],
}


# ── Statistical utilities ─────────────────────────────────────────────────────

def bh_reject(pvals: list, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg procedure. Returns boolean array (True = rejected)."""
    pvals = np.array(pvals, dtype=float)
    m = len(pvals)
    if m == 0:
        return np.array([], dtype=bool)
    idx = np.argsort(pvals)
    thresholds = (np.arange(1, m + 1) / m) * alpha
    rejected = pvals[idx] <= thresholds
    result = np.zeros(m, dtype=bool)
    if rejected.any():
        last = np.where(rejected)[0][-1]
        result[idx[:last + 1]] = True
    return result


def rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    """Rank-biserial correlation (effect size for Mann-Whitney U)."""
    u, _ = stats.mannwhitneyu(x, y, alternative="two-sided")
    return float(1 - (2 * u) / (len(x) * len(y)))


def compare_groups(df_c: pd.DataFrame, df_d: pd.DataFrame,
                   alpha: float = 0.05) -> float:
    """
    Compare class c against class d across all features.
    Returns B_{c,d} = Σ|r| for features that survive BH correction.
    Returns 0.0 if fewer than 3 samples in either group.
    """
    pvals = []
    valid_feats = []

    for feat in FEATURES:
        x = df_c[feat].dropna().values
        y = df_d[feat].dropna().values
        if len(x) < 3 or len(y) < 3:
            continue
        try:
            _, p = stats.mannwhitneyu(x, y, alternative="two-sided")
        except ValueError:
            continue
        pvals.append(p)
        valid_feats.append(feat)

    if not pvals:
        return 0.0

    rejected = bh_reject(pvals, alpha)
    b = 0.0
    for feat, p, rej in zip(valid_feats, pvals, rejected):
        if rej:
            x = df_c[feat].dropna().values
            y = df_d[feat].dropna().values
            b += abs(rank_biserial(x, y))

    return round(b, 4)


# ── Matrix builders ───────────────────────────────────────────────────────────

def build_matrix(df_rows: pd.DataFrame, df_cols: pd.DataFrame,
                 row_labels: list, col_labels: list,
                 alpha: float = 0.05) -> pd.DataFrame:
    """
    Build a |row_labels| × |col_labels| matrix where entry (c, d) = B_{c,d}.
    df_rows and df_cols have already been uppercased on ETIQUETA.
    """
    data = {}
    for d in col_labels:
        group_d = df_cols[df_cols["ETIQUETA"] == d]
        col = []
        for c in row_labels:
            group_c = df_rows[df_rows["ETIQUETA"] == c]
            if group_c.empty or group_d.empty:
                col.append(float("nan"))
            else:
                col.append(compare_groups(group_c, group_d, alpha))
        data[d] = col

    return pd.DataFrame(data, index=row_labels)


# ── B̃_c derivation ───────────────────────────────────────────────────────────

def compute_b_tilde(matrix_tc: pd.DataFrame) -> pd.Series:
    """
    Given the TCGA×CPTAC matrix, compute B̃_c for each row class c:
      B̃_c = min_{d ≠ c} B_{c,d} − B_{c,c}
    The diagonal B_{c,c} corresponds to same-class TCGA vs CPTAC.
    If the diagonal label is missing from columns, B_{c,c} = NaN.
    """
    b_tilde = {}
    for c in matrix_tc.index:
        row = matrix_tc.loc[c]
        b_cc = row.get(c, float("nan"))
        off_diag = row.drop(labels=[c], errors="ignore")
        if off_diag.empty:
            b_tilde[c] = float("nan")
        else:
            b_tilde[c] = round(float(off_diag.min()) - float(b_cc), 4)
    return pd.Series(b_tilde, name="b_tilde")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(input_path: Path, output_dir: Path, alpha: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading annotations from {input_path}")
    tcga  = pd.read_excel(input_path, sheet_name="TCGA")
    cptac = pd.read_excel(input_path, sheet_name="CPTAC")

    for df in (tcga, cptac):
        df.columns = df.columns.str.strip()
        df["ETIQUETA"] = df["ETIQUETA"].str.strip().str.upper()

    print(f"[INFO] TCGA labels:  {sorted(tcga['ETIQUETA'].unique())}")
    print(f"[INFO] CPTAC labels: {sorted(cptac['ETIQUETA'].unique())}")

    b_tilde_rows = []

    for task, labels in TASK_GROUPS.items():
        # Filter to labels present in both cohorts
        row_labels = [l for l in labels if l in tcga["ETIQUETA"].values]
        col_labels = [l for l in labels if l in cptac["ETIQUETA"].values]

        if not row_labels or not col_labels:
            print(f"[WARN] {task}: no labels found, skipping")
            continue

        print(f"\n[INFO] Task: {task}  |  labels: {row_labels}")

        # ── TCGA × TCGA ──────────────────────────────────────────────────────
        mat_tt = build_matrix(tcga, tcga, row_labels, row_labels, alpha)
        out_tt = output_dir / f"bio_tcga_tcga_{task}_matriz.csv"
        mat_tt.to_csv(out_tt)
        print(f"  TCGA×TCGA saved → {out_tt}")
        print(mat_tt.to_string())

        # ── TCGA × CPTAC ─────────────────────────────────────────────────────
        mat_tc = build_matrix(tcga, cptac, row_labels, col_labels, alpha)
        out_tc = output_dir / f"bio_tcga_cptac_{task}_matriz.csv"
        mat_tc.to_csv(out_tc)
        print(f"  TCGA×CPTAC saved → {out_tc}")
        print(mat_tc.to_string())

        # ── B̃_c ──────────────────────────────────────────────────────────────
        b_tilde = compute_b_tilde(mat_tc)
        for cls, val in b_tilde.items():
            b_tilde_rows.append({"task": task, "cls": cls, "b_tilde": val})
        print(f"  B̃_c: {b_tilde.to_dict()}")

    # Summary CSV
    if b_tilde_rows:
        df_bt = pd.DataFrame(b_tilde_rows)
        out_bt = output_dir / "bio_b_tilde.csv"
        df_bt.to_csv(out_bt, index=False)
        print(f"\n[INFO] B̃_c summary saved → {out_bt}")
        print(df_bt.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="results/representative_images_annotation_mean_input.xlsx",
                        help="Path to the mean annotations Excel file")
    parser.add_argument("--output_dir", default="results",
                        help="Directory for output CSV files (default: results)")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="BH significance threshold (default: 0.05)")
    args = parser.parse_args()

    main(Path(args.input), Path(args.output_dir), args.alpha)