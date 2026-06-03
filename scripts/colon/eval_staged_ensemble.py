"""
Ensemble evaluation of staged MIL training on hold-out test datasets.
Loads all stage checkpoints, runs inference on each test dataset, and averages probabilities.
"""
import os
import glob
import logging
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score, confusion_matrix,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import slideflow as sf
from slideflow.mil import eval_mil, mil_config
from pathbench.models import aggregators
from pathbench.utils.utils import get_model_class


EXPERIMENT_DIR = "experiments/colon_mss_msi_staged"
BAGS_PATH      = "/home/PARADIS/colon/tfrecords/shared_bags/256_128_none_h_optimus_0"
ANNOTATIONS    = "./config/annotations/annotations_colon_mss_msi_all.csv"

TEST_DATASETS = {
    "tcga_coad":  {
        "slides":    "/home/PARADIS/colon/wsi/TCGA-COAD",
        "tfrecords": "/home/PARADIS/colon/tfrecords/tcga_coad/256px_20x",
    },
    "cptac_coad": {
        "slides":    "/home/PARADIS/colon/wsi/CPTAC-COAD",
        "tfrecords": "/home/PARADIS/colon/tfrecords/cptac_coad/256px_20x",
    },
}

MIL_PARAMS = dict(
    model=get_model_class(aggregators, "dsmil"),
    aggregation_level="patient",
    trainer="lightning",
    epochs=1,
    batch_size=32,
    bag_size=128,
    z_dim=270,
    encoder_layers=1,
    dropout_p=0.3966,
    activation_function="ReLU",
    slide_level=False,
    task="classification",
)


def get_stage_dirs(experiment_dir):
    mil_dir = os.path.join(experiment_dir, "mil")
    dirs = sorted(glob.glob(os.path.join(mil_dir, "*_stage*")))
    return dirs


def run_ensemble_eval(experiment_dir, test_datasets, bags_path, annotations):
    stage_dirs = get_stage_dirs(experiment_dir)
    logging.info(f"Found {len(stage_dirs)} stage checkpoints")

    mil_conf = mil_config(**MIL_PARAMS)
    model_string = f"<class 'pathbench.models.aggregators.dsmil'>"

    project = sf.Project(experiment_dir, annotations=annotations)
    for ds_name, ds_paths in test_datasets.items():
        project.add_source(
            name=ds_name,
            slides=ds_paths["slides"],
            tfrecords=ds_paths["tfrecords"],
        )

    for ds_name in test_datasets:
        logging.info(f"\n=== Ensemble evaluation on {ds_name} ===")
        dataset = project.dataset(tile_px=256, tile_um=128).filter(
            filters={"dataset": [ds_name]}
        )
        stage_preds = []

        for i, stage_dir in enumerate(stage_dirs):
            outdir = os.path.join(experiment_dir, f"staged_eval_{ds_name}_stage{i + 1:02d}")
            preds_path = os.path.join(outdir, f"00000-{model_string}", "predictions.parquet")

            if os.path.exists(preds_path):
                logging.info(f"  Stage {i+1}: loading cached predictions")
            else:
                logging.info(f"  Stage {i+1}: running inference from {stage_dir}")
                eval_mil(
                    weights=stage_dir,
                    outcomes="category",
                    dataset=dataset,
                    bags=bags_path,
                    config=mil_conf,
                    outdir=outdir,
                )

            if os.path.exists(preds_path):
                stage_preds.append(pd.read_parquet(preds_path))
            else:
                logging.warning(f"  Stage {i+1}: predictions not found, skipping")

        if not stage_preds:
            logging.warning(f"No predictions for {ds_name}, skipping.")
            continue

        y_pred_cols = [c for c in stage_preds[0].columns if "y_pred" in c]
        ensemble = stage_preds[0].copy()
        for col in y_pred_cols:
            ensemble[col] = np.mean([df[col].values for df in stage_preds], axis=0)

        ensemble_dir = os.path.join(experiment_dir, f"staged_eval_{ds_name}_ensemble")
        os.makedirs(ensemble_dir, exist_ok=True)
        ensemble.to_parquet(os.path.join(ensemble_dir, "predictions.parquet"), index=False)
        ensemble.to_csv(os.path.join(ensemble_dir, "predictions.csv"), index=False)

        y_true = ensemble["y_true"].values
        y_pred_prob = ensemble[y_pred_cols].values
        y_pred_class = np.argmax(y_pred_prob, axis=1)
        classes = sorted(np.unique(y_true))

        bal_acc  = balanced_accuracy_score(y_true, y_pred_class)
        precision = precision_score(y_true, y_pred_class, average=None, labels=classes, zero_division=0)
        recall    = recall_score(y_true, y_pred_class, average=None, labels=classes, zero_division=0)
        f1        = f1_score(y_true, y_pred_class, average=None, labels=classes, zero_division=0)
        cm        = confusion_matrix(y_true, y_pred_class, labels=classes)
        try:
            auc_per_class = [
                roc_auc_score((y_true == c).astype(int), y_pred_prob[:, i])
                for i, c in enumerate(classes)
            ]
        except Exception:
            auc_per_class = [float("nan")] * len(classes)

        logging.info(f"\n  === {ds_name} ===")
        logging.info(f"  Balanced Accuracy: {bal_acc:.4f}")
        logging.info(f"  {'Class':<8} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
        for i, c in enumerate(classes):
            logging.info(f"  {c:<8} {precision[i]:>10.4f} {recall[i]:>10.4f} {f1[i]:>10.4f} {auc_per_class[i]:>10.4f}")
        logging.info(f"  Confusion matrix (rows=true, cols=pred, classes={classes}):\n{cm}")

        metrics_df = pd.DataFrame({
            "class":     [str(c) for c in classes] + ["balanced_accuracy"],
            "precision": list(precision) + [float("nan")],
            "recall":    list(recall) + [bal_acc],
            "f1":        list(f1) + [float("nan")],
            "auc":       auc_per_class + [np.nanmean(auc_per_class)],
        })
        metrics_df.to_csv(os.path.join(ensemble_dir, "metrics.csv"), index=False)
        logging.info(f"  Results saved to {ensemble_dir}/")


def find_best_stage(stage_dirs, save_string):
    best_metric = -np.inf
    best_dir = None
    for i, stage_dir in enumerate(stage_dirs):
        preds_path = os.path.join(stage_dir, "predictions.parquet")
        if not os.path.exists(preds_path):
            continue
        df = pd.read_parquet(preds_path)
        y_true = df["y_true"].values
        y_pred_cols = [c for c in df.columns if "y_pred" in c]
        y_pred_class = np.argmax(df[y_pred_cols].values, axis=1)
        metric = balanced_accuracy_score(y_true, y_pred_class)
        logging.info(f"  Stage {i+1}: val balanced_accuracy = {metric:.4f}")
        if metric > best_metric:
            best_metric = metric
            best_dir = stage_dir
    logging.info(f"  Best stage: {best_dir} (val bal_acc={best_metric:.4f})")
    return best_dir


def run_single_eval(experiment_dir, test_datasets, bags_path, annotations, weights_dir, tag="best"):
    mil_conf = mil_config(**MIL_PARAMS)
    model_string = f"<class 'pathbench.models.aggregators.dsmil'>"

    project = sf.Project(experiment_dir, annotations=annotations)
    for ds_name, ds_paths in test_datasets.items():
        project.add_source(
            name=ds_name,
            slides=ds_paths["slides"],
            tfrecords=ds_paths["tfrecords"],
        )

    for ds_name in test_datasets:
        logging.info(f"\n=== {tag} model evaluation on {ds_name} ===")
        dataset = project.dataset(tile_px=256, tile_um=128).filter(
            filters={"dataset": [ds_name]}
        )
        outdir = os.path.join(experiment_dir, f"staged_eval_{ds_name}_{tag}")
        preds_path = os.path.join(outdir, f"00000-{model_string}", "predictions.parquet")

        if os.path.exists(preds_path):
            logging.info("  Loading cached predictions")
        else:
            eval_mil(
                weights=weights_dir,
                outcomes="category",
                dataset=dataset,
                bags=bags_path,
                config=mil_conf,
                outdir=outdir,
            )

        if not os.path.exists(preds_path):
            logging.warning(f"  Predictions not found for {ds_name}, skipping.")
            continue

        result = pd.read_parquet(preds_path)
        result.to_csv(preds_path.replace(".parquet", ".csv"), index=False)

        y_true = result["y_true"].values
        y_pred_cols = [c for c in result.columns if "y_pred" in c]
        y_pred_prob = result[y_pred_cols].values
        y_pred_class = np.argmax(y_pred_prob, axis=1)
        classes = sorted(np.unique(y_true))

        bal_acc  = balanced_accuracy_score(y_true, y_pred_class)
        precision_ = precision_score(y_true, y_pred_class, average=None, labels=classes, zero_division=0)
        recall_    = recall_score(y_true, y_pred_class, average=None, labels=classes, zero_division=0)
        f1_        = f1_score(y_true, y_pred_class, average=None, labels=classes, zero_division=0)
        cm         = confusion_matrix(y_true, y_pred_class, labels=classes)
        try:
            auc_per_class = [roc_auc_score((y_true == c).astype(int), y_pred_prob[:, i]) for i, c in enumerate(classes)]
        except Exception:
            auc_per_class = [float("nan")] * len(classes)

        logging.info(f"  Balanced Accuracy: {bal_acc:.4f}")
        logging.info(f"  {'Class':<8} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
        for i, c in enumerate(classes):
            logging.info(f"  {c:<8} {precision_[i]:>10.4f} {recall_[i]:>10.4f} {f1_[i]:>10.4f} {auc_per_class[i]:>10.4f}")
        logging.info(f"  Confusion matrix:\n{cm}")

        metrics_df = pd.DataFrame({
            "class":     [str(c) for c in classes] + ["balanced_accuracy"],
            "precision": list(precision_) + [float("nan")],
            "recall":    list(recall_) + [bal_acc],
            "f1":        list(f1_) + [float("nan")],
            "auc":       auc_per_class + [np.nanmean(auc_per_class)],
        })
        result_dir = os.path.join(experiment_dir, f"staged_eval_{ds_name}_{tag}")
        os.makedirs(result_dir, exist_ok=True)
        metrics_df.to_csv(os.path.join(result_dir, "metrics.csv"), index=False)
        logging.info(f"  Saved to {result_dir}/metrics.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", default=EXPERIMENT_DIR)
    parser.add_argument("--bags_path", default=BAGS_PATH)
    parser.add_argument("--annotations", default=ANNOTATIONS)
    parser.add_argument("--mode", choices=["ensemble", "best", "last"], default="ensemble",
                        help="ensemble: average all stages | best: stage with best val metric | last: final stage")
    args = parser.parse_args()

    stage_dirs = get_stage_dirs(args.experiment_dir)

    if args.mode == "ensemble":
        run_ensemble_eval(args.experiment_dir, TEST_DATASETS, args.bags_path, args.annotations)
    elif args.mode == "best":
        best_dir = find_best_stage(stage_dirs, "")
        run_single_eval(args.experiment_dir, TEST_DATASETS, args.bags_path, args.annotations, best_dir, tag="best")
    elif args.mode == "last":
        run_single_eval(args.experiment_dir, TEST_DATASETS, args.bags_path, args.annotations, stage_dirs[-1], tag="last")