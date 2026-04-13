#!/usr/bin/env python
"""
Evaluates trained MMR gene models (mlh1, msh2, msh6, pms2) on:
  - macarena  : uses held-out fold predictions (val_result.csv per fold)
  - cptac_coad: runs inference with each fold model, averages predictions
  - tcga_coad : runs inference with each fold model, averages predictions

Usage:
    python scripts/colon/eval_mmr_models.py [--gene mlh1]

If --gene is not specified, evaluates all 4 genes.
"""

import argparse
import importlib.util
import json
import logging
import os
import glob
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, classification_report, average_precision_score

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GENE_CONFIGS = {
    'mlh1': {
        'exp_dir': 'experiments/colon_mlh1_optuna/mil',
        'fold_dirs': [
            '00093-256_128_none_h_optimus_0_clam_mil_mb_CrossEntropyLoss_None_ReLU_Adam_1',
            '00094-256_128_none_h_optimus_0_clam_mil_mb_CrossEntropyLoss_None_ReLU_Adam_2',
            '00095-256_128_none_h_optimus_0_clam_mil_mb_CrossEntropyLoss_None_ReLU_Adam_3',
            '00096-256_128_none_h_optimus_0_clam_mil_mb_CrossEntropyLoss_None_ReLU_Adam_4',
            '00097-256_128_none_h_optimus_0_clam_mil_mb_CrossEntropyLoss_None_ReLU_Adam_5',
        ],
        'norm': 'none',
        'annotations': 'config/annotations/annotations_colon_mlh1_all.csv',
    },
    'msh2': {
        'exp_dir': 'experiments/colon_mlh2_optuna/mil',
        'fold_dirs': [
            '00118-256_128_none_h_optimus_0_clam_mil_mb_CrossEntropyLoss_None_ReLU_Adam_1',
            '00119-256_128_none_h_optimus_0_clam_mil_mb_CrossEntropyLoss_None_ReLU_Adam_2',
            '00120-256_128_none_h_optimus_0_clam_mil_mb_CrossEntropyLoss_None_ReLU_Adam_3',
            '00121-256_128_none_h_optimus_0_clam_mil_mb_CrossEntropyLoss_None_ReLU_Adam_4',
            '00122-256_128_none_h_optimus_0_clam_mil_mb_CrossEntropyLoss_None_ReLU_Adam_5',
        ],
        'norm': 'none',
        'annotations': 'config/annotations/annotations_colon_msh2_all.csv',
    },
    'msh6': {
        'exp_dir': 'experiments/colon_msh6_optuna/mil',
        'fold_dirs': [
            '00073-256_128_none_h_optimus_0_transmil_CrossEntropyLoss_None_ReLU_Adam_1',
            '00074-256_128_none_h_optimus_0_transmil_CrossEntropyLoss_None_ReLU_Adam_2',
            '00075-256_128_none_h_optimus_0_transmil_CrossEntropyLoss_None_ReLU_Adam_3',
            '00076-256_128_none_h_optimus_0_transmil_CrossEntropyLoss_None_ReLU_Adam_4',
            '00077-256_128_none_h_optimus_0_transmil_CrossEntropyLoss_None_ReLU_Adam_5',
        ],
        'norm': 'none',
        'annotations': 'config/annotations/annotations_colon_msh6_all.csv',
    },
    'pms2': {
        'exp_dir': 'experiments/colon_pms2_optuna/mil',
        'fold_dirs': [
            '00198-256_128_macenko_h_optimus_0_transmil_CrossEntropyLoss_None_ReLU_Adam_1',
            '00199-256_128_macenko_h_optimus_0_transmil_CrossEntropyLoss_None_ReLU_Adam_2',
            '00200-256_128_macenko_h_optimus_0_transmil_CrossEntropyLoss_None_ReLU_Adam_3',
            '00201-256_128_macenko_h_optimus_0_transmil_CrossEntropyLoss_None_ReLU_Adam_4',
            '00202-256_128_macenko_h_optimus_0_transmil_CrossEntropyLoss_None_ReLU_Adam_5',
        ],
        'norm': 'macenko',
        'annotations': 'config/annotations/annotations_colon_pms2_all.csv',
    },
}

BAGS_BASE = '/home/PARADIS/colon/tfrecords/shared_bags'

_SLIDEFLOW_MIL_FILES = {
    'transmil': os.path.join(ROOT, 'slideflow_fork', 'slideflow', 'mil', 'models', 'transmil.py'),
}


def _load_module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_model_cls(model_name: str):
    key = model_name.lower()
    if key in _SLIDEFLOW_MIL_FILES:
        mod = _load_module_from_file(f'sf_mil_{key}', _SLIDEFLOW_MIL_FILES[key])
        class_name = {'transmil': 'TransMIL'}.get(key, model_name)
        return getattr(mod, class_name)
    agg = _load_module_from_file('aggregators',
                                 os.path.join(ROOT, 'pathbench', 'models', 'aggregators.py'))
    return getattr(agg, model_name)


def load_model(model_dir: str):
    with open(os.path.join(model_dir, 'mil_params.json')) as f:
        mil_params = json.load(f)
    params = mil_params['params']
    model_cls = get_model_cls(params['model'])
    model = model_cls(
        n_feats=mil_params['input_shape'],
        n_out=mil_params['output_shape'],
        z_dim=params.get('z_dim', 256),
        dropout_p=params.get('dropout_p', 0.1),
        activation_function=params.get('activation_function', 'ReLU'),
        encoder_layers=params.get('encoder_layers', 1),
    )
    pattern = os.path.join(model_dir, 'checkpoints', '**', 'best-epoch*.ckpt')
    ckpts = glob.glob(pattern, recursive=True)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint in {model_dir}")
    ckpt_path = sorted(ckpts)[-1]
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = {k.removeprefix('model.'): v for k, v in ckpt['state_dict'].items() if k.startswith('model.')}
    model.load_state_dict(state_dict, strict=True)
    return model, mil_params


def predict_bags(model, bags_dir: str, device: torch.device) -> pd.DataFrame:
    rows = []
    for fname in os.listdir(bags_dir):
        if not fname.endswith('.pt'):
            continue
        slide = os.path.splitext(fname)[0]
        feats = torch.load(os.path.join(bags_dir, fname), map_location=device).float()
        if feats.dim() == 1:
            feats = feats.unsqueeze(0)
        with torch.no_grad():
            logits = model(feats.unsqueeze(0))
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        logits = logits.squeeze()
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        rows.append({'slide': slide, 'y_pred0': float(probs[0]), 'y_pred1': float(probs[1])})
    return pd.DataFrame(rows)


def print_metrics(df: pd.DataFrame, dataset_name: str):
    y_true = df['y_true'].astype(int).values
    y_pred = (df['y_pred1'] >= 0.5).astype(int).values

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_norm = (cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)).round(3)

    acc = (y_true == y_pred).mean()
    try:
        ap1 = average_precision_score((y_true == 1).astype(int), df['y_pred1'].values)
        ap0 = average_precision_score((y_true == 0).astype(int), df['y_pred0'].values)
        map_val = (ap0 + ap1) / 2
    except Exception:
        ap0 = ap1 = map_val = float('nan')

    print(f"\n{'='*55}")
    print(f"  {dataset_name}  |  slides: {len(df)}  |  acc: {acc:.3f}  |  MAP: {map_val:.3f}")
    print(f"  pos: {(y_true==1).sum()}  neg: {(y_true==0).sum()}")
    print(f"{'='*55}")
    print("Confusion Matrix (filas=real, columnas=predicho):")
    print(f"        neg  pos")
    print(f"  neg   {cm[0,0]:4d} {cm[0,1]:4d}")
    print(f"  pos   {cm[1,0]:4d} {cm[1,1]:4d}")
    print("\nNormalizada:")
    print(f"        neg   pos")
    print(f"  neg   {cm_norm[0,0]:.3f} {cm_norm[0,1]:.3f}")
    print(f"  pos   {cm_norm[1,0]:.3f} {cm_norm[1,1]:.3f}")
    print(f"\nAP neg: {ap0:.3f}  |  AP pos: {ap1:.3f}  |  MAP: {map_val:.3f}")
    print(classification_report(y_true, y_pred, labels=[0, 1],
                                target_names=['neg', 'pos'], digits=3, zero_division=0))


def evaluate_gene(gene: str):
    cfg = GENE_CONFIGS[gene]
    annot = pd.read_csv(os.path.join(ROOT, cfg['annotations']))
    bags_subdir = f'256_128_{cfg["norm"]}_h_optimus_0'
    bags_base = os.path.join(BAGS_BASE, bags_subdir)

    print(f"\n{'#'*60}")
    print(f"# GENE: {gene.upper()}")
    print(f"{'#'*60}")

    # ── 1. MACARENA: aggregate val_result.csv from all folds ──────────────
    mac_preds = []
    for fold_dir in cfg['fold_dirs']:
        vr = os.path.join(ROOT, cfg['exp_dir'], fold_dir, 'val_result.csv')
        if os.path.exists(vr):
            mac_preds.append(pd.read_csv(vr))
    if mac_preds:
        mac_df = pd.concat(mac_preds, ignore_index=True)
        # val_result.csv already has y_true — just add dataset column
        mac_df = mac_df.merge(annot[['slide', 'dataset']], on='slide', how='left')
        print_metrics(mac_df, 'MACARENA (5-fold val)')
    else:
        print("  [macarena] No val_result.csv found")

    # ── 2. MACARENA + CPTAC + TCGA: ensemble inference with all fold models ─
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for ds_name, ds_filter in [('macarena_inference', 'macarena'),
                                ('cptac_coad', 'cptac_coad'), ('tcga_coad', 'tcga_coad')]:
        ds_annot = annot[annot['dataset'] == ds_filter].copy()
        if ds_annot.empty:
            print(f"  [{ds_name}] No slides in annotations")
            continue

        all_preds = []
        for fold_dir in cfg['fold_dirs']:
            model_dir = os.path.join(ROOT, cfg['exp_dir'], fold_dir)
            try:
                model, _ = load_model(model_dir)
                model = model.to(device).eval()
                fold_preds = predict_bags(model, bags_base, device)
                all_preds.append(fold_preds)
                del model
                torch.cuda.empty_cache()
            except Exception as e:
                logging.warning(f"  Fold {fold_dir}: {e}")

        if not all_preds:
            print(f"  [{ds_name}] No predictions generated")
            continue

        # Ensemble: average softmax probabilities
        merged = all_preds[0][['slide']].copy()
        merged['y_pred0'] = np.mean([p['y_pred0'].values for p in all_preds], axis=0) if len(all_preds) > 1 else all_preds[0]['y_pred0'].values
        merged['y_pred1'] = np.mean([p['y_pred1'].values for p in all_preds], axis=0) if len(all_preds) > 1 else all_preds[0]['y_pred1'].values

        # Align on common slides, only using ds_annot slides
        result = merged.merge(ds_annot[['slide', 'category']], on='slide', how='inner')
        result = result.rename(columns={'category': 'y_true'})

        if result.empty:
            print(f"  [{ds_name}] No overlap between bags and annotations")
            continue

        print_metrics(result, ds_name.upper())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gene', choices=list(GENE_CONFIGS.keys()), default=None,
                        help="Gene to evaluate (default: all)")
    args = parser.parse_args()

    os.chdir(ROOT)

    genes = [args.gene] if args.gene else list(GENE_CONFIGS.keys())
    for gene in genes:
        evaluate_gene(gene)


if __name__ == '__main__':
    main()