#!/usr/bin/env python3
"""
eval_coral_mil_inference.py

Inferencia MIL con CORAL condicional:
  - PCA(128) + CORAL ajustados sobre slide means de Macarena
  - Para cada slide de CPTAC/TCGA, aplica el transform patch a patch
    según la clase del slide (condicional) o sin clase (estándar)
  - PCA inverso devuelve los patches a 1536 dims para el modelo DSMIL

Modos:
  --mode none      : sin CORAL (baseline)
  --mode standard  : CORAL estándar (sin etiquetas del test)
  --mode oracle    : CORAL condicional con todas las etiquetas (límite superior)
  --mode fewshot   : CORAL condicional con N etiquetas por clase (--fewshot_n N)

Uso:
  python eval_coral_mil_inference.py --mode oracle
  python eval_coral_mil_inference.py --mode fewshot --fewshot_n 10 --fewshot_seeds 5
"""

import warnings
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import importlib.util
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

warnings.filterwarnings('ignore')

ROOT      = Path(__file__).resolve().parents[2]
BAGS_DIR  = Path('/home/PARADIS/colon/tfrecords/shared_bags/256_128_none_h_optimus_0')
MODEL_DIR = ROOT / 'experiments/colon_mss_msi_benchmark/mil/00000-256_128_none_h_optimus_0_dsmil_CrossEntropyLoss_ReLU_Adam_1'
ANNOT     = ROOT / 'config/annotations/annotations_colon_mss_msi_all.csv'
OUT_DIR   = ROOT / 'results/colon/coral_mil'

DATASET_LABELS = {'macarena': 'Macarena', 'cptac_coad': 'CPTAC-COAD', 'tcga_coad': 'TCGA-COAD'}


# ── Carga ─────────────────────────────────────────────────────────────────────

def load_slide_means(ann):
    vecs, rows = [], []
    for _, row in ann.iterrows():
        p = BAGS_DIR / f'{row["slide"]}.pt'
        if not p.exists():
            continue
        feat = torch.load(p, map_location='cpu', weights_only=True).float().numpy()
        vecs.append(feat.mean(axis=0))
        rows.append({'slide': row['slide'], 'dataset': row['dataset'],
                     'category': int(row['category'])})
    return np.stack(vecs).astype(np.float32), pd.DataFrame(rows)


# ── CORAL ─────────────────────────────────────────────────────────────────────

def fit_coral_transform(X_src, X_tgt, alpha):
    """Calcula W tal que (X_src - mu_s) @ W + mu_t ~ distribución de X_tgt."""
    mu_s = X_src.mean(0);  mu_t = X_tgt.mean(0)
    d = X_src.shape[1]
    Cs = np.cov((X_src - mu_s).T) + alpha * np.eye(d)
    Ct = np.cov((X_tgt - mu_t).T) + alpha * np.eye(d)
    Us, Ss, _ = np.linalg.svd(Cs);  Ut, St, _ = np.linalg.svd(Ct)
    W = Us @ np.diag(1/np.sqrt(np.maximum(Ss, 1e-10))) @ Us.T @ \
        Ut @ np.diag(np.sqrt(np.maximum(St, 0))) @ Ut.T
    return mu_s, mu_t, W


def apply_transform(X, mu_s, mu_t, W):
    return (X - mu_s) @ W + mu_t


def fit_conditional_transforms(X_pca, meta, ds, slide_labels, alpha):
    """
    Ajusta un transform CORAL por clase para el dominio ds,
    usando slide_labels como etiquetas (true o pseudo).
    Devuelve dict {cls_id: (mu_s, mu_t, W)} o None si pocos samples.
    """
    transforms = {}
    d = X_pca.shape[1]
    mask_ds = (meta['dataset'] == ds).values

    for cls_id in [0, 1]:
        mask_src = mask_ds & (slide_labels == cls_id)
        mask_tgt = (meta['dataset'] == 'macarena').values & \
                   (meta['category'] == cls_id).values
        n_s, n_t = mask_src.sum(), mask_tgt.sum()
        if n_s < 3 or n_t < 3:
            transforms[cls_id] = None
            continue
        alpha_eff = alpha * max(1.0, d / min(n_s, n_t))
        transforms[cls_id] = fit_coral_transform(
            X_pca[mask_src], X_pca[mask_tgt], alpha_eff)
    return transforms


# ── Modelo MIL ────────────────────────────────────────────────────────────────

def load_model(device):
    spec = importlib.util.spec_from_file_location(
        'aggregators', ROOT / 'pathbench/models/aggregators.py')
    agg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agg)

    with open(MODEL_DIR / 'mil_params.json') as f:
        mp = json.load(f)
    params = mp['params']
    model = getattr(agg, params['model'])(
        n_feats=mp['input_shape'], n_out=mp['output_shape'],
        z_dim=params.get('z_dim', 256), dropout_p=params.get('dropout_p', 0.1),
        activation_function=params.get('activation_function', 'ReLU'),
        encoder_layers=params.get('encoder_layers', 1),
    )
    ckpt  = sorted(MODEL_DIR.glob('checkpoints/**/best-epoch*.ckpt'))[-1]
    state = torch.load(ckpt, map_location='cpu')
    sd    = {k.removeprefix('model.'): v
             for k, v in state['state_dict'].items() if k.startswith('model.')}
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()


def predict_bag(model, bag, device):
    with torch.no_grad():
        x = torch.tensor(bag, dtype=torch.float32).unsqueeze(0).to(device)
        logits = model(x)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        return int(logits.squeeze().argmax().cpu())


# ── Inferencia ────────────────────────────────────────────────────────────────

def run_inference(meta, ds, transforms_by_class, scaler, pca, model, device,
                  coral_mode, std_transform=None):
    """
    coral_mode: 'none' | 'standard' | 'conditional'
    transforms_by_class: {cls_id: (mu_s, mu_t, W)} para modo conditional
    std_transform: (mu_s, mu_t, W) para modo standard
    """
    mask   = (meta['dataset'] == ds).values
    slides = meta.loc[mask, 'slide'].tolist()
    labels = meta.loc[mask, 'category'].tolist()

    preds = []
    for slide, label in zip(slides, labels):
        p = BAGS_DIR / f'{slide}.pt'
        if not p.exists():
            preds.append(-1)
            continue
        bag = torch.load(p, map_location='cpu', weights_only=True).float().numpy()

        if coral_mode != 'none':
            bag_pca = pca.transform(scaler.transform(bag))   # (N_patches × 128)

            if coral_mode == 'standard' and std_transform is not None:
                mu_s, mu_t, W = std_transform
                bag_pca = apply_transform(bag_pca, mu_s, mu_t, W)

            elif coral_mode == 'conditional':
                t = transforms_by_class.get(label)
                if t is not None:
                    mu_s, mu_t, W = t
                    bag_pca = apply_transform(bag_pca, mu_s, mu_t, W)

            bag = pca.inverse_transform(bag_pca)              # (N_patches × 1536)

        preds.append(predict_bag(model, bag, device))

    valid    = [(p, y) for p, y in zip(preds, labels) if p != -1]
    preds_v  = np.array([v[0] for v in valid])
    labels_v = np.array([v[1] for v in valid])

    mss = labels_v == 0;  msi = labels_v == 1
    acc_mss = (preds_v[mss] == 0).mean() if mss.sum() > 0 else float('nan')
    acc_msi = (preds_v[msi] == 1).mean() if msi.sum() > 0 else float('nan')
    bal     = balanced_accuracy_score(labels_v, preds_v)
    return acc_mss, acc_msi, bal, len(valid)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_pca',        type=int,   default=128)
    parser.add_argument('--alpha',        type=float, default=1.0)
    parser.add_argument('--seed',         type=int,   default=42)
    parser.add_argument('--mode',         choices=['none', 'standard', 'oracle', 'fewshot'],
                        default='oracle')
    parser.add_argument('--fewshot_n',    type=int,   default=10,
                        help='Slides etiquetados por clase en modo fewshot')
    parser.add_argument('--fewshot_seeds', type=int,  default=5,
                        help='Semillas para el muestreo few-shot')
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}  |  Modo: {args.mode}')

    # 1. Cargar slide means + ajustar PCA sobre Macarena
    print('\nCargando slide means...')
    ann = pd.read_csv(ANNOT)
    X_all, meta = load_slide_means(ann)
    mask_mac = (meta['dataset'] == 'macarena').values
    X_mac    = X_all[mask_mac]
    print(f'  Macarena: {len(X_mac)} slides')

    scaler = StandardScaler().fit(X_mac)
    pca    = PCA(n_components=args.n_pca, random_state=args.seed).fit(
                 scaler.transform(X_mac))
    X_pca  = pca.transform(scaler.transform(X_all))
    print(f'  Varianza explicada PCA: {pca.explained_variance_ratio_.sum():.1%}')

    # 2. Cargar modelo
    print('\nCargando modelo DSMIL...')
    model = load_model(device)
    print('  Modelo cargado.')

    rows = []
    print(f'\nInferencia MIL — modo: {args.mode}')
    print(f'  {"tag":30s}  {"dataset":12s}  MSS     MSI     bal-acc  n')
    print(f'  {"-"*72}')

    def report(tag, ds, a_mss, a_msi, bal, n):
        name = DATASET_LABELS[ds]
        print(f'  [{tag:28s}]  {name:12s}  '
              f'MSS={a_mss:.3f}  MSI={a_msi:.3f}  bal={bal:.3f}  n={n}')
        rows.append({'tag': tag, 'dataset': name,
                     'acc_mss': a_mss, 'acc_msi': a_msi, 'bal_acc': bal})

    # Siempre evaluamos Macarena sin CORAL como referencia
    for ds in ['macarena', 'cptac_coad', 'tcga_coad']:
        a_mss, a_msi, bal, n = run_inference(
            meta, ds, {}, scaler, pca, model, device, 'none')
        report('Sin CORAL', ds, a_mss, a_msi, bal, n)
    print()

    if args.mode == 'standard':
        # CORAL estándar: un transform global por dominio (sin etiquetas)
        for ds in ['cptac_coad', 'tcga_coad']:
            mask_ds = (meta['dataset'] == ds).values
            std_t = fit_coral_transform(X_pca[mask_ds], X_pca[mask_mac], args.alpha)
            a_mss, a_msi, bal, n = run_inference(
                meta, ds, {}, scaler, pca, model, device, 'standard', std_t)
            report('CORAL estándar', ds, a_mss, a_msi, bal, n)

    elif args.mode == 'oracle':
        # CORAL condicional con todas las etiquetas verdaderas
        for ds in ['cptac_coad', 'tcga_coad']:
            true_labels = meta['category'].values
            transforms  = fit_conditional_transforms(
                X_pca, meta, ds, true_labels, args.alpha)
            a_mss, a_msi, bal, n = run_inference(
                meta, ds, transforms, scaler, pca, model, device, 'conditional')
            report('CORAL cond. oracle', ds, a_mss, a_msi, bal, n)

    elif args.mode == 'fewshot':
        # CORAL condicional con N slides etiquetados por clase
        N = args.fewshot_n
        for ds in ['cptac_coad', 'tcga_coad']:
            mask_ds  = (meta['dataset'] == ds).values
            idx_ds   = np.where(mask_ds)[0]
            y_ds     = meta['category'].values[mask_ds]
            idx_mss  = idx_ds[y_ds == 0]
            idx_msi  = idx_ds[y_ds == 1]

            n_avail = min(len(idx_mss), len(idx_msi))
            if N > n_avail:
                print(f'  Aviso: N={N} > {n_avail} disponibles en {ds}, usando {n_avail}')
                N_use = n_avail
            else:
                N_use = N

            accs = []
            for seed in range(args.seed, args.seed + args.fewshot_seeds):
                rng = np.random.default_rng(seed)
                labeled = np.concatenate([
                    rng.choice(idx_mss, N_use, replace=False),
                    rng.choice(idx_msi, N_use, replace=False),
                ])
                pseudo = np.full(len(meta), -1)
                pseudo[labeled] = meta['category'].values[labeled]
                pseudo[mask_mac] = meta['category'].values[mask_mac]

                transforms = fit_conditional_transforms(
                    X_pca, meta, ds, pseudo, args.alpha)
                a_mss, a_msi, bal, n = run_inference(
                    meta, ds, transforms, scaler, pca, model, device, 'conditional')
                accs.append(bal)

            mu, sd = np.mean(accs), np.std(accs)
            tag = f'CORAL cond. few-shot N={N_use}'
            print(f'  [{tag:28s}]  {DATASET_LABELS[ds]:12s}  '
                  f'bal={mu:.3f} ± {sd:.3f}  ({args.fewshot_seeds} semillas)')
            rows.append({'tag': tag, 'dataset': DATASET_LABELS[ds],
                         'bal_acc': mu, 'bal_std': sd})

    pd.DataFrame(rows).to_csv(OUT_DIR / f'results_{args.mode}_pca{args.n_pca}.csv',
                               index=False)
    print(f'\nResultados en: {OUT_DIR}')


if __name__ == '__main__':
    main()