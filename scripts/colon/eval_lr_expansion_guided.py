#!/usr/bin/env python3
"""
eval_lr_expansion_guided.py

Expansión guiada del espacio de Macarena usando slides de calibración del target.

Representación: media de los K patches más atentos del DSMIL por slide.
Expansión: los slides de Macarena más alejados del centroide (distancia coseno)
se empujan en la dirección Macarena → target (estimada con N slides etiquetados).
Los sintéticos llenan el camino entre clusters clase a clase.

LR entrenada en Macarena + sintéticos + calibración, evaluada en el 90% restante.
Repite con múltiples semillas.

Uso:
  python scripts/colon/eval_lr_expansion_guided.py --n_gen 3 --lam 0.5 --seeds 10
"""

import warnings
import json
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

warnings.filterwarnings('ignore')

ROOT      = Path(__file__).resolve().parents[2]
BAGS_DIR  = Path('/home/PARADIS/colon/tfrecords/shared_bags/256_128_none_h_optimus_0')
MODEL_DIR = ROOT / 'experiments/colon_mss_msi_benchmark/mil/00000-256_128_none_h_optimus_0_dsmil_CrossEntropyLoss_ReLU_Adam_1'
ANNOT     = ROOT / 'config/annotations/annotations_colon_mss_msi_all.csv'
OUT_DIR   = ROOT / 'results/colon/lr_expansion_guided'

DATASET_LABELS = {'cptac_coad': 'CPTAC-COAD', 'tcga_coad': 'TCGA-COAD'}
N_VALUES = [5, 10, 20, 30]


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


def top_k_attention_mean(model, slide, device, k=8):
    """Media de los k patches con mayor peso de atención del DSMIL."""
    p = BAGS_DIR / f'{slide}.pt'
    if not p.exists():
        return None
    bag = torch.load(p, map_location='cpu', weights_only=True).float()
    with torch.no_grad():
        x           = bag.unsqueeze(0).to(device)
        inst_feat   = model.instance_encoder(x.view(-1, x.size(-1)))
        inst_feat   = inst_feat.view(1, bag.shape[0], -1)
        inst_scores = model.instance_classifier(inst_feat).view(1, bag.shape[0])
        _, max_idx  = inst_scores.max(dim=1)
        critical    = inst_feat[0, max_idx[0]]
        attn_w      = F.softmax(
            model.attention(inst_feat - critical.unsqueeze(0).unsqueeze(0)), dim=1
        ).squeeze()
        topk_idx    = attn_w.topk(min(k, bag.shape[0])).indices
    return bag[topk_idx.cpu()].mean(dim=0).numpy().astype(np.float32)


def cosine_distance_to(X, ref):
    """Distancia coseno de cada fila de X al vector ref."""
    X_n   = X   / (np.linalg.norm(X,   axis=1, keepdims=True) + 1e-10)
    ref_n = ref / (np.linalg.norm(ref) + 1e-10)
    return 1.0 - X_n @ ref_n


def guided_expansion(X_mac_cls, centroid_mac, centroid_tgt,
                     n_synthetic, lam, top_frac, rng):
    """
    Genera n_synthetic slides virtuales empujando los slides de borde de Macarena
    en la dirección Macarena → target.

    Selección de borde: top_frac slides de Macarena con mayor distancia coseno
                        a su propio centroide.
    Dirección guiada:   centroid_tgt - centroid_mac  (normalizada)
    Generación:         x_new = x_i + λ * dirección + ruido suave
                        con λ ~ U(0, lam)  → puntos a lo largo del camino
    """
    # Dirección de dominio normalizada (coseno)
    direction  = centroid_tgt - centroid_mac
    direction /= (np.linalg.norm(direction) + 1e-10)

    # Seleccionar slides de borde por distancia coseno al centroide de Macarena
    cos_dists = cosine_distance_to(X_mac_cls, centroid_mac)
    n_top     = max(1, int(len(X_mac_cls) * top_frac))
    top_idx   = np.argsort(cos_dists)[-n_top:]

    mean_norm = np.linalg.norm(X_mac_cls, axis=1).mean()

    synthetic = []
    for _ in range(n_synthetic):
        i     = rng.choice(top_idx)
        x_i   = X_mac_cls[i]
        lam_i = rng.uniform(0, lam)
        x_new = x_i + lam_i * direction * np.linalg.norm(x_i)
        x_new += rng.normal(0, mean_norm * 0.02, size=x_new.shape)
        synthetic.append(x_new)

    return np.stack(synthetic).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_pca',         type=int,   default=128)
    parser.add_argument('--k',             type=int,   default=8)
    parser.add_argument('--n_gen',         type=int,   default=3,
                        help='Generaciones de expansión')
    parser.add_argument('--lam',           type=float, default=1.0,
                        help='Máx. desplazamiento por generación (fracción de la dirección)')
    parser.add_argument('--top_frac',      type=float, default=0.3,
                        help='Fracción de slides de borde para extrapolar')
    parser.add_argument('--n_synthetic',   type=int,   default=100,
                        help='Sintéticos por clase y generación')
    parser.add_argument('--seeds',         type=int,   default=10)
    parser.add_argument('--seed0',         type=int,   default=0)
    parser.add_argument('--min_per_class', type=int,   default=5)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}  |  top-K={args.k}  |  n_gen={args.n_gen}  |  λ={args.lam}\n')

    ann = pd.read_csv(ANNOT)
    model = load_model(device)

    # 1. Extraer representación top-K atención para todos los slides
    print('Extrayendo representación top-K atención...')
    embeds, metas = {}, []
    for _, row in ann.iterrows():
        emb = top_k_attention_mean(model, row['slide'], device, k=args.k)
        if emb is None:
            continue
        embeds[row['slide']] = emb
        metas.append({'slide': row['slide'], 'dataset': row['dataset'],
                      'category': int(row['category'])})

    meta  = pd.DataFrame(metas)
    X_raw = np.stack([embeds[s] for s in meta['slide']]).astype(np.float32)
    print(f'  Slides procesados: {len(meta)}')

    # PCA(n_pca) ajustado sobre representaciones top-K de Macarena
    mask_mac = (meta['dataset'] == 'macarena').values
    scaler   = StandardScaler().fit(X_raw[mask_mac])
    pca      = PCA(n_components=args.n_pca, random_state=args.seed0).fit(
                   scaler.transform(X_raw[mask_mac]))
    X_all    = pca.transform(scaler.transform(X_raw))
    print(f'  Varianza explicada PCA: {pca.explained_variance_ratio_.sum():.1%}\n')

    X_mac    = X_all[mask_mac]
    y_mac    = meta.loc[mask_mac, 'category'].values

    # Centroides de Macarena por clase (fijos)
    centroids_mac = {cls: X_mac[y_mac == cls].mean(axis=0) for cls in [0, 1]}

    # Baseline: LR solo Macarena
    lr_base = LogisticRegression(max_iter=2000, random_state=args.seed0, C=1.0)
    lr_base.fit(X_mac, y_mac)
    print('Baseline (LR solo Macarena, representación top-K atención):')
    for ds in ['cptac_coad', 'tcga_coad']:
        mask = (meta['dataset'] == ds).values
        y    = meta.loc[mask, 'category'].values
        yp   = lr_base.predict(X_all[mask])
        print(f'  {DATASET_LABELS[ds]}:  '
              f'MSS={( yp[y==0]==0).mean():.3f}  '
              f'MSI={(yp[y==1]==1).mean():.3f}  '
              f'bal={balanced_accuracy_score(y, yp):.3f}')

    # 2. Curva: calibración + expansión guiada
    print(f'\nCurva expansión guiada ({args.seeds} semillas × {len(N_VALUES)} N)...\n')
    records = []

    for ds in ['cptac_coad', 'tcga_coad']:
        mask_ds = (meta['dataset'] == ds).values
        idx_ds  = np.where(mask_ds)[0]
        y_ds    = meta.loc[mask_ds, 'category'].values
        idx_mss = idx_ds[y_ds == 0]
        idx_msi = idx_ds[y_ds == 1]
        print(f'  {DATASET_LABELS[ds]}  (MSS={len(idx_mss)}, MSI={len(idx_msi)})')

        for N in N_VALUES:
            n_mss = min(max(args.min_per_class, N), len(idx_mss) - 1)
            n_msi = min(max(args.min_per_class, N), len(idx_msi) - 1)

            rng_seeds = np.random.default_rng(args.seed0 + hash(ds) % 1000).integers(
                0, 100000, size=args.seeds)

            accs_lr, accs_ex = [], []

            for seed in rng_seeds:
                rng = np.random.default_rng(seed)
                calib_idx = np.concatenate([
                    rng.choice(idx_mss, n_mss, replace=False),
                    rng.choice(idx_msi, n_msi, replace=False),
                ])
                test_idx = np.setdiff1d(idx_ds, calib_idx)
                y_test   = meta['category'].values[test_idx]
                X_calib  = X_all[calib_idx]
                y_calib  = meta['category'].values[calib_idx]

                # ── LR calibrada (sin expansión) ──────────────────────────────
                X_tr = np.concatenate([X_mac, X_calib])
                y_tr = np.concatenate([y_mac, y_calib])
                lr = LogisticRegression(max_iter=2000, random_state=int(seed), C=1.0)
                lr.fit(X_tr, y_tr)
                accs_lr.append(balanced_accuracy_score(y_test, lr.predict(X_all[test_idx])))

                # ── LR + expansión guiada iterativa ───────────────────────────
                X_aug = X_mac.copy()
                y_aug = y_mac.copy()

                for gen in range(args.n_gen):
                    new_X, new_y = [], []
                    for cls_id in [0, 1]:
                        X_cls_calib = X_calib[y_calib == cls_id]
                        if len(X_cls_calib) == 0:
                            continue
                        centroid_tgt = X_cls_calib.mean(axis=0)
                        X_mac_cls    = X_aug[y_aug == cls_id]
                        centroid_mac = X_mac_cls.mean(axis=0)

                        syn = guided_expansion(
                            X_mac_cls, centroid_mac, centroid_tgt,
                            args.n_synthetic, args.lam, args.top_frac, rng)
                        new_X.append(syn)
                        new_y.append(np.full(len(syn), cls_id, dtype=int))

                    X_aug = np.concatenate([X_aug] + new_X)
                    y_aug = np.concatenate([y_aug] + new_y)

                X_tr_ex = np.concatenate([X_aug, X_calib])
                y_tr_ex = np.concatenate([y_aug, y_calib])
                lr_ex = LogisticRegression(max_iter=2000, random_state=int(seed), C=1.0)
                lr_ex.fit(X_tr_ex, y_tr_ex)
                accs_ex.append(balanced_accuracy_score(y_test, lr_ex.predict(X_all[test_idx])))

            mu_lr, sd_lr = np.mean(accs_lr), np.std(accs_lr)
            mu_ex, sd_ex = np.mean(accs_ex), np.std(accs_ex)
            print(f'    N={N:2d}/clase  '
                  f'LR={mu_lr:.3f}±{sd_lr:.3f}  '
                  f'LR+Expansión={mu_ex:.3f}±{sd_ex:.3f}')
            records.append({
                'dataset': DATASET_LABELS[ds], 'N': N,
                'mean_lr': mu_lr, 'std_lr': sd_lr,
                'mean_ex': mu_ex, 'std_ex': sd_ex,
            })
        print()

    pd.DataFrame(records).to_csv(
        OUT_DIR / f'results_k{args.k}_gen{args.n_gen}_lam{args.lam}.csv', index=False)
    print(f'Listo. Resultados en: {OUT_DIR}')


if __name__ == '__main__':
    main()