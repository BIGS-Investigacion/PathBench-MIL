#!/usr/bin/env python3
"""
eval_dsmil_finetune.py

Re-entrenamiento (fine-tuning) del modelo DSMIL preentrenado en Macarena,
usando un 10% estratificado (por clase) del dominio target con etiquetas.
Evaluación sobre el 90% restante.

Repite con múltiples semillas y reporta media ± std de balanced accuracy.

Uso:
  python scripts/colon/eval_dsmil_finetune.py --ds cptac_coad --seeds 10
  python scripts/colon/eval_dsmil_finetune.py --ds tcga_coad  --seeds 10
"""

import warnings
import json
import argparse
import importlib.util
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score

warnings.filterwarnings('ignore')

ROOT      = Path(__file__).resolve().parents[2]
BAGS_DIR  = Path('/home/PARADIS/colon/tfrecords/shared_bags/256_128_none_h_optimus_0')
MODEL_DIR = ROOT / 'experiments/colon_mss_msi_benchmark/mil/00000-256_128_none_h_optimus_0_dsmil_CrossEntropyLoss_ReLU_Adam_1'
ANNOT     = ROOT / 'config/annotations/annotations_colon_mss_msi_all.csv'
OUT_DIR   = ROOT / 'results/colon/dsmil_finetune'

DATASET_LABELS = {'cptac_coad': 'CPTAC-COAD', 'tcga_coad': 'TCGA-COAD'}


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
    return model.to(device)


def load_bag(slide):
    p = BAGS_DIR / f'{slide}.pt'
    if not p.exists():
        return None
    return torch.load(p, map_location='cpu', weights_only=True).float()


def predict_all(model, slides, device):
    model.eval()
    preds, probs = [], []
    with torch.no_grad():
        for slide in slides:
            bag = load_bag(slide)
            if bag is None:
                preds.append(-1); probs.append(-1.0)
                continue
            x = bag.unsqueeze(0).to(device)
            logits = model(x)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            p = torch.softmax(logits.squeeze(), dim=-1)
            preds.append(int(p.argmax().cpu()))
            probs.append(p[1].item())
    return np.array(preds), np.array(probs)


def finetune(model_init, train_slides, train_labels, device, lr, epochs, bag_size,
             accum_steps=4):
    """Fine-tune sobre train_slides con CrossEntropy + acumulación de gradientes."""
    model = copy.deepcopy(model_init).to(device)

    # BN en eval para evitar error con batch_size=1; el resto en train
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.eval()

    # Class weights para compensar desbalance
    n_mss = (train_labels == 0).sum()
    n_msi = (train_labels == 1).sum()
    w = torch.tensor([1.0/n_mss, 1.0/n_msi], dtype=torch.float32).to(device)
    w = w / w.sum()
    criterion = nn.CrossEntropyLoss(weight=w)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    idx = np.arange(len(train_slides))
    for epoch in range(epochs):
        np.random.shuffle(idx)
        total_loss = 0.0
        optimizer.zero_grad()
        for step, i in enumerate(idx):
            bag = load_bag(train_slides[i])
            if bag is None:
                continue
            if bag.shape[0] > bag_size:
                perm = torch.randperm(bag.shape[0])[:bag_size]
                bag  = bag[perm]
            x = bag.unsqueeze(0).to(device)
            y = torch.tensor([train_labels[i]], dtype=torch.long).to(device)

            logits = model(x)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            loss = criterion(logits, y) / accum_steps
            loss.backward()
            total_loss += loss.item() * accum_steps

            if (step + 1) % accum_steps == 0 or step == len(idx) - 1:
                optimizer.step()
                optimizer.zero_grad()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'      epoch {epoch+1:3d}/{epochs}  loss={total_loss/len(train_slides):.4f}')

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds',       choices=['cptac_coad', 'tcga_coad'], required=True)
    parser.add_argument('--calib',    type=float, default=0.10,
                        help='Fracción del dominio target usada para calibración')
    parser.add_argument('--min_per_class', type=int, default=5,
                        help='Mínimo de slides por clase en calibración')
    parser.add_argument('--seeds',    type=int,   default=10)
    parser.add_argument('--seed0',    type=int,   default=42)
    parser.add_argument('--lr',       type=float, default=1e-4)
    parser.add_argument('--epochs',   type=int,   default=20)
    parser.add_argument('--bag_size', type=int,   default=512,
                        help='Máx. patches por bag durante fine-tuning')
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds_name = DATASET_LABELS[args.ds]
    print(f'Device: {device}  |  Dataset: {ds_name}  |  calib={args.calib:.0%}')
    print(f'LR={args.lr}  |  epochs={args.epochs}  |  seeds={args.seeds}\n')

    ann = pd.read_csv(ANNOT)
    sub = ann[ann['dataset'] == args.ds].reset_index(drop=True)
    slides = sub['slide'].tolist()
    labels = sub['category'].astype(int).tolist()

    # Filtrar slides que existen
    exists = [i for i, s in enumerate(slides) if (BAGS_DIR / f'{s}.pt').exists()]
    slides = [slides[i] for i in exists]
    labels = np.array([labels[i] for i in exists])
    print(f'Slides disponibles: {len(slides)}  '
          f'(MSS={( labels==0).sum()}, MSI={(labels==1).sum()})')

    # Baseline: modelo original sin fine-tuning
    print('\nCargando modelo base...')
    model_base = load_model(device)
    model_base.eval()
    preds_base, _ = predict_all(model_base, slides, device)
    valid = preds_base != -1
    bal_base = balanced_accuracy_score(labels[valid], preds_base[valid])
    print(f'Baseline (sin fine-tuning): bal-acc={bal_base:.3f}')

    # Calcular n_train por clase: max(calib%, min_per_class)
    idx_mss = np.where(labels == 0)[0]
    idx_msi = np.where(labels == 1)[0]
    n_train_mss = max(args.min_per_class, int(round(len(idx_mss) * args.calib)))
    n_train_msi = max(args.min_per_class, int(round(len(idx_msi) * args.calib)))
    n_train_mss = min(n_train_mss, len(idx_mss) - 1)
    n_train_msi = min(n_train_msi, len(idx_msi) - 1)
    print(f'Calibración por clase: MSS={n_train_mss}, MSI={n_train_msi} '
          f'(de {len(idx_mss)} y {len(idx_msi)} disponibles)\n')

    results = []
    rng_seeds = np.random.default_rng(args.seed0).integers(0, 100000, size=args.seeds)
    for seed_idx, seed in enumerate(rng_seeds):
        rng = np.random.default_rng(seed)
        torch.manual_seed(int(seed)); np.random.seed(int(seed))

        train_mss = rng.choice(idx_mss, n_train_mss, replace=False)
        train_msi = rng.choice(idx_msi, n_train_msi, replace=False)
        train_idx = np.concatenate([train_mss, train_msi])
        test_idx  = np.setdiff1d(np.arange(len(slides)), train_idx)

        train_slides = [slides[i] for i in train_idx]
        train_labels = labels[train_idx]
        test_slides  = [slides[i] for i in test_idx]
        test_labels  = labels[test_idx]

        n_mss = (train_labels == 0).sum()
        n_msi = (train_labels == 1).sum()
        print(f'\n  Semilla {seed_idx+1}/{args.seeds} — '
              f'train={len(train_slides)} (MSS={n_mss}, MSI={n_msi}), '
              f'test={len(test_slides)}')

        model_ft = finetune(model_base, train_slides, train_labels,
                            device, args.lr, args.epochs, args.bag_size)

        preds_ft, _ = predict_all(model_ft, test_slides, device)
        valid_ft = preds_ft != -1
        bal_ft = balanced_accuracy_score(test_labels[valid_ft], preds_ft[valid_ft])
        print(f'    → Fine-tuned bal-acc={bal_ft:.3f}')

        # Baseline en el mismo split test
        preds_b, _ = predict_all(model_base, test_slides, device)
        valid_b = preds_b != -1
        bal_b = balanced_accuracy_score(test_labels[valid_b], preds_b[valid_b])

        results.append({'seed': seed, 'bal_base': bal_b, 'bal_ft': bal_ft,
                        'n_train': len(train_slides), 'n_test': len(test_slides)})

    df = pd.DataFrame(results)
    print(f'\n{"="*55}')
    print(f'  {ds_name} — fine-tuning {args.calib:.0%}  ({args.seeds} semillas)')
    print(f'{"="*55}')
    print(f'  Baseline (splits test): {df["bal_base"].mean():.3f} ± {df["bal_base"].std():.3f}')
    print(f'  Fine-tuned:             {df["bal_ft"].mean():.3f} ± {df["bal_ft"].std():.3f}')
    print(f'{"="*55}')

    tag = f'{args.ds}_calib{int(args.calib*100)}'
    df.to_csv(OUT_DIR / f'results_{tag}.csv', index=False)
    print(f'\nResultados en: {OUT_DIR}')


if __name__ == '__main__':
    main()