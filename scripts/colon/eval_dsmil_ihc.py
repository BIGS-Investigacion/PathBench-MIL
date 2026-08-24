#!/usr/bin/env python3
"""
eval_dsmil_ihc.py

Evaluación del modelo DSMIL (entrenado en Macarena MSS/MSI) sobre los slides
de TCGA-COAD con etiquetas IHC de proteínas MMR.

Para cada tarea (MSS/MSI, MLH1, MSH2, MSH6, PMS2) reporta:
  - Acc(0), Acc(1), Balanced Accuracy

Uso:
  python scripts/colon/eval_dsmil_ihc.py
"""

import warnings
import json
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score

warnings.filterwarnings('ignore')

ROOT      = Path(__file__).resolve().parents[2]
BAGS_DIR  = Path('/home/PARADIS/datos/features/shared_bags/256_128_none_h_optimus_0')
MODEL_DIR = ROOT / 'experiments/colon_mss_msi_benchmark/mil/00000-256_128_none_h_optimus_0_dsmil_CrossEntropyLoss_ReLU_Adam_1'
ANNOT_DIR = ROOT / 'config/annotations'

IHC_FILES = {
    'MSS/MSI': ANNOT_DIR / 'annotations_tcga_coad_mss_msi_ihc.csv',
    'MLH1':    ANNOT_DIR / 'annotations_tcga_coad_mlh1_ihc.csv',
    'MSH2':    ANNOT_DIR / 'annotations_tcga_coad_msh2_ihc.csv',
    'MSH6':    ANNOT_DIR / 'annotations_tcga_coad_msh6_ihc.csv',
    'PMS2':    ANNOT_DIR / 'annotations_tcga_coad_pms2_ihc.csv',
}


def load_dsmil(device):
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


def predict_slide(model, slide, device):
    p = BAGS_DIR / f'{slide}.pt'
    if not p.exists():
        return None
    bag = torch.load(p, map_location='cpu', weights_only=True).float()
    with torch.no_grad():
        logits = model(bag.unsqueeze(0).to(device))
        prob = torch.softmax(logits, dim=1).squeeze()
    return prob[1].item()  # P(MSI)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')
    print(f'Modelo: DSMIL MSS/MSI (Macarena)\n')

    model = load_dsmil(device)

    print(f'{"Tarea":<12} {"N(0)/N(1)":>12} {"Acc(0)":>8} {"Acc(1)":>8} {"Bal":>8}')
    print('=' * 54)

    for task, ann_path in IHC_FILES.items():
        ann = pd.read_csv(ann_path)
        preds, labels = [], []
        for _, row in ann.iterrows():
            prob = predict_slide(model, row['slide'], device)
            if prob is None:
                continue
            preds.append(1 if prob >= 0.5 else 0)
            labels.append(int(row['category']))

        y_true = np.array(labels)
        y_pred = np.array(preds)
        n0, n1 = (y_true == 0).sum(), (y_true == 1).sum()
        a0 = (y_pred[y_true == 0] == 0).mean() if n0 > 0 else np.nan
        a1 = (y_pred[y_true == 1] == 1).mean() if n1 > 0 else np.nan
        bal = balanced_accuracy_score(y_true, y_pred)

        print(f'  {task:<10} {str(n0)+"/"+str(n1):>12} {a0:>8.3f} {a1:>8.3f} {bal:>8.3f}')

    print()


if __name__ == '__main__':
    main()