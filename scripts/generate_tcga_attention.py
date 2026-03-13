#!/usr/bin/env python3
"""
generate_tcga_attention.py

Para cada combinación (tarea, MIL) con normalización 'none', carga el modelo
entrenado y computa los vectores de atención para todas las WSIs TCGA que aún
no tienen fichero _att.npz en la carpeta mil/.../attention/.

Los ficheros se guardan en el mismo formato que usa PathBench durante eval:
  <att_dir>/<slide>_att.npz  →  arr_0: (n_patches,) float32

Uso:
  python scripts/generate_tcga_attention.py [--tasks er erbb2 pr pam50]
                                            [--mils transmil dsmil clam_mil_mb]
                                            [--force]  # sobreescribe existentes
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

# ── Rutas ─────────────────────────────────────────────────────────────────────

ROOT         = Path(__file__).resolve().parents[1]
EXPERIMENTS  = ROOT / 'experiments'
BAGS_DIR     = Path('/home/PARADIS/mama/tfrecords/shared_bags/256_128_none_virchow2')

TASKS    = ['er', 'erbb2', 'pr', 'pam50']
MIL_KEYS = ['transmil', 'dsmil', 'clam_mil_mb']

TASK_N_CLASSES = {'er': 2, 'erbb2': 2, 'pr': 2, 'pam50': 5}

_SF_MIL = {
    'transmil': str(ROOT / 'slideflow_fork/slideflow/mil/models/transmil.py'),
}


# ── Carga de modelo ───────────────────────────────────────────────────────────

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _get_model_cls(model_name: str):
    if model_name.lower() in _SF_MIL:
        mod = _load_module(f'sf_{model_name}', _SF_MIL[model_name.lower()])
        return getattr(mod, 'TransMIL')
    agg = _load_module('aggregators', str(ROOT / 'pathbench/models/aggregators.py'))
    return getattr(agg, model_name)


def load_model(model_dir: Path, device: torch.device):
    with open(model_dir / 'mil_params.json') as f:
        mp = json.load(f)
    params = mp['params']
    cls    = _get_model_cls(params['model'])
    model  = cls(
        n_feats=mp['input_shape'],
        n_out=mp['output_shape'],
        z_dim=params.get('z_dim', 256),
        dropout_p=params.get('dropout_p', 0.1),
        activation_function=params.get('activation_function', 'ReLU'),
        encoder_layers=params.get('encoder_layers', 1),
    )
    ckpts = sorted(model_dir.glob('checkpoints/**/best-epoch*.ckpt'))
    if not ckpts:
        raise FileNotFoundError(f'Sin checkpoint en {model_dir}')
    ckpt  = torch.load(ckpts[-1], map_location='cpu')
    state = {k.removeprefix('model.'): v
             for k, v in ckpt['state_dict'].items() if k.startswith('model.')}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), mp['output_shape']


# ── Cómputo de atención ───────────────────────────────────────────────────────

def compute_attention(model, features: torch.Tensor,
                      n_patches: int, n_classes: int) -> np.ndarray:
    """Devuelve array (n_patches,) float32 replicando la lógica de eval.py."""
    with torch.no_grad():
        inp = features.unsqueeze(0)
        try:
            att = model.calculate_attention(inp)
        except TypeError:
            att = model.calculate_attention(inp, apply_softmax=False)

    att = torch.squeeze(att)
    if att.dim() == 2:          # TransMIL: (n_patches, heads) → avg
        att = att.mean(dim=-1)

    arr = att.cpu().float().numpy()
    if arr.ndim == 1 and arr.shape[0] == n_patches * n_classes and n_classes > 1:
        arr = arr.reshape(n_patches, n_classes).mean(axis=1)
    return arr.astype(np.float32)


# ── Lógica principal ──────────────────────────────────────────────────────────

def process(task: str, mil: str, device: torch.device, force: bool) -> None:
    exp_dir  = EXPERIMENTS / f'brca_virchow2_test_{task}_{mil}'
    runs     = sorted((exp_dir / 'mil').glob('00001-*none*'))
    if not runs:
        print(f'  [SKIP] No hay run none para {task}/{mil}')
        return
    model_dir = runs[0]
    att_dir   = model_dir / 'attention'
    att_dir.mkdir(exist_ok=True)

    n_classes = TASK_N_CLASSES[task]

    # Slides TCGA disponibles
    all_tcga = sorted(BAGS_DIR.glob('TCGA-*.pt'))

    # Filtrar los que ya tienen atención
    if not force:
        existing = {p.name.replace('_att.npz', '') for p in att_dir.glob('*_att.npz')}
        to_do    = [p for p in all_tcga if p.stem not in existing]
    else:
        to_do = all_tcga

    print(f'  {task}/{mil}: {len(to_do)} slides a procesar '
          f'({len(all_tcga) - len(to_do)} ya existentes)')

    if not to_do:
        return

    model, n_out = load_model(model_dir, device)

    for i, bag_path in enumerate(to_do, 1):
        slide    = bag_path.stem
        out_path = att_dir / f'{slide}_att.npz'
        try:
            features  = torch.load(bag_path, map_location=device,
                                   weights_only=True).float()
            n_patches = features.shape[0]
            arr       = compute_attention(model, features, n_patches, n_out)
            np.savez_compressed(out_path, arr_0=arr)
            if i % 100 == 0 or i == len(to_do):
                print(f'    [{i}/{len(to_do)}] {slide}')
        except Exception as e:
            print(f'    [WARN] {slide}: {e}')

    print(f'  Completado: {task}/{mil}')


def main():
    parser = argparse.ArgumentParser(
        description='Genera ficheros _att.npz TCGA faltantes en carpetas de experimento.')
    parser.add_argument('--tasks', nargs='+', default=TASKS, choices=TASKS)
    parser.add_argument('--mils',  nargs='+', default=MIL_KEYS,
                        choices=MIL_KEYS, dest='mils')
    parser.add_argument('--force', action='store_true',
                        help='Sobreescribir ficheros existentes')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Dispositivo: {device}')
    print(f'Tareas: {args.tasks}  |  MILs: {args.mils}\n')

    for task in args.tasks:
        for mil in args.mils:
            print(f'=== {task.upper()} / {mil} ===')
            process(task, mil, device, args.force)
            print()


if __name__ == '__main__':
    main()