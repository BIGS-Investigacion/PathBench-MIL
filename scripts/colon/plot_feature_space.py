#!/usr/bin/env python3
"""
plot_feature_space.py

Visualiza en 2D el espacio de features h-optimus-0 de las tres bases de datos
(Macarena, CPTAC-COAD, TCGA-COAD) usando UMAP o t-SNE.

Cada punto = un slide, representado por la media de sus features de patch
(o media ponderada por atención si se usa --weighted).

Genera cuatro figuras:
  1. Coloreado por base de datos
  2. Coloreado por etiqueta MSS/MSI
  3. Coloreado por base de datos × MSS/MSI (combinado)
  4. Subplots por base de datos con etiqueta MSS/MSI

Uso:
  python scripts/colon/plot_feature_space.py [--method umap|tsne] [--weighted] [--seed 42]
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

warnings.filterwarnings('ignore')

# ── Rutas ─────────────────────────────────────────────────────────────────────

ROOT      = Path(__file__).resolve().parents[2]
BAGS_DIR  = Path('/home/PARADIS/colon/tfrecords/shared_bags/256_128_none_h_optimus_0')
MODEL_DIR = ROOT / 'experiments/colon_mss_msi_benchmark/mil/00000-256_128_none_h_optimus_0_dsmil_CrossEntropyLoss_ReLU_Adam_1'
ANNOT     = ROOT / 'config/annotations/annotations_colon_mss_msi_all.csv'
OUT_DIR   = ROOT / 'results/colon/feature_space'

DATASET_COLORS = {
    'macarena':  '#2196F3',   # azul
    'cptac_coad':'#FF9800',   # naranja
    'tcga_coad': '#4CAF50',   # verde
}
DATASET_LABELS = {
    'macarena':  'Macarena',
    'cptac_coad':'CPTAC-COAD',
    'tcga_coad': 'TCGA-COAD',
}
CLASS_COLORS = {0: '#E53935', 1: '#1E88E5'}   # 0=MSS rojo, 1=MSI azul
CLASS_NAMES  = {0: 'MSS', 1: 'MSI'}
CLASS_MARKERS = {0: 'o', 1: '^'}


# ── Carga de atención (opcional) ──────────────────────────────────────────────

def load_model_for_attention(device):
    import importlib.util, sys
    sys.path.insert(0, str(ROOT))
    agg = importlib.import_module('pathbench.models.aggregators')

    with open(MODEL_DIR / 'mil_params.json') as f:
        mp = json.load(f)
    params = mp['params']
    cls = getattr(agg, params['model'])
    model = cls(
        n_feats=mp['input_shape'],
        n_out=mp['output_shape'],
        z_dim=params.get('z_dim', 256),
        dropout_p=params.get('dropout_p', 0.1),
        activation_function=params.get('activation_function', 'ReLU'),
        encoder_layers=params.get('encoder_layers', 1),
    )
    ckpts = sorted(MODEL_DIR.glob('checkpoints/**/best-epoch*.ckpt'))
    ckpt  = torch.load(ckpts[-1], map_location='cpu')
    state = {k.removeprefix('model.'): v
             for k, v in ckpt['state_dict'].items() if k.startswith('model.')}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), mp['output_shape']


def get_attention_weights(model, features, n_out, device):
    with torch.no_grad():
        inp = features.unsqueeze(0).to(device)
        try:
            att = model.calculate_attention(inp)
        except TypeError:
            att = model.calculate_attention(inp, apply_softmax=False)
    att = torch.squeeze(att).cpu().float().numpy()
    if att.ndim == 2:
        att = att.mean(axis=-1)
    n = features.shape[0]
    if att.shape[0] == n * n_out and n_out > 1:
        att = att.reshape(n, n_out).mean(axis=1)
    att = att - att.min()
    att = att / (att.sum() + 1e-8)
    return att


# ── Carga de features ─────────────────────────────────────────────────────────

def load_slide_vector(slide, weighted, model, n_out, device):
    bag_path = BAGS_DIR / f'{slide}.pt'
    if not bag_path.exists():
        return None
    features = torch.load(bag_path, map_location='cpu', weights_only=True).float()
    if weighted and model is not None:
        w = get_attention_weights(model, features, n_out, device)
        vec = (features.numpy() * w[:, None]).sum(axis=0)
    else:
        vec = features.numpy().mean(axis=0)
    return vec.astype(np.float32)


# ── Reducción de dimensionalidad ──────────────────────────────────────────────

def reduce_2d(X, method, seed):
    if method == 'umap':
        import umap as umap_lib
        reducer = umap_lib.UMAP(n_components=2, random_state=seed,
                                n_neighbors=30, min_dist=0.1, metric='cosine')
    else:
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        # PCA pre-reduction para velocidad
        if X.shape[1] > 50:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=50, random_state=seed)
            X = pca.fit_transform(X)
        reducer = TSNE(n_components=2, random_state=seed, perplexity=30,
                       n_iter=1000, metric='cosine', init='pca')
    return reducer.fit_transform(X)


# ── Figuras ───────────────────────────────────────────────────────────────────

def plot_by_dataset(emb, meta, out_dir, method):
    fig, ax = plt.subplots(figsize=(8, 6))
    for ds in DATASET_LABELS:
        mask = meta['dataset'] == ds
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=DATASET_COLORS[ds], label=DATASET_LABELS[ds],
                   s=30, alpha=0.75, edgecolors='none')
    ax.legend(framealpha=0.9, fontsize=11)
    ax.set_xlabel(f'{method.upper()} 1', fontsize=12)
    ax.set_ylabel(f'{method.upper()} 2', fontsize=12)
    ax.set_title('Espacio de features por base de datos', fontsize=13)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    fig.tight_layout()
    fig.savefig(out_dir / f'feature_space_by_dataset_{method}.pdf', dpi=150)
    fig.savefig(out_dir / f'feature_space_by_dataset_{method}.png', dpi=150)
    plt.close(fig)
    print(f'  Guardado: feature_space_by_dataset_{method}')


def plot_by_class(emb, meta, out_dir, method):
    fig, ax = plt.subplots(figsize=(8, 6))
    for cls_id in [0, 1]:
        mask = meta['category'] == cls_id
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=CLASS_COLORS[cls_id], marker=CLASS_MARKERS[cls_id],
                   label=CLASS_NAMES[cls_id], s=30, alpha=0.75, edgecolors='none')
    ax.legend(framealpha=0.9, fontsize=11)
    ax.set_xlabel(f'{method.upper()} 1', fontsize=12)
    ax.set_ylabel(f'{method.upper()} 2', fontsize=12)
    ax.set_title('Espacio de features por etiqueta MSS/MSI', fontsize=13)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    fig.tight_layout()
    fig.savefig(out_dir / f'feature_space_by_class_{method}.pdf', dpi=150)
    fig.savefig(out_dir / f'feature_space_by_class_{method}.png', dpi=150)
    plt.close(fig)
    print(f'  Guardado: feature_space_by_class_{method}')


def plot_combined(emb, meta, out_dir, method):
    """Dataset como color, MSS/MSI como forma del marcador."""
    fig, ax = plt.subplots(figsize=(9, 6))
    for ds in DATASET_LABELS:
        for cls_id in [0, 1]:
            mask = (meta['dataset'] == ds) & (meta['category'] == cls_id)
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       c=DATASET_COLORS[ds], marker=CLASS_MARKERS[cls_id],
                       s=35, alpha=0.75, edgecolors='none')

    ds_handles = [mpatches.Patch(color=DATASET_COLORS[ds], label=DATASET_LABELS[ds])
                  for ds in DATASET_LABELS]
    cls_handles = [Line2D([0], [0], marker=CLASS_MARKERS[c], color='gray',
                          linestyle='None', markersize=8, label=CLASS_NAMES[c])
                   for c in [0, 1]]
    leg1 = ax.legend(handles=ds_handles, loc='upper left', fontsize=10,
                     title='Base de datos', framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=cls_handles, loc='upper right', fontsize=10,
              title='Etiqueta', framealpha=0.9)
    ax.set_xlabel(f'{method.upper()} 1', fontsize=12)
    ax.set_ylabel(f'{method.upper()} 2', fontsize=12)
    ax.set_title('Espacio de features: base de datos × MSS/MSI', fontsize=13)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    fig.tight_layout()
    fig.savefig(out_dir / f'feature_space_combined_{method}.pdf', dpi=150)
    fig.savefig(out_dir / f'feature_space_combined_{method}.png', dpi=150)
    plt.close(fig)
    print(f'  Guardado: feature_space_combined_{method}')


def plot_subplots_by_dataset(emb, meta, out_dir, method):
    """Un subplot por base de datos, coloreado por MSS/MSI."""
    datasets = list(DATASET_LABELS.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

    # fondo gris con todos los puntos para contexto
    for ax, ds in zip(axes, datasets):
        ax.scatter(emb[:, 0], emb[:, 1], c='#DDDDDD', s=15, alpha=0.4,
                   edgecolors='none', zorder=1)
        mask_ds = meta['dataset'] == ds
        for cls_id in [0, 1]:
            mask = mask_ds & (meta['category'] == cls_id)
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       c=CLASS_COLORS[cls_id], marker=CLASS_MARKERS[cls_id],
                       s=40, alpha=0.85, edgecolors='none', zorder=2,
                       label=CLASS_NAMES[cls_id])
        n_mss = (mask_ds & (meta['category'] == 0)).sum()
        n_msi = (mask_ds & (meta['category'] == 1)).sum()
        ax.set_title(f'{DATASET_LABELS[ds]}\n(MSS={n_mss}, MSI={n_msi})', fontsize=12)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.legend(fontsize=10, framealpha=0.9)

    axes[0].set_ylabel(f'{method.upper()} 2', fontsize=11)
    fig.text(0.5, 0.02, f'{method.upper()} 1', ha='center', fontsize=11)
    fig.suptitle('Espacio de features por base de datos (MSS/MSI)', fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / f'feature_space_subplots_{method}.pdf', dpi=150,
                bbox_inches='tight')
    fig.savefig(out_dir / f'feature_space_subplots_{method}.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print(f'  Guardado: feature_space_subplots_{method}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['umap', 'tsne'], default='umap')
    parser.add_argument('--weighted', action='store_true',
                        help='Usa media ponderada por atención en lugar de media simple')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Cargar modelo si se usa media ponderada
    model, n_out = None, 2
    if args.weighted:
        print('Cargando modelo para cálculo de atención...')
        model, n_out = load_model_for_attention(device)

    # Cargar anotaciones
    ann = pd.read_csv(ANNOT)

    # Cargar features por slide
    print(f'\nCargando features ({len(ann)} slides)...')
    vecs, meta_rows = [], []
    skipped = 0
    for _, row in ann.iterrows():
        vec = load_slide_vector(row['slide'], args.weighted, model, n_out, device)
        if vec is None:
            skipped += 1
            continue
        vecs.append(vec)
        meta_rows.append({'slide': row['slide'],
                          'dataset': row['dataset'],
                          'category': int(row['category'])})

    X    = np.stack(vecs)
    meta = pd.DataFrame(meta_rows)
    print(f'  Cargados: {len(X)}  |  Omitidos (sin bag): {skipped}')
    print(f'  Shape matriz features: {X.shape}')
    for ds in DATASET_LABELS:
        n = (meta['dataset'] == ds).sum()
        print(f'  {DATASET_LABELS[ds]}: {n} slides')

    # Reducción 2D
    print(f'\nReduciendo a 2D con {args.method.upper()} (seed={args.seed})...')
    emb = reduce_2d(X, args.method, args.seed)
    print(f'  Embedding shape: {emb.shape}')

    # Guardar embedding
    emb_df = meta.copy()
    emb_df['x'] = emb[:, 0]
    emb_df['y'] = emb[:, 1]
    suffix = f'{"weighted_" if args.weighted else ""}{args.method}'
    emb_df.to_csv(OUT_DIR / f'embedding_{suffix}.csv', index=False)

    # Gráficas
    print('\nGenerando gráficas...')
    plot_by_dataset(emb, meta, OUT_DIR, args.method)
    plot_by_class(emb, meta, OUT_DIR, args.method)
    plot_combined(emb, meta, OUT_DIR, args.method)
    plot_subplots_by_dataset(emb, meta, OUT_DIR, args.method)

    print(f'\nListo. Resultados en: {OUT_DIR}')


if __name__ == '__main__':
    main()