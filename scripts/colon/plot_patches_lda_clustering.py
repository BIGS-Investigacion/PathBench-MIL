#!/usr/bin/env python3
"""
plot_patches_lda_clustering.py

Visualiza en 2D los top-8 patches por slide (seleccionados por atención)
usando únicamente las features más discriminativas de Macarena (ANOVA MSS vs MSI).

Pipeline:
  1. Carga los top-8 patches de cada slide → ~5666 puntos × 1536 features
  2. ANOVA F-score sobre Macarena: selecciona las top_k features más discriminativas
  3. Proyecta los tres datasets sobre ese subespacio
  4. Dos visualizaciones:
       A) UMAP sobre las features seleccionadas
       B) PCA(2) directa sobre las features seleccionadas (sin reducción adicional)

Cada figura se genera en dos variantes:
  - Coloreado por base de datos
  - Coloreado por etiqueta MSS/MSI

Uso:
  python scripts/colon/plot_patches_lda_clustering.py [--top_k 128] [--seed 42]
"""

import argparse
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
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

ROOT     = Path(__file__).resolve().parents[2]
BAGS_DIR = Path('/home/PARADIS/colon/tfrecords/shared_bags/256_128_none_h_optimus_0')
OUT_DIR  = ROOT / 'results/colon/patches_feature_space'

DATASETS  = ['macarena', 'cptac_coad', 'tcga_coad']
DS_LABEL  = {'macarena': 'Macarena', 'cptac_coad': 'CPTAC-COAD', 'tcga_coad': 'TCGA-COAD'}
DS_COLOR  = {'macarena': '#2196F3',  'cptac_coad': '#FF9800',    'tcga_coad': '#4CAF50'}
CLS_COLOR = {0: '#E53935', 1: '#1E88E5'}
CLS_NAME  = {0: 'MSS', 1: 'MSI'}
CLS_MARK  = {0: 'o', 1: '^'}


# ── Carga ─────────────────────────────────────────────────────────────────────

def load_top8_features(dataset: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Carga los features de los top-8 patches por slide."""
    df = pd.read_csv(ROOT / f'results/colon/top8_patches_{dataset}.csv')

    bag_cache, coord_cache = {}, {}
    vecs, rows = [], []

    for _, row in df.iterrows():
        slide = row['slide']
        x, y  = int(row['x']), int(row['y'])

        if slide not in bag_cache:
            bp = BAGS_DIR / f'{slide}.pt'
            ip = BAGS_DIR / f'{slide}.index.npz'
            if not bp.exists() or not ip.exists():
                continue
            bag_cache[slide]   = torch.load(bp, map_location='cpu',
                                            weights_only=True).numpy().astype(np.float32)
            coord_cache[slide] = np.load(ip)['arr_0']

        coords = coord_cache[slide]
        hits   = np.where((coords[:, 0] == x) & (coords[:, 1] == y))[0]
        if len(hits) == 0:
            continue

        vecs.append(bag_cache[slide][hits[0]])
        rows.append({'slide': slide, 'dataset': dataset,
                     'class_id': int(row['class_id']),
                     'rank': int(row['rank']),
                     'attention': float(row['attention'])})

    return np.stack(vecs).astype(np.float32), pd.DataFrame(rows)


# ── Feature selection sobre Macarena ─────────────────────────────────────────

def select_from_macarena(X_all: np.ndarray, meta: pd.DataFrame,
                         top_k: int) -> np.ndarray:
    """Selecciona las top_k features por ANOVA F-score en Macarena (MSS vs MSI)."""
    mac   = meta['dataset'] == 'macarena'
    X_mac = X_all[mac.values]
    y_mac = meta.loc[mac, 'class_id'].values

    sel = SelectKBest(f_classif, k=top_k)
    sel.fit(X_mac, y_mac)

    idx    = np.argsort(sel.scores_)[::-1][:top_k]
    scores = sel.scores_[idx]
    print(f'  F-score — máx: {scores[0]:.1f}  |  mín seleccionado: {scores[-1]:.1f}')

    pd.DataFrame({'feature_idx': idx, 'f_score': scores}).to_csv(
        OUT_DIR / 'feature_scores_anova.csv', index=False)

    return X_all[:, idx]


# ── Figuras ───────────────────────────────────────────────────────────────────

def _make_fig_dataset(emb, meta, xlabel, ylabel, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    for ds in DATASETS:
        mask = (meta['dataset'] == ds).values
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=DS_COLOR[ds], s=12, alpha=0.65, edgecolors='none',
                   label=DS_LABEL[ds])
    ax.legend(fontsize=11, framealpha=0.9)
    ax.set_xlabel(xlabel, fontsize=12); ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.tick_params(labelleft=False, labelbottom=False)
    return fig


def _make_fig_class(emb, meta, xlabel, ylabel, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    for cls_id in [0, 1]:
        mask = (meta['class_id'] == cls_id).values
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=CLS_COLOR[cls_id], marker=CLS_MARK[cls_id],
                   s=12, alpha=0.65, edgecolors='none', label=CLS_NAME[cls_id])
    ax.legend(fontsize=11, framealpha=0.9)
    ax.set_xlabel(xlabel, fontsize=12); ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.tick_params(labelleft=False, labelbottom=False)
    return fig


def _make_fig_subplots(emb, meta, xlabel, ylabel, title):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    for ax, ds in zip(axes, DATASETS):
        mask_ds = (meta['dataset'] == ds).values
        ax.scatter(emb[~mask_ds, 0], emb[~mask_ds, 1],
                   c='#DDDDDD', s=8, alpha=0.25, edgecolors='none', zorder=1)
        for cls_id in [0, 1]:
            mask = mask_ds & (meta['class_id'] == cls_id).values
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       c=CLS_COLOR[cls_id], marker=CLS_MARK[cls_id],
                       s=14, alpha=0.75, edgecolors='none', zorder=2,
                       label=CLS_NAME[cls_id])
        n_mss = int((mask_ds & (meta['class_id'] == 0).values).sum())
        n_msi = int((mask_ds & (meta['class_id'] == 1).values).sum())
        ax.set_title(f'{DS_LABEL[ds]}\n(MSS={n_mss}, MSI={n_msi})', fontsize=12)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.tick_params(labelleft=False)
        ax.legend(fontsize=9, framealpha=0.9)
    axes[0].set_ylabel(ylabel, fontsize=10)
    fig.suptitle(title, fontsize=13, y=1.01)
    return fig


def save_all(emb, meta, xlabel, ylabel, prefix, out_dir):
    figs = [
        (_make_fig_dataset(emb, meta, xlabel, ylabel,
                           f'{prefix} — coloreado por base de datos'),
         f'{prefix}_dataset'),
        (_make_fig_class(emb, meta, xlabel, ylabel,
                         f'{prefix} — coloreado por MSS/MSI'),
         f'{prefix}_class'),
        (_make_fig_subplots(emb, meta, xlabel, ylabel,
                            f'{prefix} — subplots por base de datos'),
         f'{prefix}_subplots'),
    ]
    for fig, name in figs:
        fig.tight_layout()
        for ext in ('pdf', 'png'):
            fig.savefig(out_dir / f'{name}.{ext}', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Guardado: {name}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_k', type=int, default=128,
                        help='Número de features seleccionadas por ANOVA en Macarena')
    parser.add_argument('--seed',  type=int, default=42)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f'k{args.top_k}_s{args.seed}'

    # 1. Cargar top-8 patches
    print('Cargando top-8 patches...')
    all_X, all_meta = [], []
    for ds in DATASETS:
        X_ds, m_ds = load_top8_features(ds)
        all_X.append(X_ds)
        all_meta.append(m_ds)
        print(f'  {DS_LABEL[ds]}: {len(X_ds)} patches  '
              f'(MSS={(m_ds["class_id"]==0).sum()}, MSI={(m_ds["class_id"]==1).sum()})')

    X_all = np.vstack(all_X)
    meta  = pd.concat(all_meta, ignore_index=True)
    print(f'  Total: {len(X_all)} patches  |  features: {X_all.shape[1]}')

    # 2. Feature selection sobre Macarena
    print(f'\nSelección ANOVA top-{args.top_k} sobre Macarena...')
    X_sel = select_from_macarena(X_all, meta, args.top_k)
    print(f'  Shape tras selección: {X_sel.shape}')

    # 3A. UMAP
    print('\nA) UMAP...')
    import umap as umap_lib
    emb_umap = umap_lib.UMAP(n_components=2, random_state=args.seed,
                              n_neighbors=20, min_dist=0.1,
                              metric='cosine').fit_transform(X_sel)

    save_all(emb_umap, meta, 'UMAP 1', 'UMAP 2',
             f'umap_{tag}', OUT_DIR)

    # 3B. PCA directa
    print('\nB) PCA directa...')
    X_scaled = StandardScaler().fit_transform(X_sel)
    pca = PCA(n_components=2, random_state=args.seed)
    emb_pca = pca.fit_transform(X_scaled)
    v1, v2  = pca.explained_variance_ratio_
    print(f'  Varianza explicada: PC1={v1:.1%}  PC2={v2:.1%}')

    save_all(emb_pca, meta, f'PC1 ({v1:.1%})', f'PC2 ({v2:.1%})',
             f'pca_{tag}', OUT_DIR)

    # Guardar embedding
    emb_df = meta.copy()
    emb_df[['umap1', 'umap2']] = emb_umap
    emb_df[['pc1',   'pc2']]   = emb_pca
    emb_df.to_csv(OUT_DIR / f'embeddings_{tag}.csv', index=False)

    print(f'\nListo. Resultados en: {OUT_DIR}')


if __name__ == '__main__':
    main()