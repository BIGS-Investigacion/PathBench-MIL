#!/usr/bin/env python3
"""
plot_feature_space_selected.py

Visualiza en 2D el espacio de features h-optimus-0 de las tres bases de datos
usando únicamente las dimensiones seleccionadas a partir de Macarena (MSS vs MSI).

Pipeline:
  1. Carga los bags de todos los slides → media por slide (708 × 1536)
  2. Selección de características supervisada sobre Macarena:
       --selector anova   : ANOVA F-score entre MSS y MSI (SelectKBest + f_classif)
       --selector mi      : Información mutua (SelectKBest + mutual_info_classif)
       --selector lda     : LDA → proyecta en el subespacio más discriminante (1 dim → usa PCA primero)
       --selector pca_sup : PCA sobre la diferencia de medias de clase (PCA supervisada simple)
  3. Proyecta los tres datasets sobre las dimensiones/transformación aprendida en Macarena
  4. UMAP o t-SNE sobre el espacio reducido
  5. Genera las mismas 4 figuras que plot_feature_space.py

Uso:
  python scripts/colon/plot_feature_space_selected.py [--method umap|tsne]
                                                       [--selector anova|mi|lda|pca_sup]
                                                       [--top_k 256]
                                                       [--seed 42]
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
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ── Rutas ─────────────────────────────────────────────────────────────────────

ROOT     = Path(__file__).resolve().parents[2]
BAGS_DIR = Path('/home/PARADIS/colon/tfrecords/shared_bags/256_128_none_h_optimus_0')
ANNOT    = ROOT / 'config/annotations/annotations_colon_mss_msi_all.csv'
OUT_DIR  = ROOT / 'results/colon/feature_space_selected'

DATASET_COLORS  = {'macarena': '#2196F3', 'cptac_coad': '#FF9800', 'tcga_coad': '#4CAF50'}
DATASET_LABELS  = {'macarena': 'Macarena', 'cptac_coad': 'CPTAC-COAD', 'tcga_coad': 'TCGA-COAD'}
CLASS_COLORS    = {0: '#E53935', 1: '#1E88E5'}
CLASS_NAMES     = {0: 'MSS', 1: 'MSI'}
CLASS_MARKERS   = {0: 'o', 1: '^'}


# ── Carga de features ─────────────────────────────────────────────────────────

def load_all_features(ann: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    vecs, meta_rows = [], []
    for _, row in ann.iterrows():
        bag_path = BAGS_DIR / f'{row["slide"]}.pt'
        if not bag_path.exists():
            continue
        feat = torch.load(bag_path, map_location='cpu', weights_only=True).float().numpy()
        vecs.append(feat.mean(axis=0))
        meta_rows.append({'slide': row['slide'],
                          'dataset': row['dataset'],
                          'category': int(row['category'])})
    return np.stack(vecs).astype(np.float32), pd.DataFrame(meta_rows)


# ── Selección de características sobre Macarena ───────────────────────────────

def select_features(X_mac: np.ndarray, y_mac: np.ndarray,
                    X_all: np.ndarray, selector: str, top_k: int, seed: int):
    """
    Aprende la transformación/selección sobre Macarena y la aplica a todos los datasets.
    Devuelve X_all transformado y una descripción de la reducción.
    """
    print(f'\nSelección de características ({selector.upper()}, k={top_k}) sobre Macarena...')

    if selector in ('anova', 'mi'):
        score_fn = f_classif if selector == 'anova' else mutual_info_classif
        sel = SelectKBest(score_fn, k=top_k)
        sel.fit(X_mac, y_mac)
        scores = sel.scores_
        selected_idx = np.argsort(scores)[::-1][:top_k]
        X_out = X_all[:, selected_idx]
        desc = (f'{selector.upper()}: {top_k} features más discriminativas de {X_mac.shape[1]}\n'
                f'Score máximo: {scores[selected_idx[0]]:.2f}  |  mínimo seleccionado: {scores[selected_idx[-1]]:.2f}')
        # Guardar scores para inspección
        score_df = pd.DataFrame({'feature_idx': np.arange(len(scores)), 'score': scores})
        score_df = score_df.sort_values('score', ascending=False)
        score_df.to_csv(OUT_DIR / f'feature_scores_{selector}.csv', index=False)

    elif selector == 'lda':
        # LDA binaria produce 1 componente; hacemos PCA primero para reducir colinealidad
        n_pca = min(top_k, X_mac.shape[0] - 1, X_mac.shape[1])
        pca = PCA(n_components=n_pca, random_state=seed)
        scaler = StandardScaler()
        X_mac_s = scaler.fit_transform(X_mac)
        X_mac_pca = pca.fit_transform(X_mac_s)
        lda = LinearDiscriminantAnalysis(n_components=1)
        lda.fit(X_mac_pca, y_mac)
        X_all_s   = scaler.transform(X_all)
        X_all_pca = pca.transform(X_all_s)
        X_out = lda.transform(X_all_pca)          # (N, 1) — sirve como eje de separación
        # Añadir segunda componente PCA para visualización 2D
        X_out = np.hstack([X_out, X_all_pca[:, 1:2]])
        desc = (f'LDA: PCA({n_pca}) → LDA(1). '
                f'Varianza explicada PCA: {pca.explained_variance_ratio_[:n_pca].sum():.2%}')

    elif selector == 'pca_sup':
        # PCA supervisada: PCA sobre la diferencia de medias de clase + varianza within-class
        scaler = StandardScaler()
        X_mac_s = scaler.fit_transform(X_mac)
        # Centrado por clase (within-class covariance)
        classes = np.unique(y_mac)
        X_centered = X_mac_s.copy()
        for c in classes:
            mask = y_mac == c
            X_centered[mask] -= X_mac_s[mask].mean(axis=0)
        pca = PCA(n_components=top_k, random_state=seed)
        pca.fit(X_centered)
        X_out = pca.transform(scaler.transform(X_all))
        desc = (f'PCA supervisada (within-class): {top_k} componentes\n'
                f'Varianza explicada: {pca.explained_variance_ratio_[:top_k].sum():.2%}')

    else:
        raise ValueError(f'Selector desconocido: {selector}')

    print(f'  {desc}')
    print(f'  Shape tras selección: {X_out.shape}')
    return X_out, desc


# ── Reducción 2D ──────────────────────────────────────────────────────────────

def reduce_2d(X: np.ndarray, method: str, seed: int) -> np.ndarray:
    if method == 'umap':
        import umap as umap_lib
        reducer = umap_lib.UMAP(n_components=2, random_state=seed,
                                n_neighbors=30, min_dist=0.1, metric='cosine')
    else:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=seed, perplexity=30,
                       n_iter=1000, metric='cosine', init='pca')
    return reducer.fit_transform(X)


# ── Figuras ───────────────────────────────────────────────────────────────────

def _scatter(ax, emb, mask, color, marker, label, **kw):
    ax.scatter(emb[mask, 0], emb[mask, 1], c=color, marker=marker,
               label=label, s=35, alpha=0.80, edgecolors='none', **kw)


def plot_by_dataset(emb, meta, tag, out_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    for ds in DATASET_LABELS:
        _scatter(ax, emb, meta['dataset'] == ds, DATASET_COLORS[ds], 'o', DATASET_LABELS[ds])
    ax.legend(framealpha=0.9, fontsize=11)
    _style(ax, tag, 'Espacio de features por base de datos')
    _save(fig, out_dir, f'by_dataset_{tag}')


def plot_by_class(emb, meta, tag, out_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    for cls_id in [0, 1]:
        _scatter(ax, emb, meta['category'] == cls_id,
                 CLASS_COLORS[cls_id], CLASS_MARKERS[cls_id], CLASS_NAMES[cls_id])
    ax.legend(framealpha=0.9, fontsize=11)
    _style(ax, tag, 'Espacio de features por etiqueta MSS/MSI')
    _save(fig, out_dir, f'by_class_{tag}')


def plot_combined(emb, meta, tag, out_dir):
    fig, ax = plt.subplots(figsize=(9, 6))
    for ds in DATASET_LABELS:
        for cls_id in [0, 1]:
            mask = (meta['dataset'] == ds) & (meta['category'] == cls_id)
            _scatter(ax, emb, mask, DATASET_COLORS[ds], CLASS_MARKERS[cls_id], None)
    ds_handles  = [mpatches.Patch(color=DATASET_COLORS[ds], label=DATASET_LABELS[ds])
                   for ds in DATASET_LABELS]
    cls_handles = [Line2D([0], [0], marker=CLASS_MARKERS[c], color='gray',
                          linestyle='None', markersize=8, label=CLASS_NAMES[c])
                   for c in [0, 1]]
    leg1 = ax.legend(handles=ds_handles, loc='upper left', fontsize=10,
                     title='Base de datos', framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=cls_handles, loc='upper right', fontsize=10,
              title='Etiqueta', framealpha=0.9)
    _style(ax, tag, 'Espacio de features: base de datos × MSS/MSI')
    _save(fig, out_dir, f'combined_{tag}')


def plot_subplots(emb, meta, tag, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    for ax, ds in zip(axes, DATASET_LABELS):
        ax.scatter(emb[:, 0], emb[:, 1], c='#DDDDDD', s=12, alpha=0.35,
                   edgecolors='none', zorder=1)
        mask_ds = meta['dataset'] == ds
        for cls_id in [0, 1]:
            mask = mask_ds & (meta['category'] == cls_id)
            _scatter(ax, emb, mask, CLASS_COLORS[cls_id], CLASS_MARKERS[cls_id],
                     CLASS_NAMES[cls_id], zorder=2)
        n_mss = int((mask_ds & (meta['category'] == 0)).sum())
        n_msi = int((mask_ds & (meta['category'] == 1)).sum())
        ax.set_title(f'{DATASET_LABELS[ds]}\n(MSS={n_mss}, MSI={n_msi})', fontsize=12)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.legend(fontsize=10, framealpha=0.9)
    fig.text(0.5, 0.02, f'{tag.split("_")[0].upper()} 1', ha='center', fontsize=11)
    axes[0].set_ylabel(f'{tag.split("_")[0].upper()} 2', fontsize=11)
    fig.suptitle('Espacio de features por base de datos (MSS/MSI) — selección Macarena',
                 fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, out_dir, f'subplots_{tag}')


def _style(ax, tag, title):
    method = tag.split('_')[0].upper()
    ax.set_xlabel(f'{method} 1', fontsize=12)
    ax.set_ylabel(f'{method} 2', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


def _save(fig, out_dir, name):
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(out_dir / f'{name}.{ext}', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Guardado: {name}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method',   choices=['umap', 'tsne'], default='umap')
    parser.add_argument('--selector', choices=['anova', 'mi', 'lda', 'pca_sup'],
                        default='anova',
                        help='anova: F-score | mi: información mutua | '
                             'lda: análisis discriminante lineal | pca_sup: PCA supervisada')
    parser.add_argument('--top_k', type=int, default=256,
                        help='Número de features a seleccionar (no aplica a lda)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Cargar features de todos los slides
    print('Cargando features...')
    ann  = pd.read_csv(ANNOT)
    X_all, meta = load_all_features(ann)
    print(f'  Total slides: {len(X_all)}  |  Features: {X_all.shape[1]}')

    # 2. Separar Macarena para selección
    mac_mask  = meta['dataset'] == 'macarena'
    X_mac     = X_all[mac_mask.values]
    y_mac     = meta.loc[mac_mask, 'category'].values
    print(f'  Macarena: {len(X_mac)} slides  (MSS={( y_mac==0).sum()}, MSI={(y_mac==1).sum()})')

    # 3. Selección de características
    X_sel, desc = select_features(X_mac, y_mac, X_all, args.selector, args.top_k, args.seed)

    # 4. Reducción 2D
    print(f'\nReduciendo a 2D con {args.method.upper()}...')
    emb = reduce_2d(X_sel, args.method, args.seed)

    # Guardar embedding
    tag = f'{args.method}_{args.selector}_k{args.top_k}'
    emb_df = meta.copy()
    emb_df['x'] = emb[:, 0]
    emb_df['y'] = emb[:, 1]
    emb_df.to_csv(OUT_DIR / f'embedding_{tag}.csv', index=False)

    # 5. Figuras
    print('\nGenerando figuras...')
    plot_by_dataset(emb, meta, tag, OUT_DIR)
    plot_by_class(emb, meta, tag, OUT_DIR)
    plot_combined(emb, meta, tag, OUT_DIR)
    plot_subplots(emb, meta, tag, OUT_DIR)

    # Guardar descripción del experimento
    with open(OUT_DIR / f'config_{tag}.txt', 'w') as f:
        f.write(f'method:   {args.method}\n')
        f.write(f'selector: {args.selector}\n')
        f.write(f'top_k:    {args.top_k}\n')
        f.write(f'seed:     {args.seed}\n\n')
        f.write(desc + '\n')

    print(f'\nListo. Resultados en: {OUT_DIR}')


if __name__ == '__main__':
    main()