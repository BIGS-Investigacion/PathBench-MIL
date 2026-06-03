#!/usr/bin/env python3
"""
eval_coral_alignment.py

Evalúa la alineación de dominio CORAL sobre las features h-optimus-0
de las tres bases de datos (Macarena, CPTAC-COAD, TCGA-COAD).

Pipeline:
  1. Carga features de todos los slides (media por slide) → 708 × 1536
  2. StandardScaler + PCA a n_pca componentes (evita rango deficiente en CORAL)
  3. CORAL estándar: alinea CPTAC y TCGA a Macarena (distribución completa)
  4. CORAL condicional: alinea MSS↔MSS y MSI↔MSI por separado entre dominios
  5. Figura 2×3: sin alineación / CORAL estándar / CORAL condicional
     × coloreado por dominio / por clase
  6. Métricas cuantitativas (LR 5-fold CV):
       - Domain accuracy     — debería BAJAR
       - Class balanced-acc  — debería SUBIR o mantenerse

Uso:
  python scripts/colon/eval_coral_alignment.py [--n_pca 128] [--alpha 1.0] [--seed 42]
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

warnings.filterwarnings('ignore')

ROOT     = Path(__file__).resolve().parents[2]
BAGS_DIR = Path('/home/PARADIS/colon/tfrecords/shared_bags/256_128_none_h_optimus_0')
ANNOT    = ROOT / 'config/annotations/annotations_colon_mss_msi_all.csv'
OUT_DIR  = ROOT / 'results/colon/coral_alignment'

DATASET_COLORS = {'macarena': '#2196F3', 'cptac_coad': '#FF9800', 'tcga_coad': '#4CAF50'}
DATASET_LABELS = {'macarena': 'Macarena', 'cptac_coad': 'CPTAC-COAD', 'tcga_coad': 'TCGA-COAD'}
CLASS_COLORS   = {0: '#E53935', 1: '#1E88E5'}
CLASS_NAMES    = {0: 'MSS', 1: 'MSI'}
CLASS_MARKERS  = {0: 'o', 1: '^'}


# ── Carga ─────────────────────────────────────────────────────────────────────

def load_features(ann: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
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

def coral(X_src: np.ndarray, X_tgt: np.ndarray, alpha: float) -> np.ndarray:
    """
    Alinea la covarianza de X_src a la de X_tgt.
    Whitening (C_src^{-1/2}) seguido de re-coloring (C_tgt^{1/2}).
    alpha: regularización diagonal para evitar singularidades.
    """
    mu_s = X_src.mean(axis=0)
    mu_t = X_tgt.mean(axis=0)
    Xs = X_src - mu_s
    Xt = X_tgt - mu_t

    d  = X_src.shape[1]
    Cs = np.cov(Xs.T) + alpha * np.eye(d)
    Ct = np.cov(Xt.T) + alpha * np.eye(d)

    Us, Ss, _ = np.linalg.svd(Cs)
    W_white   = Us @ np.diag(1.0 / np.sqrt(np.maximum(Ss, 1e-10))) @ Us.T

    Ut, St, _ = np.linalg.svd(Ct)
    W_color   = Ut @ np.diag(np.sqrt(np.maximum(St, 0.0))) @ Ut.T

    return Xs @ W_white @ W_color + mu_t


def apply_coral(X: np.ndarray, meta: pd.DataFrame, alpha: float) -> np.ndarray:
    """CORAL estándar: alinea CPTAC y TCGA a Macarena; Macarena queda sin cambios."""
    X_out = X.copy()
    X_mac = X[(meta['dataset'] == 'macarena').values]
    for ds in ['cptac_coad', 'tcga_coad']:
        mask = (meta['dataset'] == ds).values
        X_out[mask] = coral(X[mask], X_mac, alpha)
        print(f'  {DATASET_LABELS[ds]} alineado ({mask.sum()} slides)')
    return X_out


def apply_coral_conditional(X: np.ndarray, meta: pd.DataFrame, alpha: float) -> np.ndarray:
    """
    CORAL condicional: alinea MSS↔MSS y MSI↔MSI por separado entre dominios.
    Preserva la estructura de clase al alinear dentro de cada etiqueta.
    Usa regularización adaptativa cuando n_samples < n_features.
    """
    X_out = X.copy()
    d = X.shape[1]

    for ds in ['cptac_coad', 'tcga_coad']:
        for cls_id in [0, 1]:
            mask_src = ((meta['dataset'] == ds) & (meta['category'] == cls_id)).values
            mask_tgt = ((meta['dataset'] == 'macarena') & (meta['category'] == cls_id)).values

            n_src, n_tgt = mask_src.sum(), mask_tgt.sum()
            if n_src < 5 or n_tgt < 5:
                print(f'  Skip {DATASET_LABELS[ds]} {CLASS_NAMES[cls_id]}: '
                      f'n_src={n_src}, n_tgt={n_tgt}')
                continue

            # Regularización adaptativa: escala con la deficiencia de rango
            alpha_eff = alpha * max(1.0, d / min(n_src, n_tgt))
            X_out[mask_src] = coral(X[mask_src], X[mask_tgt], alpha_eff)
            print(f'  {DATASET_LABELS[ds]} {CLASS_NAMES[cls_id]:3s}: '
                  f'{n_src} slides → {n_tgt} ref  (alpha_eff={alpha_eff:.1f})')

    return X_out


# ── Métricas ──────────────────────────────────────────────────────────────────

def evaluate(X: np.ndarray, meta: pd.DataFrame, label: str, cv: int = 5):
    """
    Regresión logística 5-fold:
      - domain_acc    : predecir base de datos (queremos que BAJE tras CORAL)
      - class_bal_acc : predecir MSS/MSI      (queremos que SUBA o se mantenga)
    """
    le       = LabelEncoder()
    y_domain = le.fit_transform(meta['dataset'])
    y_class  = meta['category'].values

    lr = LogisticRegression(max_iter=2000, random_state=0, C=1.0)
    acc_d = cross_val_score(lr, X, y_domain, cv=cv, scoring='accuracy').mean()
    acc_c = cross_val_score(lr, X, y_class,  cv=cv, scoring='balanced_accuracy').mean()

    print(f'  [{label:15s}]  domain acc: {acc_d:.3f}  |  class bal-acc: {acc_c:.3f}')
    return acc_d, acc_c


# ── UMAP ──────────────────────────────────────────────────────────────────────

def run_umap(X: np.ndarray, seed: int) -> np.ndarray:
    import umap as umap_lib
    return umap_lib.UMAP(n_components=2, random_state=seed,
                         n_neighbors=30, min_dist=0.1,
                         metric='cosine').fit_transform(X)


# ── Figuras ───────────────────────────────────────────────────────────────────

def plot_comparison(embs: list, titles: list, metrics: list, meta, tag, out_dir):
    """Figura 2×N: filas = por dominio / por clase; columnas = variantes de alineación."""
    n = len(embs)
    fig, axes = plt.subplots(2, n, figsize=(6 * n, 11))

    for col, (emb, title, m) in enumerate(zip(embs, titles, metrics)):
        d_acc, c_acc = m
        subtitle = f'domain acc={d_acc:.3f} | class bal-acc={c_acc:.3f}'

        # Fila 0: por base de datos
        ax = axes[0, col]
        for ds in DATASET_LABELS:
            mask = (meta['dataset'] == ds).values
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       c=DATASET_COLORS[ds], s=22, alpha=0.75,
                       edgecolors='none', label=DATASET_LABELS[ds])
        if col == 0:
            ax.legend(fontsize=8, framealpha=0.9)
        ax.set_title(f'{title}\npor base de datos\n{subtitle}', fontsize=9)
        ax.tick_params(labelleft=False, labelbottom=False)

        # Fila 1: por clase
        ax = axes[1, col]
        for cls_id in [0, 1]:
            mask = (meta['category'] == cls_id).values
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       c=CLASS_COLORS[cls_id], marker=CLASS_MARKERS[cls_id],
                       s=22, alpha=0.75, edgecolors='none',
                       label=CLASS_NAMES[cls_id])
        if col == 0:
            ax.legend(fontsize=8, framealpha=0.9)
        ax.set_title(f'{title}\npor MSS/MSI', fontsize=9)
        ax.tick_params(labelleft=False, labelbottom=False)

    for ax in axes.flat:
        ax.set_xlabel('UMAP 1', fontsize=9)
    axes[0, 0].set_ylabel('UMAP 2', fontsize=9)
    axes[1, 0].set_ylabel('UMAP 2', fontsize=9)

    fig.suptitle(f'CORAL domain alignment  [{tag}]', fontsize=13)
    fig.tight_layout()

    for ext in ('png', 'pdf'):
        fig.savefig(out_dir / f'coral_comparison_{tag}.{ext}',
                    dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Guardado: coral_comparison_{tag}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_pca', type=int, default=128,
                        help='Componentes PCA previos a CORAL (default: 128)')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Regularización diagonal en matrices de covarianza (default: 1.0)')
    parser.add_argument('--seed',  type=int, default=42)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f'pca{args.n_pca}_a{args.alpha}_s{args.seed}'

    # 1. Cargar features
    print('Cargando features...')
    ann = pd.read_csv(ANNOT)
    X_raw, meta = load_features(ann)
    print(f'  Total: {len(X_raw)} slides  |  Features: {X_raw.shape[1]}')
    for ds in DATASET_LABELS:
        n   = (meta['dataset'] == ds).sum()
        mss = (meta[meta['dataset'] == ds]['category'] == 0).sum()
        msi = (meta[meta['dataset'] == ds]['category'] == 1).sum()
        print(f'  {DATASET_LABELS[ds]}: {n} slides  (MSS={mss}, MSI={msi})')

    # 2. StandardScaler + PCA
    print(f'\nPCA → {args.n_pca} componentes...')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    pca = PCA(n_components=args.n_pca, random_state=args.seed)
    X_pca = pca.fit_transform(X_scaled)
    print(f'  Varianza explicada: {pca.explained_variance_ratio_.sum():.1%}')

    # 3a. CORAL estándar
    print(f'\nCORAL estándar (referencia: Macarena, alpha={args.alpha})...')
    X_coral_std = apply_coral(X_pca, meta, args.alpha)

    # 3b. CORAL condicional
    print(f'\nCORAL condicional (MSS↔MSS, MSI↔MSI)...')
    X_coral_cond = apply_coral_conditional(X_pca, meta, args.alpha)

    # 4. Métricas cuantitativas
    print('\nMétricas de separabilidad (LR 5-fold CV):')
    m_base = evaluate(X_pca,        meta, 'Sin alineación  ')
    m_std  = evaluate(X_coral_std,  meta, 'CORAL estándar  ')
    m_cond = evaluate(X_coral_cond, meta, 'CORAL condicional')

    print(f'\n  {"":20s}  domain acc  class bal-acc')
    print(f'  {"Sin alineación":20s}  {m_base[0]:.3f}       {m_base[1]:.3f}')
    print(f'  {"CORAL estándar":20s}  {m_std[0]:.3f}       {m_std[1]:.3f}')
    print(f'  {"CORAL condicional":20s}  {m_cond[0]:.3f}       {m_cond[1]:.3f}')

    pd.DataFrame({
        'variant':         ['sin_alineacion', 'coral_estandar', 'coral_condicional'],
        'domain_accuracy': [m_base[0], m_std[0], m_cond[0]],
        'class_bal_acc':   [m_base[1], m_std[1], m_cond[1]],
    }).to_csv(OUT_DIR / f'metrics_{tag}.csv', index=False)

    # 5. UMAP para las tres variantes
    print('\nUMAP sin alineación...')
    emb_base = run_umap(X_pca, args.seed)
    print('UMAP CORAL estándar...')
    emb_std  = run_umap(X_coral_std, args.seed)
    print('UMAP CORAL condicional...')
    emb_cond = run_umap(X_coral_cond, args.seed)

    # 6. Figura comparativa 2×3
    print('\nGenerando figuras...')
    plot_comparison(
        embs    = [emb_base,       emb_std,          emb_cond],
        titles  = ['Sin alineación', 'CORAL estándar', 'CORAL condicional'],
        metrics = [m_base, m_std, m_cond],
        meta    = meta,
        tag     = tag,
        out_dir = OUT_DIR,
    )

    # Guardar embeddings
    emb_df = meta.copy()
    emb_df[['base_1',  'base_2']]  = emb_base
    emb_df[['std_1',   'std_2']]   = emb_std
    emb_df[['cond_1',  'cond_2']]  = emb_cond
    emb_df.to_csv(OUT_DIR / f'embeddings_{tag}.csv', index=False)

    print(f'\nListo. Resultados en: {OUT_DIR}')


if __name__ == '__main__':
    main()