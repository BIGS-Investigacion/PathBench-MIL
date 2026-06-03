#!/usr/bin/env python3
"""
eval_coral_orthogonal.py

CORAL ortogonal a la clase:

  Descompone cada feature en dos componentes:
    x_clase   = proyección sobre la dirección MSS→MSI de Macarena
    x_dominio = componente ortogonal (subespacio perpendicular a la clase)

  Aplica CORAL solo en el subespacio de dominio.
  La componente de clase queda intacta.

  Resultado esperado: alineación de covarianzas sin destruir señal MSS/MSI.

Compara:
  A) Sin alineación
  C) CORAL cond. labels verdaderas  (límite superior)
  E) Corrección geométrica (medias)
  G) CORAL ortogonal a la clase     (este método)

Uso:
  python scripts/colon/eval_coral_orthogonal.py [--n_pca 128] [--seed 42]
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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

warnings.filterwarnings('ignore')

ROOT     = Path(__file__).resolve().parents[2]
BAGS_DIR = Path('/home/PARADIS/colon/tfrecords/shared_bags/256_128_none_h_optimus_0')
ANNOT    = ROOT / 'config/annotations/annotations_colon_mss_msi_all.csv'
OUT_DIR  = ROOT / 'results/colon/coral_orthogonal'

DATASET_LABELS = {'macarena': 'Macarena', 'cptac_coad': 'CPTAC-COAD', 'tcga_coad': 'TCGA-COAD'}
DATASET_COLORS = {'macarena': '#2196F3', 'cptac_coad': '#FF9800', 'tcga_coad': '#4CAF50'}
CLASS_COLORS   = {0: '#E53935', 1: '#1E88E5'}
CLASS_NAMES    = {0: 'MSS', 1: 'MSI'}
CLASS_MARKERS  = {0: 'o', 1: '^'}


# ── Carga ─────────────────────────────────────────────────────────────────────

def load_features(ann):
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


# ── Proyección ────────────────────────────────────────────────────────────────

def project_out(X, direction):
    """Elimina la componente de X en la dirección dada (unitaria)."""
    return X - np.outer(X @ direction, direction)


def split_components(X, direction):
    """Separa X en componente paralela y ortogonal a direction."""
    coeff    = X @ direction                        # (N,)
    X_para   = np.outer(coeff, direction)           # (N, d)
    X_ortho  = X - X_para                          # (N, d)
    return X_para, X_ortho, coeff


# ── CORAL ─────────────────────────────────────────────────────────────────────

def coral(X_src, X_tgt, alpha):
    mu_s = X_src.mean(0);  mu_t = X_tgt.mean(0)
    d = X_src.shape[1]
    Cs = np.cov((X_src - mu_s).T) + alpha * np.eye(d)
    Ct = np.cov((X_tgt - mu_t).T) + alpha * np.eye(d)
    Us, Ss, _ = np.linalg.svd(Cs);  Ut, St, _ = np.linalg.svd(Ct)
    W = Us @ np.diag(1/np.sqrt(np.maximum(Ss, 1e-10))) @ Us.T @ \
        Ut @ np.diag(np.sqrt(np.maximum(St, 0))) @ Ut.T
    return (X_src - mu_s) @ W + mu_t


def apply_coral_std(X, meta, alpha):
    X_out = X.copy()
    X_mac = X[(meta['dataset'] == 'macarena').values]
    for ds in ['cptac_coad', 'tcga_coad']:
        mask = (meta['dataset'] == ds).values
        X_out[mask] = coral(X[mask], X_mac, alpha)
    return X_out


def apply_coral_cond(X, meta, labels, alpha):
    X_out = X.copy();  d = X.shape[1]
    for ds in ['cptac_coad', 'tcga_coad']:
        for cls_id in [0, 1]:
            mask_src = (meta['dataset'] == ds).values & (labels == cls_id)
            mask_tgt = (meta['dataset'] == 'macarena').values & \
                       (meta['category'].values == cls_id)
            n_s, n_t = mask_src.sum(), mask_tgt.sum()
            if n_s < 5 or n_t < 5:
                continue
            alpha_eff = alpha * max(1.0, d / min(n_s, n_t))
            X_out[mask_src] = coral(X[mask_src], X[mask_tgt], alpha_eff)
    return X_out


def apply_coral_orthogonal(X, meta, class_dir, alpha):
    """
    CORAL en el subespacio ortogonal a class_dir.
    La componente de clase (proyección sobre class_dir) queda sin modificar.
    """
    # Separar componentes
    X_para, X_ortho, coeffs = split_components(X, class_dir)

    # CORAL solo sobre la parte ortogonal
    X_ortho_aligned = apply_coral_std(X_ortho, meta, alpha)

    # Reconstruir: clase original + dominio alineado
    return X_ortho_aligned + X_para


# ── Métricas ──────────────────────────────────────────────────────────────────

def evaluate(lr, X, meta, tag, rows):
    for ds in ['macarena', 'cptac_coad', 'tcga_coad']:
        mask   = (meta['dataset'] == ds).values
        y_true = meta.loc[meta['dataset'] == ds, 'category'].values
        y_pred = lr.predict(X[mask])
        mss = y_true == 0;  msi = y_true == 1
        acc_mss = (y_pred[mss] == 0).mean() if mss.sum() > 0 else float('nan')
        acc_msi = (y_pred[msi] == 1).mean() if msi.sum() > 0 else float('nan')
        bal     = balanced_accuracy_score(y_true, y_pred)
        print(f'  [{tag:38s}] {DATASET_LABELS[ds]:12s}  '
              f'MSS={acc_mss:.3f}  MSI={acc_msi:.3f}  bal={bal:.3f}')
        rows.append({'tag': tag, 'dataset': DATASET_LABELS[ds],
                     'acc_mss': acc_mss, 'acc_msi': acc_msi, 'bal_acc': bal})


# ── UMAP ──────────────────────────────────────────────────────────────────────

def plot_umap(embs, titles, meta, tag, out_dir):
    import umap as umap_lib
    n = len(embs)
    fig, axes = plt.subplots(2, n, figsize=(6 * n, 11))

    for col, (X, title) in enumerate(zip(embs, titles)):
        emb = umap_lib.UMAP(n_components=2, random_state=42,
                            n_neighbors=30, min_dist=0.1,
                            metric='cosine').fit_transform(X)
        for row, key in enumerate(['dataset', 'category']):
            ax = axes[row, col]
            if key == 'dataset':
                for ds in DATASET_LABELS:
                    mask = (meta['dataset'] == ds).values
                    ax.scatter(emb[mask, 0], emb[mask, 1], c=DATASET_COLORS[ds],
                               s=22, alpha=0.75, edgecolors='none',
                               label=DATASET_LABELS[ds])
                if col == 0:
                    ax.legend(fontsize=8, framealpha=0.9)
                ax.set_title(f'{title}\npor base de datos', fontsize=9)
            else:
                for cls_id in [0, 1]:
                    mask = (meta['category'] == cls_id).values
                    ax.scatter(emb[mask, 0], emb[mask, 1], c=CLASS_COLORS[cls_id],
                               marker=CLASS_MARKERS[cls_id], s=22, alpha=0.75,
                               edgecolors='none', label=CLASS_NAMES[cls_id])
                if col == 0:
                    ax.legend(fontsize=8, framealpha=0.9)
                ax.set_title(f'{title}\npor MSS/MSI', fontsize=9)
            ax.tick_params(labelleft=False, labelbottom=False)

    fig.suptitle(f'CORAL ortogonal a la clase [{tag}]', fontsize=13)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(out_dir / f'umap_orthogonal_{tag}.{ext}', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Guardado: umap_orthogonal_{tag}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_pca', type=int,   default=128)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--seed',  type=int,   default=42)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f'pca{args.n_pca}_s{args.seed}'

    # 1. Cargar y proyectar con PCA
    print('Cargando features...')
    ann = pd.read_csv(ANNOT)
    X_raw, meta = load_features(ann)
    mask_mac = (meta['dataset'] == 'macarena').values

    scaler = StandardScaler().fit(X_raw[mask_mac])
    pca    = PCA(n_components=args.n_pca, random_state=args.seed).fit(
                 scaler.transform(X_raw[mask_mac]))
    X_pca  = pca.transform(scaler.transform(X_raw))
    print(f'  Varianza explicada PCA: {pca.explained_variance_ratio_.sum():.1%}')

    # 2. Dirección de clase en Macarena
    y_mac     = meta.loc[mask_mac, 'category'].values
    mu_mss    = X_pca[mask_mac][y_mac == 0].mean(axis=0)
    mu_msi    = X_pca[mask_mac][y_mac == 1].mean(axis=0)
    class_dir = mu_msi - mu_mss
    class_dir /= np.linalg.norm(class_dir)

    # Varianza explicada por la dirección de clase en cada dominio
    print('\nVarianza en la dirección de clase por dominio:')
    for ds in DATASET_LABELS:
        mask  = (meta['dataset'] == ds).values
        var_c = np.var(X_pca[mask] @ class_dir)
        var_t = np.var(X_pca[mask])
        print(f'  {DATASET_LABELS[ds]:12s}: var_clase={var_c:.3f}  '
              f'var_total={var_t:.3f}  ratio={var_c/var_t:.3%}')

    # 3. LR sobre Macarena
    lr = LogisticRegression(max_iter=2000, random_state=args.seed, C=1.0)
    lr.fit(X_pca[mask_mac], y_mac)

    rows = []
    print('\nResultados:')
    print(f'  {"variante":38s} {"dataset":12s}  MSS     MSI     bal-acc')
    print(f'  {"-"*80}')

    # A) Sin alineación
    evaluate(lr, X_pca, meta, 'A) Sin alineación', rows)
    print()

    # B) CORAL estándar
    X_std = apply_coral_std(X_pca, meta, args.alpha)
    evaluate(lr, X_std, meta, 'B) CORAL estándar', rows)
    print()

    # C) CORAL cond. labels verdaderas (límite superior)
    X_cond = apply_coral_cond(X_pca, meta, meta['category'].values, args.alpha)
    evaluate(lr, X_cond, meta, 'C) CORAL cond. — labels verdaderas', rows)
    print()

    # G) CORAL ortogonal a la clase
    print('  Aplicando CORAL ortogonal...')
    X_orth = apply_coral_orthogonal(X_pca, meta, class_dir, args.alpha)
    evaluate(lr, X_orth, meta, 'G) CORAL ortogonal a la clase', rows)
    print()

    # Guardar
    pd.DataFrame(rows).to_csv(OUT_DIR / f'results_{tag}.csv', index=False)

    # UMAP
    print('\nGenerando UMAP...')
    plot_umap(
        embs   = [X_pca, X_std, X_cond, X_orth],
        titles = ['Sin alineación', 'CORAL estándar',
                  'CORAL cond. (oracle)', 'CORAL ortogonal'],
        meta=meta, tag=tag, out_dir=OUT_DIR)

    print(f'\nListo. Resultados en: {OUT_DIR}')


if __name__ == '__main__':
    main()