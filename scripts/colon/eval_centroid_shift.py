#!/usr/bin/env python3
"""
eval_centroid_shift.py

Prueba si el domain shift Macarena->TCGA es DIRECCIONAL (una traslación pura).

Representación: slide mean (media de todos los patches, 1536-dim).

Experimentos (LR sobre slide means):
  A. Baseline: StandardScaler+PCA ajustados en Macarena (ya centra por la media
     de Macarena) -> test TCGA.
  B. Alineación de medias: a cada dominio se le resta SU PROPIO centroide global
     antes del pipeline (X_mac -= c_mac ; X_tcga -= c_tcga). Si el shift es una
     traslación, esto lo elimina y la clasificación mejora.

Diagnóstico de direccionalidad:
  - Norma del vector de shift ||c_tcga - c_mac|| vs dispersión intra-dominio.
  - Separabilidad de dominio (AUC de un clasificador Macarena-vs-TCGA) ANTES y
    DESPUÉS de restar la media de cada dominio. Si tras centrar sigue siendo
    separable -> el shift NO es solo direccional (hay rotación/escala).

Uso:
  python scripts/colon/eval_centroid_shift.py [--n_pca 128] [--tcga_label pcr|ihc]
"""

import warnings, argparse
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold

warnings.filterwarnings('ignore')
ROOT     = Path(__file__).resolve().parents[2]
BAGS_DIR = Path('/home/PARADIS/datos/features/shared_bags/256_128_none_h_optimus_0')

_cache={}
def slide_mean(s):
    if s not in _cache:
        p=BAGS_DIR/f'{s}.pt'
        _cache[s]=None if not p.exists() else \
            torch.load(p,map_location='cpu',weights_only=True).float().numpy().mean(0).astype(np.float32)
    return _cache[s]

def load(ann):
    v,r=[],[]
    for _,x in ann.iterrows():
        f=slide_mean(x['slide'])
        if f is None: continue
        v.append(f); r.append(x['category'])
    return np.stack(v).astype(np.float32), np.array(r,dtype=int)

def lr_eval(Xtr,ytr,Xte,yte,n_pca,seed=42):
    sc=StandardScaler().fit(Xtr)
    pca=PCA(n_components=min(n_pca,Xtr.shape[0]-1),random_state=seed).fit(sc.transform(Xtr))
    lr=LogisticRegression(max_iter=2000,C=1.0,random_state=seed).fit(pca.transform(sc.transform(Xtr)),ytr)
    yp=lr.predict(pca.transform(sc.transform(Xte)))
    a0=(yp[yte==0]==0).mean() if (yte==0).sum() else np.nan
    a1=(yp[yte==1]==1).mean() if (yte==1).sum() else np.nan
    return a0,a1,balanced_accuracy_score(yte,yp)

def domain_auc(Xa,Xb,seed=42):
    """AUC de un LR que separa dominio A vs B (5-fold CV)."""
    X=np.vstack([Xa,Xb]); y=np.r_[np.zeros(len(Xa)),np.ones(len(Xb))]
    sc=StandardScaler().fit(X); Xs=sc.transform(X)
    skf=StratifiedKFold(5,shuffle=True,random_state=seed)
    p=cross_val_predict(LogisticRegression(max_iter=2000,C=1.0),Xs,y,cv=skf,method='predict_proba')[:,1]
    return roc_auc_score(y,p)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--n_pca',type=int,default=128)
    ap.add_argument('--tcga_label',choices=['pcr','ihc'],default='pcr')
    a=ap.parse_args()

    allann=pd.read_csv(ROOT/'config/annotations/annotations_colon_mss_msi_all.csv')
    mac=allann[allann['dataset']=='macarena']
    if a.tcga_label=='pcr':
        tcga=allann[allann['dataset']=='tcga_coad']
    else:
        tcga=pd.read_csv(ROOT/'config/annotations/annotations_tcga_coad_mss_msi_ihc.csv')

    Xm,ym=load(mac); Xt,yt=load(tcga)
    print(f"Macarena: {len(ym)} (MSS {int((ym==0).sum())}/MSI {int((ym==1).sum())})  "
          f"TCGA-{a.tcga_label}: {len(yt)} (MSS {int((yt==0).sum())}/MSI {int((yt==1).sum())})\n")

    cm=Xm.mean(0); ct=Xt.mean(0)
    shift=ct-cm
    spread_m=np.linalg.norm(Xm-cm,axis=1).mean()
    spread_t=np.linalg.norm(Xt-ct,axis=1).mean()
    print("=== Geometría del shift ===")
    print(f"  ||c_tcga - c_mac||        = {np.linalg.norm(shift):.3f}")
    print(f"  dispersión intra Macarena = {spread_m:.3f}")
    print(f"  dispersión intra TCGA     = {spread_t:.3f}")
    print(f"  ratio shift/dispersión    = {np.linalg.norm(shift)/((spread_m+spread_t)/2):.3f}\n")

    print("=== Separabilidad de dominio (AUC Macarena-vs-TCGA, 5-fold) ===")
    auc_before=domain_auc(Xm,Xt)
    auc_after =domain_auc(Xm-cm, Xt-ct)
    print(f"  ANTES de centrar cada dominio:  AUC = {auc_before:.3f}")
    print(f"  DESPUÉS de restar cada centroide: AUC = {auc_after:.3f}")
    print(f"  -> {'shift casi puramente DIRECCIONAL' if auc_after<0.65 else 'shift NO es solo direccional (queda estructura)'}\n")

    print("=== Clasificación MSS/MSI (LR, train Macarena -> test TCGA) ===")
    a0,a1,bal=lr_eval(Xm,ym,Xt,yt,a.n_pca)
    print(f"  A. Baseline (centra por media de Macarena):        MSS={a0:.3f} MSI={a1:.3f} bal={bal:.3f}")
    a0,a1,bal=lr_eval(Xm-cm,ym,Xt-ct,yt,a.n_pca)
    print(f"  B. Alineación de medias (resta centroide propio):  MSS={a0:.3f} MSI={a1:.3f} bal={bal:.3f}")

if __name__=='__main__':
    main()
