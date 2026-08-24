#!/usr/bin/env python3
"""
eval_lodo_lr.py

Vías HONESTAS (sin etiquetas del target) para el shift cross-domain, MSS/MSI:
  - Multi-fuente (LODO): entrena con 2 dominios, testea en el 3º.
  - Comparado con single-source (1 dominio -> otro) como referencia.
  - Todo repetido con features SIN normalizar y con MACENKO.

Representación: slide mean + StandardScaler + PCA + LR (ajustados en el train).

Uso:
  python scripts/colon/eval_lodo_lr.py [--n_pca 128]
"""
import warnings, argparse
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

warnings.filterwarnings('ignore')
ROOT=Path(__file__).resolve().parents[2]
BAGS={'none':Path('/home/PARADIS/datos/features/shared_bags/256_128_none_h_optimus_0'),
      'macenko':Path('/home/PARADIS/datos/features/shared_bags/256_128_macenko_h_optimus_0')}
ANN=ROOT/'config/annotations/annotations_colon_mss_msi_all.csv'
DOMS=['macarena','tcga_coad','cptac_coad']
LBL={'macarena':'Macarena','tcga_coad':'TCGA','cptac_coad':'CPTAC'}

def load(norm):
    d=BAGS[norm]; cache={}
    ann=pd.read_csv(ANN)
    X,meta=[],[]
    for _,r in ann.iterrows():
        p=d/f"{r['slide']}.pt"
        if not p.exists(): continue
        X.append(torch.load(p,map_location='cpu',weights_only=True).float().numpy().mean(0))
        meta.append((r['dataset'],int(r['category'])))
    X=np.stack(X).astype(np.float32)
    m=pd.DataFrame(meta,columns=['dataset','y'])
    return X,m

def fit_test(Xtr,ytr,Xte,yte,n_pca,seed=42):
    sc=StandardScaler().fit(Xtr)
    pca=PCA(n_components=min(n_pca,Xtr.shape[0]-1),random_state=seed).fit(sc.transform(Xtr))
    lr=LogisticRegression(max_iter=2000,C=1.0,random_state=seed).fit(pca.transform(sc.transform(Xtr)),ytr)
    Zte=pca.transform(sc.transform(Xte))
    yp=lr.predict(Zte); pr=lr.predict_proba(Zte)[:,1]
    bal=balanced_accuracy_score(yte,yp)
    auc=roc_auc_score(yte,pr) if len(set(yte))>1 else np.nan
    a0=(yp[yte==0]==0).mean() if (yte==0).sum() else np.nan
    a1=(yp[yte==1]==1).mean() if (yte==1).sum() else np.nan
    return a0,a1,bal,auc

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--n_pca',type=int,default=128); a=ap.parse_args()
    for norm in ['none','macenko']:
        X,m=load(norm)
        print(f"\n{'='*66}\nFeatures: {norm.upper()}  (slides={len(m)}, PCA={a.n_pca})\n{'='*66}")
        print(f"  {'target':<10} {'fuente':<22} {'MSS':>6} {'MSI':>6} {'bal':>6} {'AUC':>6}")
        for tgt in DOMS:
            te=(m['dataset']==tgt).values
            Xte,yte=X[te],m.loc[te,'y'].values
            srcs=[d for d in DOMS if d!=tgt]
            # single-source (cada uno de los otros)
            for s in srcs:
                tr=(m['dataset']==s).values
                a0,a1,bal,auc=fit_test(X[tr],m.loc[tr,'y'].values,Xte,yte,a.n_pca)
                print(f"  {LBL[tgt]:<10} {LBL[s]+' (single)':<22} {a0:>6.3f} {a1:>6.3f} {bal:>6.3f} {auc:>6.3f}")
            # multi-source (los dos)
            tr=(m['dataset'].isin(srcs)).values
            a0,a1,bal,auc=fit_test(X[tr],m.loc[tr,'y'].values,Xte,yte,a.n_pca)
            print(f"  {LBL[tgt]:<10} {'+'.join(LBL[s] for s in srcs)+' (LODO)':<22} {a0:>6.3f} {a1:>6.3f} {bal:>6.3f} {auc:>6.3f}")
            print()

if __name__=='__main__':
    main()
