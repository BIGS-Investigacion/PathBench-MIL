#!/usr/bin/env python3
"""
eval_shift_pair.py

Geometría del domain shift entre dos cohortes, con features SIN normalizar y con
MACENKO, para ver si la normalización de tinción reduce el shift.

Uso:
  python scripts/colon/eval_shift_pair.py --source tcga_coad --target cptac_coad
"""
import warnings, argparse
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold
warnings.filterwarnings('ignore')

ROOT=Path(__file__).resolve().parents[2]
BAGS={'none':Path('/home/PARADIS/datos/features/shared_bags/256_128_none_h_optimus_0'),
      'macenko':Path('/home/PARADIS/datos/features/shared_bags/256_128_macenko_h_optimus_0')}
ANN=ROOT/'config/annotations/annotations_colon_mss_msi_all.csv'

def load_df(norm,df):
    """Carga X (slide mean) e y para una tabla con columnas slide, category."""
    d=BAGS[norm]; X,y=[],[]
    for _,r in df.iterrows():
        p=d/f"{r['slide']}.pt"
        if not p.exists(): continue
        X.append(torch.load(p,map_location='cpu',weights_only=True).float().numpy().mean(0))
        y.append(int(r['category']))
    return np.stack(X).astype(np.float32), np.array(y)

def cosdist(a,b): return 1-np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9)
def cossim(a,b): return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9)
def dom_auc(Xa,Xb,seed=42):
    X=np.vstack([Xa,Xb]); y=np.r_[np.zeros(len(Xa)),np.ones(len(Xb))]
    Xs=StandardScaler().fit_transform(X)
    p=cross_val_predict(LogisticRegression(max_iter=2000),Xs,y,cv=StratifiedKFold(5,shuffle=True,random_state=seed),method='predict_proba')[:,1]
    return roc_auc_score(y,p)
def lr_eval(Xtr,ytr,Xte,yte,n_pca=128,seed=42):
    sc=StandardScaler().fit(Xtr); pca=PCA(min(n_pca,Xtr.shape[0]-1),random_state=seed).fit(sc.transform(Xtr))
    lr=LogisticRegression(max_iter=2000,random_state=seed).fit(pca.transform(sc.transform(Xtr)),ytr)
    yp=lr.predict(pca.transform(sc.transform(Xte)))
    return (yp[yte==0]==0).mean(),(yp[yte==1]==1).mean(),balanced_accuracy_score(yte,yp)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',default='tcga_coad'); ap.add_argument('--target',default='cptac_coad')
    ap.add_argument('--target_ann',default=None,
                    help='CSV con etiquetas del target (slide,category). Si no, usa all.csv filtrado por --target')
    ap.add_argument('--label',default='',help='etiqueta descriptiva del target para el print')
    ap.add_argument('--n_pca',type=int,default=128); a=ap.parse_args()
    allann=pd.read_csv(ANN)
    src_df=allann[allann['dataset']==a.source][['slide','category']]
    tgt_df=(pd.read_csv(a.target_ann)[['slide','category']] if a.target_ann
            else allann[allann['dataset']==a.target][['slide','category']])
    for norm in ['none','macenko']:
        Xs,ys=load_df(norm,src_df); Xt,yt=load_df(norm,tgt_df)
        cs=Xs.mean(0); ct=Xt.mean(0)
        sp=(np.linalg.norm(Xs-cs,axis=1).mean()+np.linalg.norm(Xt-ct,axis=1).mean())/2
        tgtlab=a.label or a.target
        print(f"\n{'='*60}\n{norm.upper()}  {a.source}(src {len(ys)}: {int((ys==0).sum())}/{int((ys==1).sum())}) "
              f"-> {tgtlab}(tgt {len(yt)}: {int((yt==0).sum())}/{int((yt==1).sum())})\n{'='*60}")
        print(f"  ||shift|| = {np.linalg.norm(ct-cs):.3f}   dispersión = {sp:.3f}   ratio = {np.linalg.norm(ct-cs)/sp:.3f}")
        print(f"  AUC dominio: antes={dom_auc(Xs,Xt):.3f}  tras centrar={dom_auc(Xs-cs,Xt-ct):.3f}")
        sMSS,sMSI=Xs[ys==0].mean(0),Xs[ys==1].mean(0); tMSS,tMSI=Xt[yt==0].mean(0),Xt[yt==1].mean(0)
        print(f"  cos-dist misma clase cross: MSS={cosdist(sMSS,tMSS):.3f} MSI={cosdist(sMSI,tMSI):.3f}")
        print(f"  cos-dist intra: src={cosdist(sMSS,sMSI):.3f} tgt={cosdist(tMSS,tMSI):.3f}")
        print(f"  eje MSS->MSI coseno(src,tgt) = {cossim(sMSI-sMSS,tMSI-tMSS):.3f}")
        a0,a1,b=lr_eval(Xs,ys,Xt,yt,a.n_pca)
        print(f"  LR train->test: MSS={a0:.3f} MSI={a1:.3f} bal={b:.3f}")

if __name__=='__main__': main()
