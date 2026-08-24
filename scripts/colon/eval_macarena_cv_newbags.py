#!/usr/bin/env python3
"""
eval_macarena_cv_newbags.py

Recalcula el CV de Macarena (DSMIL 5-fold, colon_mss_msi_kfold5) usando los BAGS
NUEVOS (/home/PARADIS/datos/features/shared_bags/256_128_none_h_optimus_0).

Para cada fold: carga su checkpoint, infiere sobre sus slides held-out (las que
figuran en su predictions.csv original) con los bags nuevos, y compara con las
predicciones originales.
"""
import json, importlib.util, glob
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT=Path(__file__).resolve().parents[2]
BAGS=Path('/home/PARADIS/datos/features/shared_bags/256_128_none_h_optimus_0')
EXP=ROOT/'experiments/colon_mss_msi_kfold5'

def load_model(model_dir,dev):
    spec=importlib.util.spec_from_file_location('agg',ROOT/'pathbench/models/aggregators.py')
    agg=importlib.util.module_from_spec(spec); spec.loader.exec_module(agg)
    mp=json.load(open(model_dir/'mil_params.json')); pr=mp['params']
    m=getattr(agg,pr['model'])(n_feats=mp['input_shape'],n_out=mp['output_shape'],
        z_dim=pr.get('z_dim',256),dropout_p=pr.get('dropout_p',0.1),
        activation_function=pr.get('activation_function','ReLU'),encoder_layers=pr.get('encoder_layers',1))
    ck=sorted(model_dir.glob('checkpoints/**/best-epoch*.ckpt'))[-1]
    st=torch.load(ck,map_location='cpu')
    sd={k.removeprefix('model.'):v for k,v in st['state_dict'].items() if k.startswith('model.')}
    m.load_state_dict(sd,strict=True); return m.to(dev).eval()

def pmsi(model,slide,dev):
    p=BAGS/f'{slide}.pt'
    if not p.exists(): return None
    bag=torch.load(p,map_location='cpu',weights_only=True).float()
    with torch.no_grad():
        return torch.softmax(model(bag.unsqueeze(0).to(dev)),1).squeeze()[1].item()

def main():
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dirs=sorted([Path(p) for p in glob.glob(str(EXP/'mil/0*')) if (Path(p)/'predictions.csv').exists()])
    print(f"Folds encontrados: {len(dirs)}\n")
    Y,Pnew,Pold=[],[],[]
    n_diff=0; n_tot=0
    for md in dirs:
        old=pd.read_csv(md/'predictions.csv')
        model=load_model(md,dev)
        for _,r in old.iterrows():
            pr=pmsi(model,r['slide'],dev)
            if pr is None: continue
            Y.append(int(r['y_true'])); Pnew.append(pr); Pold.append(float(r['y_pred1']))
            n_tot+=1
            if (pr>=0.5)!=(r['y_pred1']>=0.5): n_diff+=1
    Y=np.array(Y); Pn=np.array(Pnew); Po=np.array(Pold); yp=(Pn>=0.5).astype(int)
    n0,n1=(Y==0).sum(),(Y==1).sum()
    print("=== Macarena DSMIL 5-fold CV — BAGS NUEVOS ===")
    print(f"  n={len(Y)} (MSS={n0}/MSI={n1})")
    print(f"  accMSS={(yp[Y==0]==0).mean():.3f} accMSI={(yp[Y==1]==1).mean():.3f} "
          f"bal={balanced_accuracy_score(Y,yp):.3f} AUC={roc_auc_score(Y,Pn):.3f}")
    print(f"\n=== Consistencia con bags viejos (predicciones originales) ===")
    print(f"  predicciones que cambian de clase (umbral 0.5): {n_diff}/{n_tot} ({100*n_diff/n_tot:.1f}%)")
    print(f"  correlación P(MSI) nuevo vs viejo: {np.corrcoef(Pn,Po)[0,1]:.4f}")
    print(f"  dif. media |P_new - P_old|: {np.abs(Pn-Po).mean():.4f}")

if __name__=='__main__': main()
