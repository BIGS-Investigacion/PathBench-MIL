#!/usr/bin/env python3
"""
eval_dsmil_base_targets.py

Modelo DSMIL base entrenado en Macarena (benchmark original, sin macenko),
evaluado por inferencia sobre 3 dianas:
  1. CPTAC-frozen-PCR      (annotations_cptac_coad_mss_msi.csv)
  2. TCGA-diagnostic-PCR   (annotations_tcga_coad_mss_msi.csv)
  3. TCGA-diagnostic-IHC   (annotations_tcga_coad_mss_msi_ihc.csv)
Reporta acc_MSS, acc_MSI, balanced accuracy y AUC.
"""
import json, importlib.util
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT=Path(__file__).resolve().parents[2]
BAGS=Path('/home/PARADIS/datos/features/shared_bags/256_128_none_h_optimus_0')
MODEL=ROOT/'experiments/colon_mss_msi_benchmark/mil/00000-256_128_none_h_optimus_0_dsmil_CrossEntropyLoss_ReLU_Adam_1'
TARGETS={
 'CPTAC-frozen-PCR':    ROOT/'config/annotations/annotations_cptac_coad_mss_msi.csv',
 'TCGA-diagnostic-PCR': ROOT/'config/annotations/annotations_tcga_coad_mss_msi.csv',
 'TCGA-diagnostic-IHC': ROOT/'config/annotations/annotations_tcga_coad_mss_msi_ihc.csv',
}

def load_model(dev):
    spec=importlib.util.spec_from_file_location('agg',ROOT/'pathbench/models/aggregators.py')
    agg=importlib.util.module_from_spec(spec); spec.loader.exec_module(agg)
    mp=json.load(open(MODEL/'mil_params.json')); pr=mp['params']
    m=getattr(agg,pr['model'])(n_feats=mp['input_shape'],n_out=mp['output_shape'],
        z_dim=pr.get('z_dim',256),dropout_p=pr.get('dropout_p',0.1),
        activation_function=pr.get('activation_function','ReLU'),encoder_layers=pr.get('encoder_layers',1))
    ck=sorted(MODEL.glob('checkpoints/**/best-epoch*.ckpt'))[-1]
    st=torch.load(ck,map_location='cpu')
    sd={k.removeprefix('model.'):v for k,v in st['state_dict'].items() if k.startswith('model.')}
    m.load_state_dict(sd,strict=True); return m.to(dev).eval()

def pmsi(model,slide,dev):
    p=BAGS/f'{slide}.pt'
    if not p.exists(): return None
    bag=torch.load(p,map_location='cpu',weights_only=True).float()
    with torch.no_grad():
        logits=model(bag.unsqueeze(0).to(dev))
        return torch.softmax(logits,1).squeeze()[1].item()

def main():
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=load_model(dev)
    out=ROOT/'results/colon/dsmil_base_targets'; out.mkdir(parents=True,exist_ok=True)
    print(f"Modelo base: DSMIL Macarena (benchmark original, sin macenko)\n")
    print(f"{'Diana':<22} {'N(MSS/MSI)':>12} {'accMSS':>8} {'accMSI':>8} {'bal':>7} {'AUC':>7}")
    print('='*68)
    summary=[]
    for name,ann_p in TARGETS.items():
        ann=pd.read_csv(ann_p)
        rows=[]
        for _,r in ann.iterrows():
            pr=pmsi(model,r['slide'],dev)
            if pr is None: continue
            rows.append({'slide':r['slide'],'y_true':int(r['category']),
                         'y_pred1':pr,'y_pred':int(pr>=0.5)})
        df=pd.DataFrame(rows)
        # guardar predicciones por slide (auditable)
        fn=name.replace('/','_').replace(' ','_')
        df.to_csv(out/f'preds_{fn}.csv',index=False)
        y=df['y_true'].values; p=df['y_pred1'].values; yp=df['y_pred'].values
        n0,n1=(y==0).sum(),(y==1).sum()
        a0=(yp[y==0]==0).mean() if n0 else np.nan
        a1=(yp[y==1]==1).mean() if n1 else np.nan
        auc=roc_auc_score(y,p) if len(set(y))>1 else np.nan
        bal=balanced_accuracy_score(y,yp)
        print(f"{name:<22} {str(n0)+'/'+str(n1):>12} {a0:>8.3f} {a1:>8.3f} "
              f"{bal:>7.3f} {auc:>7.3f}")
        summary.append({'target':name,'n_MSS':int(n0),'n_MSI':int(n1),
                        'accMSS':a0,'accMSI':a1,'bal':bal,'auc':auc,
                        'preds_file':str(out/f'preds_{fn}.csv')})
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
    print(f"\nGuardado: {out}/summary.csv  y  preds_*.csv (predicciones por slide)")

if __name__=='__main__': main()
