#!/usr/bin/env python3
"""
agg_kfold5_seeds.py

Agrega el CV DSMIL 5-fold de Macarena repetido con 5 semillas
(colon_mss_msi_kfold5_seed{42..46}), sobre bags nuevos.
Reporta accMSS, accMSI, bal y AUC por semilla y su media±std.
"""
import glob
import numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT='experiments'
SEEDS=[42,43,44,45,46]

rows=[]
for s in SEEDS:
    fs=sorted(glob.glob(f'{ROOT}/colon_mss_msi_kfold5_seed{s}/mil/0*/predictions.csv'))
    if not fs:
        print(f"seed {s}: sin predicciones (¿job en curso?)"); continue
    D=pd.concat([pd.read_csv(f) for f in fs],ignore_index=True)
    y=D['y_true'].astype(int).values; p=D['y_pred1'].values; yp=(p>=0.5).astype(int)
    rows.append(dict(seed=s, n=len(D), folds=len(fs),
        accMSS=(yp[y==0]==0).mean(), accMSI=(yp[y==1]==1).mean(),
        bal=balanced_accuracy_score(y,yp), auc=roc_auc_score(y,p)))

if not rows:
    print("Aún no hay resultados."); raise SystemExit
df=pd.DataFrame(rows)
print(f"{'seed':>5} {'folds':>6} {'n':>5} {'accMSS':>8} {'accMSI':>8} {'bal':>7} {'AUC':>7}")
for _,r in df.iterrows():
    print(f"{int(r['seed']):>5} {int(r['folds']):>6} {int(r['n']):>5} "
          f"{r['accMSS']:>8.3f} {r['accMSI']:>8.3f} {r['bal']:>7.3f} {r['auc']:>7.3f}")
print("-"*50)
print(f"{'MEDIA':>5} {'':>6} {'':>5} "
      f"{df['accMSS'].mean():>8.3f} {df['accMSI'].mean():>8.3f} {df['bal'].mean():>7.3f} {df['auc'].mean():>7.3f}")
print(f"{'STD':>5} {'':>6} {'':>5} "
      f"{df['accMSS'].std():>8.3f} {df['accMSI'].std():>8.3f} {df['bal'].std():>7.3f} {df['auc'].std():>7.3f}")
print(f"\nResumen: bal = {df['bal'].mean():.3f} ± {df['bal'].std():.3f}   "
      f"AUC = {df['auc'].mean():.3f} ± {df['auc'].std():.3f}  (n_semillas={len(df)})")
