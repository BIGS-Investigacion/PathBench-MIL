#!/bin/bash
#SBATCH --job-name=shift_cptac_src
#SBATCH --output=slurm_logs/%j_shift_cptac_src.out
#SBATCH --error=slurm_logs/%j_shift_cptac_src.err
#SBATCH --nodelist=c08
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

source /shared/home/jorgarcia/monai_qupath/.venv/bin/activate
cd /shared/home/jorgarcia/PathBench-MIL
mkdir -p slurm_logs

echo "############## CV INTERNO CPTAC (¿es aprendible?) ##############"
python3 - << 'PY'
import numpy as np, pandas as pd, torch
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
for norm in ['none','macenko']:
    D=Path(f'/home/PARADIS/datos/features/shared_bags/256_128_{norm}_h_optimus_0')
    ann=pd.read_csv('config/annotations/annotations_colon_mss_msi_all.csv')
    ann=ann[ann['dataset']=='cptac_coad']
    X,y,g=[],[],[]
    for _,r in ann.iterrows():
        p=D/f"{r['slide']}.pt"
        if not p.exists(): continue
        X.append(torch.load(p,map_location='cpu',weights_only=True).float().numpy().mean(0))
        y.append(int(r['category'])); g.append(r['patient'])
    X=np.stack(X); y=np.array(y); g=np.array(g)
    sk=StratifiedGroupKFold(5,shuffle=True,random_state=42)
    a0,a1,ab=[],[],[]
    for tr,te in sk.split(X,y,groups=g):
        sc=StandardScaler().fit(X[tr]); pca=PCA(min(128,len(tr)-1),random_state=42).fit(sc.transform(X[tr]))
        lr=LogisticRegression(max_iter=2000).fit(pca.transform(sc.transform(X[tr])),y[tr])
        yp=lr.predict(pca.transform(sc.transform(X[te]))); yt=y[te]
        a0.append((yp[yt==0]==0).mean()); a1.append((yp[yt==1]==1).mean()); ab.append(balanced_accuracy_score(yt,yp))
    print(f"  CPTAC {norm}: 5-fold CV  MSS={np.mean(a0):.3f} MSI={np.mean(a1):.3f} bal={np.mean(ab):.3f}")
PY

echo ""
echo "############## CPTAC -> TCGA-PCR ##############"
python3 scripts/colon/eval_shift_pair.py --source cptac_coad --target tcga_coad --label TCGA-PCR --n_pca 128
echo ""
echo "############## CPTAC -> MACARENA-IHC ##############"
python3 scripts/colon/eval_shift_pair.py --source cptac_coad --target macarena --label Macarena-IHC --n_pca 128
