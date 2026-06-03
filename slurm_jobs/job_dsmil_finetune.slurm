#!/bin/bash
#SBATCH --job-name=dsmil_ft
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
#SBATCH --nodelist=gpu04
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

source /etc/profile
source /shared/home/PARADIS/PathBench-MIL/pathbench_env/bin/activate
cd /shared/home/jorgarcia/PathBench-MIL
mkdir -p slurm_logs

echo "=== CPTAC-COAD ==="
python scripts/colon/eval_dsmil_finetune.py --ds cptac_coad --seeds 10 --lr 1e-4 --epochs 20

echo ""
echo "=== TCGA-COAD ==="
python scripts/colon/eval_dsmil_finetune.py --ds tcga_coad  --seeds 10 --lr 1e-4 --epochs 20