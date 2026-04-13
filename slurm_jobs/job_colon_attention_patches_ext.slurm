#!/bin/bash
#SBATCH --job-name=col_att_ext
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
#SBATCH --nodelist=gpu06
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:00:00

source /etc/profile
source /shared/home/PARADIS/PathBench-MIL/pathbench_env/bin/activate
cd /shared/home/jorgarcia/PathBench-MIL
mkdir -p slurm_logs

echo "=== MACARENA ==="
python scripts/colon/extract_attention_patches.py --top_k 8 --force --dataset macarena

echo "=== TCGA-COAD ==="
python scripts/colon/extract_attention_patches.py --top_k 8 --force --dataset tcga_coad

echo "=== CPTAC-COAD ==="
python scripts/colon/extract_attention_patches.py --top_k 8 --force --dataset cptac_coad