#!/bin/bash
#SBATCH --job-name=col_patches_full
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

echo "=== [1/6] EXTRACT MACARENA ==="
python scripts/colon/extract_attention_patches.py --top_k 8 --force --dataset macarena

echo "=== [2/6] CLOSEST MACARENA ==="
python scripts/colon/select_closest_to_centroid.py --top_n 100 --dataset macarena

echo "=== [3/6] EXTRACT TCGA-COAD ==="
python scripts/colon/extract_attention_patches.py --top_k 8 --force --dataset tcga_coad

echo "=== [4/6] CLOSEST TCGA-COAD ==="
python scripts/colon/select_closest_to_centroid.py --top_n 100 --dataset tcga_coad

echo "=== [5/6] EXTRACT CPTAC-COAD ==="
python scripts/colon/extract_attention_patches.py --top_k 8 --force --dataset cptac_coad

echo "=== [6/6] CLOSEST CPTAC-COAD ==="
python scripts/colon/select_closest_to_centroid.py --top_n 100 --dataset cptac_coad

echo "=== DONE ==="