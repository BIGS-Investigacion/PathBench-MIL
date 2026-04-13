#!/bin/bash
#SBATCH --job-name=col_close_ext
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
python scripts/colon/select_closest_to_centroid.py --top_n 100 --dataset macarena

echo "=== TCGA-COAD ==="
python scripts/colon/select_closest_to_centroid.py --top_n 100 --dataset tcga_coad

echo "=== CPTAC-COAD ==="
python scripts/colon/select_closest_to_centroid.py --top_n 100 --dataset cptac_coad