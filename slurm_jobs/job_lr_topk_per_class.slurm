#!/bin/bash
#SBATCH --job-name=lr_topk_per_class
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
#SBATCH --nodelist=gpu04
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

source /etc/profile
source /shared/home/PARADIS/PathBench-MIL/pathbench_env/bin/activate
cd /shared/home/jorgarcia/PathBench-MIL
mkdir -p slurm_logs

python scripts/colon/eval_lr_topk_per_class.py \
    --k 8 --n_pca 128 --seeds 20