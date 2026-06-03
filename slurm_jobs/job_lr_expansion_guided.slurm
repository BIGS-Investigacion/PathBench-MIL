#!/bin/bash
#SBATCH --job-name=lr_exp_guided
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

python scripts/colon/eval_lr_expansion_guided.py \
    --k 8 --n_gen 3 --lam 1.0 --n_synthetic 100 --seeds 10