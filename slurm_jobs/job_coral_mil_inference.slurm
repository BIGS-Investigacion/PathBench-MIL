#!/bin/bash
#SBATCH --job-name=coral_mil
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

echo "=== ORACLE ==="
python scripts/colon/eval_coral_mil_inference.py --mode oracle --n_pca 128 --seed 42

echo "=== FEW-SHOT N=10 ==="
python scripts/colon/eval_coral_mil_inference.py --mode fewshot --fewshot_n 10 --fewshot_seeds 5 --n_pca 128 --seed 42