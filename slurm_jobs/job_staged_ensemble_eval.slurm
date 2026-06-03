#!/bin/bash
#SBATCH --job-name=staged_eval
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
#SBATCH --gpus=a100:1
#SBATCH --nodelist=gpu08
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:00:00

source /etc/profile
source /shared/home/PARADIS/PathBench-MIL/pathbench_env/bin/activate
cd /shared/home/jorgarcia/PathBench-MIL
mkdir -p slurm_logs

python scripts/colon/eval_staged_ensemble.py --experiment_dir experiments/colon_mss_msi_staged_val50 --mode best