#!/bin/bash
#SBATCH --job-name=brca_optuna
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=a100:1
#SBATCH --time=00:00:00
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

source /etc/profile
source /shared/home/PARADIS/PathBench-MIL/pathbench_env/bin/activate

cd /shared/home/jorgarcia/PathBench-MIL
mkdir -p slurm_logs 

./run_pathbench.sh ./config/conf_brca_virchow2_optuna.yaml