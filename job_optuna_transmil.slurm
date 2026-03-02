#!/bin/bash
#SBATCH --job-name=optuna_transmil
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=a100:1
#SBATCH --time=00:00:00
#SBATCH --output=slurm/%j.out
#SBATCH --error=slurm/%j.err

mkdir -p slurm

source /etc/profile
source /shared/home/PARADIS/PathBench-MIL/pathbench_env/bin/activate

cd /shared/home/jorgarcia/PathBench-MIL

./run_pathbench.sh ./config/conf_brca_pam50_virchow2_optuna_transmil.yaml