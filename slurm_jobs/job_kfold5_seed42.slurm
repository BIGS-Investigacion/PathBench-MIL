#!/bin/bash
#SBATCH --job-name=kf5_s42
#SBATCH --output=slurm_logs/%j_kf5_s42.out
#SBATCH --error=slurm_logs/%j_kf5_s42.err
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00

source /etc/profile
cd /shared/home/jorgarcia/PathBench-MIL
mkdir -p slurm_logs
./run_pathbench_recovered.sh ./config/conf_kfold5_seed42.yaml
