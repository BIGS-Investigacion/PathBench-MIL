#!/bin/bash
#SBATCH --job-name=col_mss_tcgatrain
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
#SBATCH --nodelist=gpu04
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

source /etc/profile
cd /shared/home/jorgarcia/PathBench-MIL
mkdir -p slurm_logs

./run_pathbench_recovered.sh ./config/conf_colon_mss_msi_tcga_train.yaml
