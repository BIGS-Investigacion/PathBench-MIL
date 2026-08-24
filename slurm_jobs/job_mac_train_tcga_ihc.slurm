#!/bin/bash
#SBATCH --job-name=mac_tcga_ihc
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
#SBATCH --nodelist=gpu04
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

source /etc/profile
cd /shared/home/jorgarcia/PathBench-MIL
mkdir -p slurm_logs
./run_pathbench_recovered.sh ./config/conf_colon_mss_msi_mac_train_tcga_ihc.yaml
