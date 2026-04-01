#!/bin/bash
#SBATCH --job-name=tcga_coad_dl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

source /etc/profile

cd /shared/home/jorgarcia/PathBench-MIL
mkdir -p slurm_logs

bash scripts/colon/download_tcga_coad_slides.sh