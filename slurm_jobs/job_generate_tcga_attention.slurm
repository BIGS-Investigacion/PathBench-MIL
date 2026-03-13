#!/bin/bash
#SBATCH --job-name=tcga_att
#SBATCH --output=slurm_logs/tcga_attention_%j.out
#SBATCH --error=slurm_logs/tcga_attention_%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

source /shared/home/PARADIS/PathBench-MIL/pathbench_env/bin/activate

cd /shared/home/jorgarcia/PathBench-MIL

python scripts/generate_tcga_attention.py \
    --tasks er erbb2 pr pam50 \
    --mils transmil dsmil clam_mil_mb