#!/bin/bash
#SBATCH --job-name=patch_inter_all
#SBATCH --output=/shared/home/jorgarcia/PathBench-MIL/slurm_logs/patch_inter_all_%j.out
#SBATCH --error=/shared/home/jorgarcia/PathBench-MIL/slurm_logs/patch_inter_all_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=main

source /shared/home/PARADIS/PathBench-MIL/pathbench_env/bin/activate
cd /shared/home/jorgarcia/PathBench-MIL

python scripts/patch_intersection_all.py \
    --top_k 8 \
    --output_dir results/patch_intersection_all