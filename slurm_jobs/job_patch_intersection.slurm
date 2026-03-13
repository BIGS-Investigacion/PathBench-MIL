#!/bin/bash
#SBATCH --job-name=patch_inter
#SBATCH --output=slurm_logs/patch_intersection_%j.out
#SBATCH --error=slurm_logs/patch_intersection_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

source /shared/home/PARADIS/PathBench-MIL/pathbench_env/bin/activate

cd /shared/home/jorgarcia/PathBench-MIL

python scripts/patch_intersection.py \
    --tasks er erbb2 pr pam50 \
    --top_k 8 \
    --output_dir results/patch_intersection