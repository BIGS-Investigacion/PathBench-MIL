#!/bin/bash
#SBATCH --job-name=extract_patches
#SBATCH --output=slurm_logs/extract_top_patches_%j.out
#SBATCH --error=slurm_logs/extract_top_patches_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

source /shared/home/PARADIS/PathBench-MIL/pathbench_env/bin/activate

cd /shared/home/jorgarcia/PathBench-MIL
mkdir -p slurm_logs

python scripts/extract_top_patches.py \
    --tasks er erbb2 pr pam50 \
    --top_k 8 \
    --n_closest 25 \
    --tile_px 256 \
    --output_dir results/patch_intersection
