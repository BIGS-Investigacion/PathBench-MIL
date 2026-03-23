#!/bin/bash
#SBATCH --job-name=patch_overlap_v2
#SBATCH --output=slurm_logs/patch_overlap_v2_%j.out
#SBATCH --error=slurm_logs/patch_overlap_v2_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

source /shared/home/PARADIS/PathBench-MIL/pathbench_env/bin/activate

cd /shared/home/jorgarcia/PathBench-MIL

python scripts/compare_patch_overlap_virchow2.py \
    --threshold 1.0 \
    --batch_size 32 \
    --tasks er erbb2 pr pam50