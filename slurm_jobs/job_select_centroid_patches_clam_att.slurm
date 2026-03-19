#!/bin/bash
#SBATCH --job-name=centroid_patches_clam_att
#SBATCH --output=slurm_logs/centroid_patches_clam_att_%j.out
#SBATCH --error=slurm_logs/centroid_patches_clam_att_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

source /shared/home/PARADIS/PathBench-MIL/pathbench_env/bin/activate

cd /shared/home/jorgarcia/PathBench-MIL

python scripts/select_centroid_patches_clam_att.py \
    --top_k 8 \
    --n_closest 25 \
    --tasks er erbb2 pr pam50 \
    --output_dir results/centroid_patches_clam_att