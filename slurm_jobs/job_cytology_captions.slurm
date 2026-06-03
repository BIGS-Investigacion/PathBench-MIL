#!/bin/bash
#SBATCH --job-name=cyto_captions
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:00:00

source /etc/profile
source /shared/home/PARADIS/PathBench-MIL/pathbench_env/bin/activate
cd /shared/home/jorgarcia/PathBench-MIL
mkdir -p slurm_logs

python scripts/cytology/caption_cytology.py