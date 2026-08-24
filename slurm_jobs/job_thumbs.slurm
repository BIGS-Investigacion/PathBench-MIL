#!/bin/bash
#SBATCH --job-name=wsi_thumbs
#SBATCH --output=slurm_logs/%j_thumbs.out
#SBATCH --error=slurm_logs/%j_thumbs.err
#SBATCH --nodelist=c08
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
source /shared/home/jorgarcia/monai_qupath/.venv/bin/activate
cd /shared/home/jorgarcia/PathBench-MIL
python3 scripts/colon/gen_wsi_thumbs.py
