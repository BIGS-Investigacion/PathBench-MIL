#!/bin/bash
#SBATCH --job-name=shift_pair
#SBATCH --output=slurm_logs/%j_shift_pair.out
#SBATCH --error=slurm_logs/%j_shift_pair.err
#SBATCH --nodelist=c07
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

source /shared/home/jorgarcia/monai_qupath/.venv/bin/activate
cd /shared/home/jorgarcia/PathBench-MIL
mkdir -p slurm_logs
echo "###### TCGA -> CPTAC ######"
python3 scripts/colon/eval_shift_pair.py --source tcga_coad --target cptac_coad --n_pca 128
echo ""
echo "###### MACARENA -> TCGA ######"
python3 scripts/colon/eval_shift_pair.py --source macarena --target tcga_coad --n_pca 128
