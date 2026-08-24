#!/bin/bash
#SBATCH --job-name=shift_mac_tcga
#SBATCH --output=slurm_logs/%j_shift_mac_tcga.out
#SBATCH --error=slurm_logs/%j_shift_mac_tcga.err
#SBATCH --nodelist=c07
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

source /shared/home/jorgarcia/monai_qupath/.venv/bin/activate
cd /shared/home/jorgarcia/PathBench-MIL
mkdir -p slurm_logs
echo "############## VERSIÓN 1: Macarena-IHC -> TCGA-IHC (25/16) ##############"
python3 scripts/colon/eval_shift_pair.py --source macarena \
    --target_ann config/annotations/annotations_tcga_coad_mss_msi_ihc.csv \
    --label TCGA-IHC --n_pca 128
echo ""
echo "############## VERSIÓN 2: Macarena-IHC -> TCGA-PCR (207/35) ##############"
python3 scripts/colon/eval_shift_pair.py --source macarena --target tcga_coad \
    --label TCGA-PCR --n_pca 128
