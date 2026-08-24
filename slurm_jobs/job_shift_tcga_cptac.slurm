#!/bin/bash
#SBATCH --job-name=shift_tcga_cptac
#SBATCH --output=slurm_logs/%j_shift_tcga_cptac.out
#SBATCH --error=slurm_logs/%j_shift_tcga_cptac.err
#SBATCH --nodelist=c07
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

source /shared/home/jorgarcia/monai_qupath/.venv/bin/activate
cd /shared/home/jorgarcia/PathBench-MIL
mkdir -p slurm_logs
echo "############## TCGA-PCR -> CPTAC-PCR (none + macenko) ##############"
python3 scripts/colon/eval_shift_pair.py --source tcga_coad --target cptac_coad \
    --label CPTAC-PCR --n_pca 128
