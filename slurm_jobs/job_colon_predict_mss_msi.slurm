#!/bin/bash
#SBATCH --job-name=col_pred_mss
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
#SBATCH --nodelist=gpu06
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:00:00

source /etc/profile
source /shared/home/PARADIS/PathBench-MIL/pathbench_env/bin/activate
cd /shared/home/jorgarcia/PathBench-MIL
mkdir -p slurm_logs results/colon

MODEL=experiments/colon_mss_msi_benchmark/mil/00000-256_128_none_h_optimus_0_dsmil_CrossEntropyLoss_ReLU_Adam_1
BAGS_NONE=/home/PARADIS/colon/tfrecords/shared_bags/256_128_none_h_optimus_0

# Macarena
python scripts/breast/predict_slides.py \
    --model_dir $MODEL \
    --config config/conf_colon_mss_msi_benchmark.yaml \
    --bags_dir $BAGS_NONE \
    --annotations config/annotations/annotations_colon_mss_msi.csv \
    --output results/colon/predictions_macarena.csv

# CPTAC-COAD
python scripts/breast/predict_slides.py \
    --model_dir $MODEL \
    --config config/conf_colon_mss_msi_benchmark.yaml \
    --bags_dir $BAGS_NONE \
    --annotations config/annotations/annotations_cptac_coad_mss_msi.csv \
    --output results/colon/predictions_cptac.csv

# TCGA-COAD
python scripts/breast/predict_slides.py \
    --model_dir $MODEL \
    --config config/conf_colon_mss_msi_benchmark.yaml \
    --bags_dir $BAGS_NONE \
    --annotations config/annotations/annotations_tcga_coad_mss_msi.csv \
    --output results/colon/predictions_tcga.csv