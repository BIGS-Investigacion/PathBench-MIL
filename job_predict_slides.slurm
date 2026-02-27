#!/bin/bash
#SBATCH --job-name=predict_slides
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=a100:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm/%j.out
#SBATCH --error=slurm/%j.err

mkdir -p slurm

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SF_SLIDE_BACKEND=cucim
export SF_BACKEND=torch

source /etc/profile
source /shared/home/PARADIS/PathBench-MIL/pathbench_env/bin/activate

python scripts/predict_slides.py \
    --model_dir experiments/brca_pam50_virchow2_ho_tcga_tcga_slide_no_weighting/mil/00000-256_128_macenko_virchow2_clam_mil_mb_CrossEntropyLoss_ReLU_AdamW_1 \
    --config config/conf_brca_pam50_virchow2_test.yaml \
    --bags_dir /home/PARADIS/mama/tfrecords/shared_bags/256_128_macenko_virchow2 \
    --annotations config/annotations/annotations_brca_pam50.csv \
    --output predictions_per_slide_macenko_imbalance.csv