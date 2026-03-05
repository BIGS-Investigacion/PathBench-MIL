#!/bin/bash
#SBATCH --job-name=predict_slides
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=a100:1
#SBATCH --time=00:00:00
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

source /etc/profile
source /shared/home/PARADIS/PathBench-MIL/pathbench_env/bin/activate

cd /shared/home/jorgarcia/PathBench-MIL
mkdir -p slurm_logs 

python scripts/predict_slides.py \
    --model_dir /home/jorgarcia/PathBench-MIL/experiments/brca_pam50_virchow2_ho_category/mil/00000-256_128_none_virchow2_clam_mil_mb_CrossEntropyLoss_ReLU_Adam_1 \
    --config config/conf_brca_pam50_virchow2_test.yaml \
    --bags_dir /home/PARADIS/mama/tfrecords/shared_bags/256_128_none_virchow2 \
    --annotations config/annotations/annotations_brca_pam50.csv \
    --output predictions_per_slide_clam_none.csv