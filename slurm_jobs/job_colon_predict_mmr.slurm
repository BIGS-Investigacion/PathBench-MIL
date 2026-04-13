#!/bin/bash
#SBATCH --job-name=col_predict_mmr
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
mkdir -p slurm_logs results/colon_mmr

# MLH1 — clam_mil_mb, none
python scripts/breast/predict_slides.py \
    --model_dir experiments/colon_mlh1_benchmark/mil/00000-256_128_none_h_optimus_0_clam_mil_mb_CrossEntropyLoss_ReLU_Adam_1 \
    --config config/conf_colon_mlh1_benchmark.yaml \
    --bags_dir /home/PARADIS/colon/tfrecords/shared_bags/256_128_none_h_optimus_0 \
    --annotations config/annotations/annotations_colon_mlh1_all.csv \
    --output results/colon_mmr/predictions_mlh1.csv

# MSH2 — clam_mil_mb, none
python scripts/breast/predict_slides.py \
    --model_dir experiments/colon_msh2_benchmark/mil/00000-256_128_none_h_optimus_0_clam_mil_mb_CrossEntropyLoss_ReLU_Adam_1 \
    --config config/conf_colon_msh2_benchmark.yaml \
    --bags_dir /home/PARADIS/colon/tfrecords/shared_bags/256_128_none_h_optimus_0 \
    --annotations config/annotations/annotations_colon_msh2_all.csv \
    --output results/colon_mmr/predictions_msh2.csv

# MSH6 — transmil, none
python scripts/breast/predict_slides.py \
    --model_dir experiments/colon_msh6_benchmark/mil/00000-256_128_none_h_optimus_0_transmil_CrossEntropyLoss_ReLU_Adam_1 \
    --config config/conf_colon_msh6_benchmark.yaml \
    --bags_dir /home/PARADIS/colon/tfrecords/shared_bags/256_128_none_h_optimus_0 \
    --annotations config/annotations/annotations_colon_msh6_all.csv \
    --output results/colon_mmr/predictions_msh6.csv

# PMS2 — transmil, macenko
python scripts/breast/predict_slides.py \
    --model_dir experiments/colon_pms2_benchmark/mil/00000-256_128_macenko_h_optimus_0_transmil_CrossEntropyLoss_ReLU_Adam_1 \
    --config config/conf_colon_pms2_benchmark.yaml \
    --bags_dir /home/PARADIS/colon/tfrecords/shared_bags/256_128_macenko_h_optimus_0 \
    --annotations config/annotations/annotations_colon_pms2_all.csv \
    --output results/colon_mmr/predictions_pms2.csv