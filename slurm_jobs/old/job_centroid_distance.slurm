#!/bin/bash
#SBATCH --job-name=centroid_dist
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=a100:1
#SBATCH --time=00:00:00
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

source /etc/profile
source /shared/home/PARADIS/PathBench-MIL/pathbench_env/bin/activate

cd /shared/home/jorgarcia/PathBench-MIL
mkdir -p slurm_logs 

python scripts/centroid_distance.py \
    --model_dir experiments/brca_pam50_virchow2_ho_tcga_tcga/mil/00000-256_128_macenko_virchow2_clam_mil_mb_CrossEntropyLoss_ReLU_AdamW_1 \
    --bags_dir /home/PARADIS/mama/tfrecords/shared_bags/256_128_macenko_virchow2 \
    --annotations config/annotations/annotations_brca_pam50.csv \
    --labels 0:Basal 1:Her2 2:LumA 3:LumB 4:Normal \
    --top_k_attn 8 \
    --top_n_images 25 \
    --output centroid_distances_macenko.csv \
    --patches_dir top_patches/macenko \
    --tfrecord_dirs /home/PARADIS/mama/tfrecords/TCGA-BRCA/256px_128um \
                    /home/PARADIS/mama/tfrecords/CPTAC-BRCA/256px_128um