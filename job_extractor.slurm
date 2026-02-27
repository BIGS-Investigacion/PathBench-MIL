#!/bin/bash
#SBATCH --job-name=pathbench_mil                                # create a short name for your job
#SBATCH --nodes=1                                     # node count
#SBATCH --ntasks=1                                    # total number of tasks across all nodes
#SBATCH --cpus-per-task=8                             # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=4G                                     # total memory per node (4 GB per cpu-core is default)
#SBATCH --gpus=a30:1                                   # gpus per node
#SBATCH --time=00:00:00                               # max time
#SBATCH --output=slurm/%j.out
#SBATCH --error=slurm/%j.err

mkdir -p slurm

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /etc/profile

./run_pathbench.sh ./config/conf_brca_pam50_virchow2.yaml
