#!/bin/bash
#SBATCH --job-name=pathbench_mil                                # create a short name for your job
#SBATCH --nodes=1                                     # node count
#SBATCH --ntasks=1                                    # total number of tasks across all nodes
#SBATCH --cpus-per-task=1                             # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=16G                                     # total memory per node (4 GB per cpu-core is default)
#SBATCH --gpus=a100                                   # gpus per node
#SBATCH --time=00:00:00                               # max time
#SBATCH --output=slurm/%j.out
#SBATCH --error=slurm/%j.err

mkdir -p slurm

source /etc/profile

./run_pathbench.sh ./config/conf.yaml
