#!/bin/bash
#SBATCH --job-name=pathbench_mccv                             # create a short name for your job
#SBATCH --nodes=1                                     # node count
#SBATCH --ntasks=1                                    # total number of tasks across all nodes
#SBATCH --cpus-per-task=8                             # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=32G                                     # total memory per node (30 training runs, 3 MIL x 10 folds)
#SBATCH --gpus=a100:1
#SBATCH --time=00:00:00
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

source /etc/profile
source /shared/home/PARADIS/PathBench-MIL/pathbench_env/bin/activate

cd /shared/home/jorgarcia/PathBench-MIL
mkdir -p slurm_logs 

./run_pathbench.sh ./config/conf_brca_pam50_virchow2_kfold10.yaml
