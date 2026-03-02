#!/bin/bash
#SBATCH --job-name=pathbench_ho                             # create a short name for your job
#SBATCH --nodes=1                                     # node count
#SBATCH --ntasks=1                                    # total number of tasks across all nodes
#SBATCH --cpus-per-task=8                             # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=32G                                     # total memory per node (30 training runs, 3 MIL x 10 folds)
#SBATCH --gpus=a100:1                                  # gpus per node (a40 sufficient for MIL training)
#SBATCH --time=00:00:00                               # max time
#SBATCH --output=slurm/%j.out
#SBATCH --error=slurm/%j.err
source /etc/profile
source /shared/home/PARADIS/PathBench-MIL/pathbench_env/bin/activate

cd /shared/home/jorgarcia/PathBench-MIL
mkdir -p slurm_logs 

./run_pathbench.sh ./config/conf_brca_pam50_virchow2_test.yaml
