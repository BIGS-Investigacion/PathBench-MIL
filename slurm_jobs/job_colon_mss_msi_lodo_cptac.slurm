#!/bin/bash
#SBATCH --job-name=col_lodo_cptac
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
#SBATCH --gpus=a40:1
#SBATCH --nodelist=gpu04
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:00:00

source /etc/profile
source /shared/home/PARADIS/PathBench-MIL/pathbench_env/bin/activate
cd /shared/home/jorgarcia/PathBench-MIL
mkdir -p slurm_logs

./run_pathbench.sh ./config/conf_colon_mss_msi_lodo_cptac.yaml
