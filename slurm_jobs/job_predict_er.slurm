#!/bin/bash
#SBATCH --job-name=predict_er
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=a100:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

source /etc/profile
source /shared/home/PARADIS/PathBench-MIL/pathbench_env/bin/activate

cd /shared/home/jorgarcia/PathBench-MIL
mkdir -p slurm_logs/results

RESULTS=slurm_logs/results/er_results.txt
ANNOTATIONS=config/annotations/annotations_brca_er.csv
EXPBASE=/shared/home/jorgarcia/PathBench-MIL/experiments
BAGS_MACENKO=/home/PARADIS/mama/tfrecords/shared_bags/256_128_macenko_virchow2
BAGS_NONE=/home/PARADIS/mama/tfrecords/shared_bags/256_128_none_virchow2

echo "========================================" > "$RESULTS"
echo "TASK: ER  -  $(date)" >> "$RESULTS"
echo "========================================" >> "$RESULTS"

for MIL in clam_mil_mb dsmil transmil; do
    for NORM in macenko none; do
        if [ "$NORM" = "macenko" ]; then
            IDX="00000"
            BAGS_DIR="$BAGS_MACENKO"
        else
            IDX="00001"
            BAGS_DIR="$BAGS_NONE"
        fi

        CONFIG="$EXPBASE/brca_virchow2_test_er_${MIL}/conf_brca_virchow2_test.yaml"
        MODEL_DIR="$EXPBASE/brca_virchow2_test_er_${MIL}/mil/${IDX}-256_128_${NORM}_virchow2_${MIL}_CrossEntropyLoss_ReLU_Adam_1"
        PRED_CSV="slurm_logs/results/er_${MIL}_${NORM}_predictions.csv"

        echo "" >> "$RESULTS"
        echo "----------------------------------------" >> "$RESULTS"
        echo "MIL: $MIL  |  Norm: $NORM" >> "$RESULTS"
        echo "Model: $MODEL_DIR" >> "$RESULTS"
        echo "----------------------------------------" >> "$RESULTS"

        python scripts/predict_slides.py \
            --model_dir "$MODEL_DIR" \
            --config "$CONFIG" \
            --bags_dir "$BAGS_DIR" \
            --annotations "$ANNOTATIONS" \
            --output "$PRED_CSV" >> "$RESULTS"

        echo "" >> "$RESULTS"
    done
done

echo "" >> "$RESULTS"
echo "========================================" >> "$RESULTS"
echo "DONE: $(date)" >> "$RESULTS"