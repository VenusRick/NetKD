#!/bin/bash
# Complete Training Script for All Datasets
# Using the existing training/train.py infrastructure

DATASETS=("CICIoT2022" "USTC-TFC2016" "ISCXTor2016" "ISCXVPN2016" "Bing-Traffic" "Malware-Traffic")
GPUS=(0 1 2)
OUTPUT_DIR="results/full_dataset_runs_$(date +%Y%m%d_%H%M%S)"

mkdir -p $OUTPUT_DIR
mkdir -p logs

echo "=================================================================================="
echo "Starting Full Dataset Training Pipeline"
echo "Output Directory: $OUTPUT_DIR"
echo "=================================================================================="

# Function to run training on a dataset
run_dataset_training() {
    local dataset=$1
    local gpu=$2
    local log_file="logs/${dataset}_training_$(date +%Y%m%d_%H%M%S).log"
    
    echo "[$(date)] Starting training for $dataset on GPU $gpu"
    
    CUDA_VISIBLE_DEVICES=$gpu python training/train.py \
        --dataset_A "$dataset" \
        --dataset_B "$dataset" \
        --output_dir "$OUTPUT_DIR/$dataset" \
        --num_epochs_teacher 30 \
        --num_epochs_stacking 20 \
        --num_epochs_student 50 \
        --batch_size 128 \
        --lr 0.001 \
        > "$log_file" 2>&1
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "[$(date)] ✓ Training completed for $dataset"
    else
        echo "[$(date)] ✗ Training failed for $dataset (exit code: $exit_code)"
    fi
    
    return $exit_code
}

# Run training for each dataset, cycling through GPUs
for idx in "${!DATASETS[@]}"; do
    dataset="${DATASETS[$idx]}"
    gpu_idx=$((idx % ${#GPUS[@]}))
    gpu="${GPUS[$gpu_idx]}"
    
    echo ""
    echo "=================================================================================="
    echo "Dataset: $dataset (GPU $gpu)"
    echo "=================================================================================="
    
    run_dataset_training "$dataset" "$gpu"
    
    # Brief pause between datasets
    sleep 5
done

echo ""
echo "=================================================================================="
echo "All Dataset Training Complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=================================================================================="

# Generate summary
python - << PYEOF
import os
import json
from pathlib import Path

output_dir = Path("$OUTPUT_DIR")
summary = {
    "timestamp": "$(date --iso-8601=seconds)",
    "datasets": [],
    "total_trained": 0,
    "total_failed": 0
}

for dataset in ["CICIoT2022", "USTC-TFC2016", "ISCXTor2016", "ISCXVPN2016", "Bing-Traffic", "Malware-Traffic"]:
    dataset_dir = output_dir / dataset
    if dataset_dir.exists():
        summary["datasets"].append({
            "name": dataset,
            "status": "completed",
            "output_dir": str(dataset_dir)
        })
        summary["total_trained"] += 1
    else:
        summary["datasets"].append({
            "name": dataset,
            "status": "failed"
        })
        summary["total_failed"] += 1

with open(output_dir / "training_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nSummary: {summary['total_trained']} trained, {summary['total_failed']} failed")
PYEOF

