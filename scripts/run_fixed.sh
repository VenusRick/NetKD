#!/bin/bash
DATASETS=("CICIoT2022" "USTC-TFC2016" "ISCXTor2016" "ISCXVPN2016" "Bing-Traffic" "Malware-Traffic")
GPUS=(0 1 2)
OUTPUT_DIR="results/full_dataset_runs_$(date +%Y%m%d_%H%M%S)"

mkdir -p $OUTPUT_DIR
mkdir -p logs

for idx in "${!DATASETS[@]}"; do
    dataset="${DATASETS[$idx]}"
    gpu_idx=$((idx % ${#GPUS[@]}))
    gpu="${GPUS[$gpu_idx]}"
    log_file="logs/${dataset}_$(date +%Y%m%d_%H%M%S).log"
    
    echo "[$(date)] Training $dataset on GPU $gpu"
    
    CUDA_VISIBLE_DEVICES=$gpu python training/train.py \
        --use_real_data \
        --dataset "$dataset" \
        --output_dir "$OUTPUT_DIR/$dataset" \
        --epochs_teacher 30 \
        --epochs_stacking 20 \
        --epochs_student 50 \
        --batch_size 128 \
        --lr 0.001 \
        --teacher_pretrained \
        > "$log_file" 2>&1
    
    echo "[$(date)] Completed $dataset (exit: $?)"
    sleep 3
done

echo "All training complete!"
