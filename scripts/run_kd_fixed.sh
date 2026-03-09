#!/bin/bash
source ~/.bashrc
conda activate NetKD
DATASETS=("CICIoT2022" "USTC-TFC2016")
for ds in "${DATASETS[@]}"; do
  echo "=== $ds ==="
  CUDA_VISIBLE_DEVICES=0 python training/train.py --mode train_teachers --use_real_data --dataset $ds --dataset_root Dataset --batch_size 128 --epochs_teacher 30 --output_dir results/$ds --teacher_pretrained &
  sleep 60
done
wait
