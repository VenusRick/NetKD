#!/bin/bash
DATASETS=("CICIoT2022" "USTC-TFC2016" "ISCXTor2016" "ISCXVPN2016" "Bing-Traffic" "Malware-Traffic")
for ds in "${DATASETS[@]}"; do
  echo "=== $ds ==="
  CUDA_VISIBLE_DEVICES=0 python training/train.py --mode train_teachers --use_real_data --dataset $ds --dataset_root Dataset --batch_size 128 --epochs_teacher 30 --output_dir results/$ds --teacher_pretrained > logs/${ds}_t_$(date +%s).log 2>&1
  CUDA_VISIBLE_DEVICES=1 python training/train.py --mode train_stacking --use_real_data --dataset $ds --dataset_root Dataset --batch_size 128 --epochs_stacking 20 --output_dir results/$ds > logs/${ds}_st_$(date +%s).log 2>&1
  CUDA_VISIBLE_DEVICES=2 python training/train.py --mode train_student --use_real_data --dataset $ds --dataset_root Dataset --batch_size 128 --epochs_student 50 --distill_mode full --temperature 4.0 --lamb_ce 0.3 --lamb_f 0.3 --lamb_r 0.2 --lamb_s 0.2 --output_dir results/$ds > logs/${ds}_stu_$(date +%s).log 2>&1 &
done
wait
