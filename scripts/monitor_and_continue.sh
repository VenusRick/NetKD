#!/bin/bash

NETKD_ROOT="/walnut_data/yqm/NetKD"
cd "$NETKD_ROOT"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === 监控 Student bs=128 训练 ===" >> logs/auto_monitor.log

# 等待 bs=128 完成
while pgrep -f "train.py.*train_student.*batch_size 128" > /dev/null; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Student bs=128 训练中..." >> logs/auto_monitor.log
    sleep 60
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Student bs=128 完成,提取结果..." >> logs/auto_monitor.log
ACC_128=$(grep -oP "val_acc=\K[0-9.]+" logs/eca_student_bs128.log | tail -n 1)
LOSS_128=$(grep -oP "val_loss=\K[0-9.]+" logs/eca_student_bs128.log | tail -n 1)
echo "Student_bs128,$ACC_128,$LOSS_128" >> checkpoints/eca_pipeline/training_results.csv
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Student bs=128 - Acc: $ACC_128, Loss: $LOSS_128" >> logs/auto_monitor.log

# 启动 bs=256
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 启动 Student bs=256 训练..." >> logs/auto_monitor.log
CUDA_VISIBLE_DEVICES=0 /root/miniconda3/bin/conda run -p /root/miniconda3 --no-capture-output \
    python training/train.py \
    --use_real_data \
    --mode train_student \
    --dataset ISCXVPN2016 \
    --dataset_root /walnut_data/yqm/Dataset \
    --batch_size 256 \
    --num_workers 4 \
    --epochs_student 100 \
    --output_dir checkpoints/eca_pipeline \
    --resnet_use_eca \
    --mbv3_use_eca \
    > logs/eca_student_bs256.log 2>&1 &

sleep 10

# 等待 bs=256 完成
while pgrep -f "train.py.*train_student.*batch_size 256" > /dev/null; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Student bs=256 训练中..." >> logs/auto_monitor.log
    sleep 60
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Student bs=256 完成,提取结果..." >> logs/auto_monitor.log
ACC_256=$(grep -oP "val_acc=\K[0-9.]+" logs/eca_student_bs256.log | tail -n 1)
LOSS_256=$(grep -oP "val_loss=\K[0-9.]+" logs/eca_student_bs256.log | tail -n 1)
echo "Student_bs256,$ACC_256,$LOSS_256" >> checkpoints/eca_pipeline/training_results.csv
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Student bs=256 - Acc: $ACC_256, Loss: $LOSS_256" >> logs/auto_monitor.log

# 启动 bs=512
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 启动 Student bs=512 训练..." >> logs/auto_monitor.log
CUDA_VISIBLE_DEVICES=0 /root/miniconda3/bin/conda run -p /root/miniconda3 --no-capture-output \
    python training/train.py \
    --use_real_data \
    --mode train_student \
    --dataset ISCXVPN2016 \
    --dataset_root /walnut_data/yqm/Dataset \
    --batch_size 512 \
    --num_workers 4 \
    --epochs_student 100 \
    --output_dir checkpoints/eca_pipeline \
    --resnet_use_eca \
    --mbv3_use_eca \
    > logs/eca_student_bs512.log 2>&1 &

sleep 10

# 等待 bs=512 完成
while pgrep -f "train.py.*train_student.*batch_size 512" > /dev/null; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Student bs=512 训练中..." >> logs/auto_monitor.log
    sleep 60
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Student bs=512 完成,提取结果..." >> logs/auto_monitor.log
ACC_512=$(grep -oP "val_acc=\K[0-9.]+" logs/eca_student_bs512.log | tail -n 1)
LOSS_512=$(grep -oP "val_loss=\K[0-9.]+" logs/eca_student_bs512.log | tail -n 1)
echo "Student_bs512,$ACC_512,$LOSS_512" >> checkpoints/eca_pipeline/training_results.csv
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Student bs=512 - Acc: $ACC_512, Loss: $LOSS_512" >> logs/auto_monitor.log

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === 所有训练完成! ===" >> logs/auto_monitor.log
