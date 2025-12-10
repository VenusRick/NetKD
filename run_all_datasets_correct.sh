#!/bin/bash
# 正确的全数据集训练脚本 - 三阶段KD流程

DATASET_ROOT=\ /workspace/yqm/Dataset\
OUTPUT_BASE=\results/full_run_\
BATCH_SIZE=128

# 数据集列表
DATASETS=(\CICIoT2022\ \USTC-TFC2016\ \ISCXTor2016\ \ISCXVPN2016\)

# GPU分配
declare -A GPU_MAP
GPU_MAP[\CICIoT2022\]=0
GPU_MAP[\USTC-TFC2016\]=1
GPU_MAP[\ISCXTor2016\]=2
GPU_MAP[\ISCXVPN2016\]=0

echo \==================================================================================\
echo \Starting Full KD Pipeline - 12/09/2025 22:55:45 \
echo \Output: \\
echo \==================================================================================\

# 并行训练函数
train_dataset() {
    local dataset=\
    local gpu=\
    local output_dir=\\/\\
    
    echo \[\12/09/2025 22:55:45] Starting \ on GPU \\
    
    # Stage 1: Train Teachers
    CUDA_VISIBLE_DEVICES=\ python training/train.py \\
        --mode train_teachers \\
        --use_real_data \\
        --dataset \\\ \\
        --dataset_root \\\ \\
        --batch_size \ \\
        --epochs_teacher 30 \\
        --output_dir \\\ \\
        --teacher_pretrained \\
        2>&1 | tee \\/teacher_train.log\
    
    if [ \True -ne 0 ]; then
        echo \[\12/09/2025 22:55:45] ERROR: Teacher training failed for \\
        return 1
    fi
    
    # Stage 2: Train Stacking Meta-Learner
    CUDA_VISIBLE_DEVICES=\ python training/train.py \\
        --mode train_stacking \\
        --use_real_data \\
        --dataset \\\ \\
        --dataset_root \\\ \\
        --batch_size \ \\
        --epochs_stacking 20 \\
        --output_dir \\\ \\
        2>&1 | tee \\/stacking_train.log\
    
    if [ \True -ne 0 ]; then
        echo \[\12/09/2025 22:55:45] ERROR: Stacking training failed for \\
        return 1
    fi
    
    # Stage 3: Train Student with KD
    CUDA_VISIBLE_DEVICES=\ python training/train.py \\
        --mode train_student \\
        --use_real_data \\
        --dataset \\\ \\
        --dataset_root \\\ \\
        --batch_size \ \\
        --epochs_student 50 \\
        --distill_mode full \\
        --output_dir \\\ \\
        2>&1 | tee \\/student_train.log\
    
    if [ \True -ne 0 ]; then
        echo \[\12/09/2025 22:55:45] ERROR: Student training failed for \\
        return 1
    fi
    
    echo \[\12/09/2025 22:55:45] COMPLETED: \\
    return 0
}

# 创建输出目录
mkdir -p \\\

# 并行启动训练 (CICIoT2022 GPU0, USTC GPU1, ISCXTor GPU2)
train_dataset \CICIoT2022\ 0 &
PID_CIC=\$!

train_dataset \USTC-TFC2016\ 1 &
PID_USTC=\$!

train_dataset \ISCXTor2016\ 2 &
PID_TOR=\$!

# 等待第一批完成
wait \
wait \  
wait \

# 第二批 ISCXVPN2016
train_dataset \ISCXVPN2016\ 0

echo \==================================================================================\
echo \All Training Complete - 12/09/2025 22:55:45 \
echo \==================================================================================\
