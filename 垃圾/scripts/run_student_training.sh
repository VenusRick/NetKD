#!/bin/bash
echo "========================================="
echo "使用已有98%+教师模型训练学生"
echo "========================================="

# 确认教师模型存在
echo "检查教师模型..."
ls -lh checkpoints/resnet50_teacher.pth checkpoints/mbv3_teacher.pth checkpoints/densenet121_teacher.pth checkpoints/stacking_model.pth

# 运行原有的成功训练pipeline - 只训练学生部分
python train_with_real_data.py \
  --dataset ISCXVPN2016 \
  --data_root /walnut_data/yqm/Dataset \
  --batch_size 256 \
  --num_workers 0 \
  --epochs_teacher 0 \
  --epochs_stacking 0 \
  --epochs_student 100 \
  --lr 0.001 \
  --weight_decay 0.0001 \
  --temperature 4.0 \
  --output_dir ./checkpoints \
  --mode student_only \
  --student_ckpt student_baseline_fixed.pth

echo "学生模型训练完成!"
