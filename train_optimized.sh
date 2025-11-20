#!/bin/bash
# 优化版训练脚本

set -e

DATASET="ISCXVPN2016"
BATCH_SIZE=512
NUM_WORKERS=8
EPOCHS_TEACHER=15
EPOCHS_STACKING=12
EPOCHS_STUDENT=50

# 优化的超参数
TEMPERATURE=3.0
LAMB_CE=1.5
LAMB_F=0.5
LAMB_R=0.5
LAMB_S=0.05

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║         NetKD 优化训练 - 改进超参数版本                 ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}📊 训练配置:${NC}"
echo "  • 数据集: $DATASET"
echo "  • Batch Size: $BATCH_SIZE"
echo "  • Num Workers: $NUM_WORKERS"
echo "  • 教师模型轮数: $EPOCHS_TEACHER"
echo "  • Stacking轮数: $EPOCHS_STACKING"
echo "  • 学生模型轮数: $EPOCHS_STUDENT (↑ 从25)"
echo ""
echo -e "${BLUE}🎯 优化的蒸馏参数:${NC}"
echo "  • 温度 T: $TEMPERATURE (↓ 从4.0)"
echo "  • CE权重: $LAMB_CE (↑ 从1.0)"
echo "  • Forward KL: $LAMB_F"
echo "  • Reverse KL: $LAMB_R"
echo "  • Sinkhorn: $LAMB_S (↓ 从0.1)"
echo ""
echo -e "${BLUE}✨ 新增特性:${NC}"
echo "  • CosineAnnealingLR 学习率调度"
echo "  • Early Stopping (patience=10)"
echo "  • Pin Memory + Persistent Workers"
echo ""

if ! conda env list | grep -q "netkd"; then
    echo -e "${YELLOW}⚠️  警告: netkd环境不存在${NC}"
    exit 1
fi

echo -e "${GREEN}✅ 开始训练 (使用已有的教师模型)${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 只训练学生模型（复用之前的教师模型）
conda run -n netkd python train_with_real_data.py \
  --dataset "$DATASET" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --epochs_teacher "$EPOCHS_TEACHER" \
  --epochs_stacking "$EPOCHS_STACKING" \
  --epochs_student "$EPOCHS_STUDENT" \
  --temperature "$TEMPERATURE" \
  --lamb_ce "$LAMB_CE" \
  --lamb_f "$LAMB_F" \
  --lamb_r "$LAMB_R" \
  --lamb_s "$LAMB_S" \
  --mode full_pipeline

echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                  优化训练完成!                           ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
