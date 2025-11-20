#!/bin/bash
# NetKD训练启动脚本（实时输出版本）

set -e

# 配置参数
DATASET="ISCXVPN2016"
BATCH_SIZE=512
NUM_WORKERS=8
EPOCHS_TEACHER=15
EPOCHS_STACKING=12
EPOCHS_STUDENT=25

# 彩色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     NetKD 高GPU利用率训练 - 实时输出模式                ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}📊 训练配置:${NC}"
echo "  • 数据集: $DATASET"
echo "  • Batch Size: $BATCH_SIZE"
echo "  • Num Workers: $NUM_WORKERS"
echo "  • 教师模型轮数: $EPOCHS_TEACHER"
echo "  • Stacking轮数: $EPOCHS_STACKING"
echo "  • 学生模型轮数: $EPOCHS_STUDENT"
echo ""
echo -e "${BLUE}🎯 优化特性:${NC}"
echo "  • Pin Memory: True"
echo "  • Persistent Workers: True"
echo "  • Prefetch Factor: 2"
echo ""

# 检查conda环境
if ! conda env list | grep -q "netkd"; then
    echo -e "${YELLOW}⚠️  警告: netkd环境不存在${NC}"
    exit 1
fi

echo -e "${GREEN}✅ 开始训练 (Ctrl+C 停止)${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 执行训练（实时输出到终端）
conda run -n netkd python train_with_real_data.py \
  --dataset "$DATASET" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --epochs_teacher "$EPOCHS_TEACHER" \
  --epochs_stacking "$EPOCHS_STACKING" \
  --epochs_student "$EPOCHS_STUDENT" \
  --mode full_pipeline \
  2>&1 | tee training_live.log

echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    训练完成!                             ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "📁 训练日志已保存到: training_live.log"
echo "📁 模型检查点保存在: checkpoints/"
