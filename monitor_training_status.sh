#!/bin/bash
# 实时监控训练状态（单次查询）

# 彩色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║            NetKD 训练状态监控                            ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

# 1. 检查训练进程
echo -e "${GREEN}【1】训练进程状态:${NC}"
TRAIN_PID=$(pgrep -f "train_with_real_data.py")
if [ -n "$TRAIN_PID" ]; then
    echo -e "  ${GREEN}✅ 训练正在运行${NC} (PID: $TRAIN_PID)"
    ps -p $TRAIN_PID -o pid,etime,%cpu,%mem,cmd --no-headers | \
        awk '{printf "  ├─ 运行时间: %s\n  ├─ CPU: %s%%\n  ├─ 内存: %s%%\n  └─ 命令: %s\n", $2, $3, $4, $5}'
    
    # 显示worker进程数
    WORKER_COUNT=$(pgrep -P $TRAIN_PID | wc -l)
    echo "  📦 数据加载Workers: $WORKER_COUNT 个"
else
    echo -e "  ${RED}❌ 训练未运行${NC}"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 2. GPU状态
echo ""
echo -e "${GREEN}【2】GPU使用情况:${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
        --format=csv,noheader,nounits | \
    awk -F', ' '{
        printf "  GPU %s: %s\n", $1, $2
        printf "    ├─ 利用率: %s%%\n", $3
        printf "    ├─ 显存: %sMB / %sMB (%.1f%%)\n", $4, $5, ($4/$5)*100
        printf "    ├─ 温度: %s°C\n", $6
        printf "    └─ 功耗: %sW\n\n", $7
    }'
else
    echo "  ⚠️  nvidia-smi 不可用"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 3. 最新日志
echo ""
echo -e "${GREEN}【3】训练日志 (最后15行):${NC}"
LOG_FILE=""
if [ -f "training_live.log" ]; then
    LOG_FILE="training_live.log"
elif [ -f "nohup.out" ]; then
    LOG_FILE="nohup.out"
fi

if [ -n "$LOG_FILE" ]; then
    echo "  📄 来自: $LOG_FILE"
    echo "  ─────────────────────────────────────────────────────"
    tail -n 15 "$LOG_FILE" | sed 's/^/  │ /'
else
    echo "  ⚠️  未找到日志文件"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 4. 检查点文件
echo ""
echo -e "${GREEN}【4】训练检查点:${NC}"
if [ -d "checkpoints" ]; then
    CKPT_COUNT=$(ls checkpoints/*.pth 2>/dev/null | wc -l)
    if [ $CKPT_COUNT -gt 0 ]; then
        echo "  💾 共 $CKPT_COUNT 个检查点文件"
        ls -lht checkpoints/*.pth 2>/dev/null | head -5 | \
            awk '{printf "  │ %-30s %5s %s %s %s\n", $9, $5, $6, $7, $8}'
    else
        echo "  ⚠️  尚无检查点文件"
    fi
else
    echo "  ⚠️  checkpoints目录不存在"
fi

echo ""
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}💡 实时监控命令:${NC}"
echo "   watch -n 2 bash monitor_training_status.sh  # 每2秒刷新"
echo "   tail -f training_live.log                   # 实时查看日志"
echo "   watch -n 1 nvidia-smi                       # GPU监控"
