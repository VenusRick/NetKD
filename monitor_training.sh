#!/bin/bash
echo "==================================================================="
echo "NetKD 训练监控"
echo "==================================================================="
echo ""
echo "训练进程状态:"
ps aux | grep "train_with_real_data.py" | grep -v grep | head -1
echo ""
echo "GPU使用情况:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "GPU信息不可用"
echo ""
echo "检查点文件 (最新5个):"
ls -lht checkpoints/*.pth 2>/dev/null | head -5
echo ""
echo "训练日志最后20行:"
tail -20 nohup.out 2>/dev/null || tail -20 training_full_log.txt 2>/dev/null || echo "日志文件暂无内容"
