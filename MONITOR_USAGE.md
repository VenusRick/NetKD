# NetKD 训练监控使用指南

## 🚀 快速开始

### 启动实时监控
```bash
cd /walnut_data/yqm/NetKD
bash monitor_live.sh
```

**功能说明:**
- 每2秒自动刷新训练状态
- 显示进程运行情况、GPU使用率、最新日志、检查点文件
- 按 `Ctrl+C` 退出监控

## 📊 监控界面说明

### 【1】训练进程
- ✓ 运行中: 显示进程ID、运行时间、CPU/内存占用
- ✗ 未运行: 提示训练未启动
- 显示数据加载Workers数量

### 【2】GPU 状态
- GPU利用率 (颜色编码):
  - 🟢 绿色: ≥70% (高效)
  - 🟡 黄色: 30-70% (中等)
  - 🔴 红色: <30% (低效)
- 显存使用: 当前/总量 (百分比 + 进度条)
- 温度和功耗

### 【3】训练进度
- 实时显示最新10行训练日志
- 包含epoch、loss、accuracy等关键信息

### 【4】检查点文件
- 列出最新3个.pth检查点文件
- 显示文件大小和更新时间

## 🛠️ 其他监控方式

### 1. 仅查看训练日志
```bash
# 实时跟踪日志
tail -f training_high_gpu_final.log

# 查看最后50行
tail -n 50 training_high_gpu_final.log

# 搜索关键信息
grep "Epoch" training_high_gpu_final.log
grep "准确率" training_high_gpu_final.log
```

### 2. 仅监控GPU
```bash
# 每秒刷新
watch -n 1 nvidia-smi

# 简洁显示
watch -n 1 'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv'
```

### 3. 检查训练进程
```bash
# 查找进程
ps aux | grep train_with_real_data

# 查看进程树
pstree -p $(pgrep -f train_with_real_data)

# 实时资源监控
top -p $(pgrep -f train_with_real_data)
```

### 4. 查看检查点
```bash
# 列出所有检查点
ls -lh checkpoints/*.pth

# 按时间排序
ls -lt checkpoints/*.pth

# 查看最新更新
ls -t checkpoints/*.pth | head -3
```

## 📝 训练进度关键词

在日志中搜索这些关键词了解训练进度:

```bash
# 查看epoch进度
grep "epoch=" training_high_gpu_final.log | tail -20

# 教师模型训练
grep "Teacher" training_high_gpu_final.log

# Stacking模型训练
grep "Stacking" training_high_gpu_final.log

# 学生模型训练
grep "Student" training_high_gpu_final.log

# 准确率变化
grep "acc=" training_high_gpu_final.log

# 损失值
grep "loss=" training_high_gpu_final.log
```

## ⚡ 性能指标解读

### GPU利用率
- **目标**: 60-90%
- **当前配置**: batch_size=512
- **低于30%**: 考虑增大batch_size或检查数据加载

### 显存使用
- **总显存**: 24GB (RTX 4090)
- **当前配置**: 约4-6GB (16-25%)
- **优化目标**: 15-20GB (60-80%)
- **如需提升**: 增大batch_size到768或1024

### 训练速度
- **每个epoch时间**: 取决于GPU利用率
- **优化后预期**: 比原配置快2-3倍
- **总训练时间**: 约1-2小时 (15+12+25=52 epochs)

## 🚨 常见问题

### Q: 监控脚本显示"未运行"
A: 训练可能未启动或已完成,检查:
```bash
ps aux | grep train
ls -lht checkpoints/*.pth | head -3
```

### Q: GPU利用率很低
A: 可能原因:
1. batch_size太小 → 增大到768或1024
2. 数据加载慢 → 增加num_workers
3. 正在进行评估阶段 → 等待下一个训练epoch

### Q: 找不到日志文件
A: 检查所有可能的日志文件:
```bash
ls -lh training*.log nohup.out
```

### Q: 如何停止训练
A: 找到进程ID并终止:
```bash
pkill -f train_with_real_data
# 或
kill $(pgrep -f train_with_real_data)
```

## 💡 高级技巧

### 多终端监控
同时打开3个终端:
- 终端1: `bash monitor_live.sh` (综合监控)
- 终端2: `tail -f training_high_gpu_final.log` (详细日志)
- 终端3: `watch -n 1 nvidia-smi` (GPU监控)

### 后台运行保持会话
```bash
# 使用tmux
tmux new -s training
bash monitor_live.sh
# 按 Ctrl+B 然后 D 分离
# tmux attach -t training 重新连接

# 或使用screen
screen -S training
bash monitor_live.sh
# 按 Ctrl+A 然后 D 分离
# screen -r training 重新连接
```

### 定期保存状态快照
```bash
# 每30分钟保存一次状态
while true; do
    date >> training_status.log
    nvidia-smi >> training_status.log
    tail -5 training_high_gpu_final.log >> training_status.log
    echo "---" >> training_status.log
    sleep 1800
done &
```

## 📞 需要帮助?

如有问题,检查:
1. `README.md` - 项目说明
2. `TRAINING_GUIDE.md` - 训练指南
3. `MIGRATION_SUMMARY.md` - 系统迁移说明
