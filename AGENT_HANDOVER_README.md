
## 📦 仓库清理说明 (2025-12-08)

### 已清理内容
- ✅ 所有模型权重文件 (*.pth)
- ✅ 虚拟环境 (.venv/)
- ✅ Git 历史中的大文件
- ✅ 实验结果中的权重文件

### 保留内容
- ✅ 所有 Python 代码
- ✅ 模型架构定义
- ✅ 实验结果 JSON 文件
- ✅ 三个核心文档

### 仓库大小
- **清理前**: ~20GB
- **清理后**: ~598MB (.git) + 247MB (工作区) = **845MB**
- **推送成功**: ✅ 已上传到 GitHub Ubuntu 分支

### GitHub 仓库
- 🔗 URL: https://github.com/VenusRick/NetKD
- 🌿 分支: Ubuntu
- 🔑 SSH: git@github.com:VenusRick/NetKD.git


## 🚀 STEP 3 & 4: KD 消融和数据效率实验 (2025-12-08 进行中)

### 实验目标
根据论文新计划 (12.8-新计划.md)，完成以下两个关键实验：

**STEP 3 - KD 损失配置消融**:
- 目标: 验证 CE+KL (T=3) 是最佳 KD 配置
- 对比: CE-only vs CE+KL
- 学生模型: repvit_m0_9 (4.72M) 和 mobilenetv3_small (1.52M)

**STEP 4 - 数据效率实验**:
- 目标: 证明 KD 在有限数据下的优势
- 数据比例: 100%, 50%, 20%
- 配置: CE-only vs CE+KL

### 实验设置
- 总实验数: 2 models × 2 KD configs × 3 data fractions = **12 个实验**
- GPU 分配: 循环使用 GPU 0, 1, 2
- 训练设置: 100 epochs, batch_size=128, lr=0.001
- 预计耗时: 6-8 小时

### 运行状态
- **启动时间**: 2025-12-08 19:13
- **PID**: 246381
- **日志文件**: logs/step3_step4_experiments.log
- **结果目录**: results/kd_ablation_20251208_191348/

### 监控命令
```bash
# 查看实验进度
tail -f logs/step3_step4_experiments.log

# 查看 GPU 使用情况
nvidia-smi

# 检查进程状态
ps aux | grep 246381

# 查看已完成的实验结果
find results/kd_ablation_20251208_191348 -name "metrics.json"
```

### 生成的脚本
1. **scripts/run_kd_data_efficiency.py**  
   单个实验运行脚本，支持参数:
   - `--student`: 学生模型名称
   - `--kd_config`: ce_only 或 ce_kl
   - `--train_fraction`: 训练数据比例 (1.0, 0.5, 0.2)
   - `--gpu`: GPU ID

2. **scripts/run_step3_step4_experiments.sh**  
   批量实验编排脚本，自动运行所有 12 个实验组合

### 下一步 (完成后)
1. ✅ STEP 3 & 4: 运行 KD 消融和数据效率实验（进行中）
2. ⏳ STEP 1: 教师多样性和贡献分析
3. ⏳ STEP 2: 学生 Pareto 前沿分析
4. ⏳ 生成最终的汇总表格和文档

