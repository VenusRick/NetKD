# NetKD 文档索引

**最后更新**: 2025-12-11

---

## 📚 文档导航

### 🚀 快速开始
- **[Agent交接指南](01_AGENT_HANDOVER_GUIDE.md)** - 项目交接、快速上手
  - SSH连接信息
  - 常用命令
  - GPU监控方法
  - 数据集路径

### 📊 实验相关
- **[实验结果记录](02_EXPERIMENT_RESULTS.md)** - 当前实验进度和结果
  - 最佳模型性能
  - 各数据集进度
  - 消融实验结果

- **[完整实验计划](COMPLETE_EXPERIMENT_PLAN.md)** - Phase 1-4完整方案
  - 自监督预训练（MAE/SimCLR）
  - 教师模型微调
  - Stacking集成
  - 知识蒸馏

- **[TODO: Teacher2.0 & Student2.0](TODO_Teacher2.0_Student2.0.md)** - 模型升级计划
  - Teacher 2.0任务列表
  - Student 2.0任务列表
  - 实验配置

### 🏗️ 模型架构
- **[模型架构文档](03_MODEL_ARCHITECTURE.md)** - 详细的模型设计
  - 教师模型架构
  - 学生模型架构
  - 注意力机制
  - Stacking集成设计

- **[教师集成方案](Teacher_Ensemble_Section_Revised.md)** - Stacking技术说明

### 📈 历史记录
- **[实验历史](EXPERIMENT_HISTORY.md)** - 过往实验记录和经验

---

## 🗂️ 其他重要文件

### 项目根目录
- `README.md` - 项目主文档
- `WORK_SUMMARY_20251211.md` - 最新工作总结
- `CURRENT_STATUS_SUMMARY.txt` - 当前状态

### 配置文件
- `configs/teachers.yaml` - 教师模型配置
- `configs/students.yaml` - 学生模型配置
- `configs/full_experiment_v2.yaml` - 完整实验配置

### 实验结果
- `results/complete_experiment/FINAL_REPORT.md` - **最新实验报告**
- `results/complete_experiment/summary_statistics.csv` - 统计数据

### 分析脚本
- `analysis/compute_disagreement.py` - 教师多样性分析
- `scripts/leave_one_out_stacking.py` - Leave-One-Out分析
- `scripts/monitor_and_summarize.py` - 进度监控
- `scripts/generate_final_report.py` - 报告生成

---

## 📦 归档文档

旧版本和重复文档已移至 `archive/` 目录：
- `archive/AGENT_HANDOFF*.md` - 旧版交接文档
- `archive/Experiment_Results_Summary.md` - 旧版实验结果
- `archive/MODEL_ARCHITECTURE.md` - 旧版模型文档
- `archive/EXECUTION_PLAN.md` - 已过时的执行计划

---

## 🔍 文档更新日志

### 2025-12-11
- ✅ 创建文档索引
- ✅ 归档6个重复文档
- ✅ 新增工作总结 `WORK_SUMMARY_20251211.md`
- ✅ 新增分析工具文档

### 2025-12-10
- 创建 `01_AGENT_HANDOVER_GUIDE.md`
- 创建 `02_EXPERIMENT_RESULTS.md`
- 创建 `03_MODEL_ARCHITECTURE.md`

---

*建议：优先阅读 `01_AGENT_HANDOVER_GUIDE.md` 和 `results/complete_experiment/FINAL_REPORT.md`*
