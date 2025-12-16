# NetKD 文档索引

**最后更新**: 2025-12-16

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

## 🗂️ 项目关键文件

### 配置文件
| 文件 | 说明 |
|------|------|
| `configs/teachers.yaml` | 教师模型配置 |
| `configs/students.yaml` | 学生模型配置 |
| `configs/full_experiment.yaml` | 完整实验配置 |

### 核心脚本
| 文件 | 说明 |
|------|------|
| `scripts/run_complete_experiment.py` | 完整实验入口 |
| `scripts/monitor_and_summarize.py` | 进度监控 |
| `scripts/generate_final_report.py` | 报告生成 |
| `analysis/compute_disagreement.py` | 教师多样性分析 |
| `scripts/leave_one_out_stacking.py` | Leave-One-Out分析 |

### 实验结果
| 文件 | 说明 |
|------|------|
| `results/complete_experiment/FINAL_REPORT.md` | 最新实验报告 |
| `results/complete_experiment/summary_statistics.csv` | 统计数据 |

---

## 📦 归档文档

旧版本和重复文档已移至 `archive/` 目录：
- 多个版本的Agent交接文档
- 旧版实验结果文档
- 过时的执行计划
- 历史维护记录

---

## 🔍 更新日志

### 2025-12-16
- ✅ 项目结构大整理
- ✅ 归档冗余文档和脚本
- ✅ 清理备份文件
- ✅ 更新README.md
- ✅ 更新文档索引

### 2025-12-11
- 创建文档索引
- 归档6个重复文档
- 新增工作总结

### 2025-12-10
- 创建 01_AGENT_HANDOVER_GUIDE.md
- 创建 02_EXPERIMENT_RESULTS.md
- 创建 03_MODEL_ARCHITECTURE.md

---

*建议：优先阅读 [01_AGENT_HANDOVER_GUIDE.md](01_AGENT_HANDOVER_GUIDE.md) 和 [results/complete_experiment/FINAL_REPORT.md](../results/complete_experiment/FINAL_REPORT.md)*
