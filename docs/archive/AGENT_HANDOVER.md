# Agent Handover Document

**Last Updated:** $(date '+%Y-%m-%d %H:%M:%S')  
**Server:** root@10.126.126.3:32833  
**Project:** NetKD - Network Traffic Classification with Knowledge Distillation  
**Repository:** https://github.com/VenusRick/NetKD (Ubuntu branch)

---

## 🎯 Current Status

### Running Processes
- **No active training processes currently**
- All 3x RTX 4090 GPUs available

### Latest Task
Preparing to run complete training pipeline for all 6 datasets:
1. CICIoT2022
2. USTC-TFC2016  
3. ISCXTor2016
4. ISCXVPN2016
5. Bing-Traffic
6. Malware-Traffic

### Key Finding
- **Cross-dataset generalization is POOR (5.2% accuracy)**
- **Each dataset requires separate training**
- Previous experiments focused on ISCXVPN2016 only

---

## 📁 Project Structure

\`\`\`
NetKD/
├── Dataset/                   # All 6 grayscale datasets  
├── training/
│   ├── train.py              # Main 3-stage training pipeline
│   ├── engine.py             # Training engine
│   └── evaluation.py         # Model evaluation
├── models/                   # Model architectures
├── experiments/              # Experiment scripts
├── results/                  # All experiment results
├── run_all_datasets_training.sh  # NEW: Full dataset training script
└── Core Docs:
    ├── AGENT_HANDOVER.md     # This file
    ├── EXPERIMENT_LOG.md     # Results log
    └── MODEL_ARCHITECTURE.md # Model specs

\`\`\`

---

## 🚀 Quick Start (Next Agent)

###1. Connect to Server
\`\`\`bash
ssh -p 32833 root@10.126.126.3
Password: Liuliang_666
cd /workspace/yqm/NetKD
\`\`\`

### 2. Check GPU Status
\`\`\`bash
nvidia-smi
ps aux | grep python
\`\`\`

### 3. Run Training for All Datasets
\`\`\`bash
# Start the complete training pipeline
nohup bash run_all_datasets_training.sh > pipeline.log 2>&1 &

# Monitor progress (every 5 min)
tail -f pipeline.log
tail -f logs/*_training_*.log
\`\`\`

### 4. Monitor Training
\`\`\`bash
# Check GPU utilization
watch -n 60 nvidia-smi

# Check training logs
ls -lht logs/
tail -100 logs/ISCXVPN2016_training_*.log
\`\`\`

### 5. After Training Completes
\`\`\`bash
# Check results
ls -lh results/full_dataset_runs_*/
cat results/full_dataset_runs_*/training_summary.json

# Update EXPERIMENT_LOG.md with new results
# Push to GitHub
git add -A
git commit -m "Complete dataset training results"
git push origin Ubuntu
\`\`\`

---

## 🏗️ Training Architecture

### Three-Stage Pipeline (training/train.py)

**Stage 1: Teacher Pretraining**
- Models: DenseNet121, MobileNetV3Large, ResNet50
- Pretrained: ImageNet weights  
- Epochs: 30 (reduced from 50)
- Purpose: Learn dataset-specific features

**Stage 2: Stacking Ensemble**
- Combines 3 teacher outputs
- Meta-learner: Logistic Regression
- Epochs: 20
- Purpose: Optimal teacher combination

**Stage 3: Student Distillation**
- Student: StudentNet (lightweight CNN)
- Loss: CE + Forward KL + Reverse KL + Sinkhorn
- Epochs: 50
- Temperature: 4.0
- Alpha: 0.3

### Hardware Utilization
- **3x RTX 4090 GPUs (24GB each)**
- Batch Size: 128
- Parallel training across datasets

---

## 📊 Best Results So Far

### ISCXVPN2016 Dataset
| Model | Accuracy | F1-Score | Params | Notes |
|-------|----------|----------|--------|-------|
| RepVit-M0.9 | 98.05% | - | ~5M | Best student |
| MobileNetV3 | 97.40% | - | ~5M | Good student |
| EfficientNet-B0 | - | - | ~5M | Good student |

### Other Datasets
- **Status:** Not yet trained with KD pipeline
- **Plan:** Run full 3-stage training on all 6 datasets

---

## 🔧 Common Commands

### Kill Zombie Processes
\`\`\`bash
ps aux | grep python | grep -v grep
# Identify PID, then:
kill -9 <PID>
\`\`\`

### Clean Up Old Results
\`\`\`bash
# Be careful - only remove truly old/redundant results
rm -rf results/obsolete_experiment_folder/
\`\`\`

### GitHub Push
\`\`\`bash
git status
git add results/ AGENT_HANDOVER.md EXPERIMENT_LOG.md
git commit -m "Update: [brief description]"
git push origin Ubuntu
# Use token: [REDACTED]
\`\`\`

---

## ⚠️ Important Notes

1. **All datasets are GRAYSCALE (1-channel)**, not RGB
2. **Cross-dataset performance is poor** - train separately for each dataset
3. **Check convergence**: Training typically converges in 20-30 epochs for teacher, 30-40 for student
4. **Monitor GPU memory**: If OOM, reduce batch size to 64
5. **Update docs**: After each significant result, update EXPERIMENT_LOG.md

---

## 📝 TODO List

- [x] Create training pipeline for all datasets
- [ ] Run training on all 6 datasets
- [ ] Collect and analyze results
- [ ] Compare student vs teacher performance
- [ ] Update EXPERIMENT_LOG.md with comprehensive results
- [ ] Push final results to GitHub
- [ ] Write paper analysis based on results

---

## 🆘 Troubleshooting

### Training Fails
- Check dataset paths in `data_preprocessing.py`
- Verify GPU memory: `nvidia-smi`
- Check logs in `logs/` directory

### Low Accuracy (<80%)
- Verify dataset is loaded correctly
- Check class balance
- Increase training epochs
- Try different hyperparameters

### Git Push Fails
- Use personal access token (see GitHub Push section)
- Check network connection
- Verify remote URL: `git remote -v`

---

**Remember:** Always update this document and EXPERIMENT_LOG.md after completing tasks!
