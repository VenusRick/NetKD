# NetKD AI Agent Instructions

## Project Overview
Three-stage knowledge distillation for encrypted network traffic classification:  
**Teachers** (ResNet50/DenseNet121/MobileNetV3 + ECA) → **Stacking** (MLP fusion) → **Student** (ShuffleNetV2 + AgentAttention)

**Critical**: All inputs are **single-channel grayscale** `[B, 1, H, W]`. Conv1 adapted via `weight.mean(dim=1, keepdim=True)`.

## Architecture & Pipeline
| Stage | Entry Point | Models | Checkpoint |
|-------|-------------|--------|------------|
| I Teachers | `python training/train.py --mode train_teachers` | `models/teacher_models.py` | `checkpoints/{resnet50,densenet121,mbv3}_teacher.pth` |
| II Stacking | `--mode train_stacking` | `StackingModel` (MLP) | `checkpoints/stacking_model.pth` |
| III Student | `--mode train_student` | `models/student_model.py` | `checkpoints/student_sd_mkd.pth` |

**Data Flow**:
```
Raw PCAP → data_preprocessing.preprocess_data → 32×32 grayscale PNG
         → quick_load_dataset() → DataLoader → Training
```

## Data Loading Pattern
```python
from data_preprocessing.image_loader import quick_load_dataset
train_loader, val_loader, test_loader, meta = quick_load_dataset(
    dataset_name='ISCXVPN2016',
    dataset_root='/walnut_data/yqm/Dataset',
    batch_size=256,
    val_ratio=0.15, test_ratio=0.15
)
num_classes = meta['num_classes']
```
**Structure**: `Dataset/ISCXVPN2016/{train,valid,test}/class_name/*.png`

## Validated Hyperparameters ⚠️
| Model | lr | batch_size | epochs | Notes |
|-------|-----|------------|--------|-------|
| DenseNet/ResNet | 1e-3 | 512 | 25 | Standard |
| MobileNetV3-ECA | **5e-4** | 256 | 50 | `label_smoothing=0.1`, `patience=3` |
| Agent Attention | **1e-4** | 128 | 50+ | `warmup=10`, `grad_clip=1.0` |
| Student (best) | 1e-3 | **128** | 100 | 98.34% vs 256→97.61% |

**Rule**: Smaller networks → lower lr + stronger regularization

## Loss Function (`training/loss_functions.py`)
```python
# Composite: CE + Forward KL + Reverse KL + Sinkhorn
loss = lamb_ce * L_ce + lamb_f * L_fkl + lamb_r * L_rkl + lamb_s * L_sinkhorn
# Default weights: lamb_ce=1.0, lamb_f=0.5, lamb_r=0.5, lamb_s=0.1
# Stability: clamp eps=1e-8 for log operations
```
**Distillation modes**: `ce` (GT only) | `kl` (GT+FKL) | `kl2` (GT+FKL+RKL) | `full` (all)

## Quick Commands
```bash
# Full pipeline with ECA
python training/train.py --use_real_data --mode train_student \
  --dataset ISCXVPN2016 --dataset_root /walnut_data/yqm/Dataset \
  --batch_size 128 --epochs_student 100 --resnet_use_eca --mbv3_use_eca

# Ablation experiments
python run_ablation_student.py kl              # Single distill mode
bash scripts/run_ablation_experiments.sh       # All modes

# Attention ablation
python run_attention_ablation_fixed.py agent   # agent/cbam/eca/simam/none

# Monitoring
tensorboard --logdir runs/ --port 6006
```

## Common Issues & Fixes
| Symptom | Cause | Solution |
|---------|-------|----------|
| Val acc collapses epoch 5-10 | lr too high | Reduce lr 50%, add `label_smoothing=0.1` |
| Loss → NaN (Agent Attention) | Gradient explosion | `lr=1e-4`, `warmup=10`, `grad_clip=1.0` |
| FileNotFoundError on data | Wrong path structure | Use `{split}/class_name/*.png` |
| MobileNetV3 unstable | Sensitive architecture | `lr=5e-4`, `patience=3`, larger batch |

## Performance Baselines (ISCXVPN2016)
- **Best Teacher**: DenseNet121-ECA @ **98.77%**
- **Student S-KL**: 97.11% test (**98.39%** knowledge retention)
- **Agent Attention**: **98.55%** test (+0.72% gain over baseline)
- **Compression**: >97% parameter reduction

## Key Files Reference
| File | Purpose |
|------|---------|
| `training/train.py` | Main entry: `train_teachers`/`train_stacking`/`train_student` |
| `training/loss_functions.py` | CE+FKL+RKL+Sinkhorn composite loss |
| `models/teacher_models.py` | Teacher architectures + ECA injection |
| `models/student_model.py` | ShuffleNetV2 + AgentAttention student |
| `models/student_model_flexible.py` | Pluggable attention for ablation |
| `models/eca_module.py` | ECA with adaptive kernel `k = t if t%2 else t+1` |
| `data_preprocessing/image_loader.py` | `quick_load_dataset()` for data loading |

## Code Conventions
- Teacher checkpoints: `{model_name}_teacher.pth`
- Ablation outputs: `checkpoints/ablation/s_{mode}/` or `checkpoints/attention_ablation/`
- Logs: `logs/` (text), `runs/` (TensorBoard)
- Use `LiveTrainingMonitor` from `training.monitor` for progress tracking
- Experiment reports: `checkpoints/eca_pipeline/EXPERIMENT_REPORT_UPDATED.md`
