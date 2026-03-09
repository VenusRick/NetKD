#!/bin/bash
# 并行启动实验 - 3块GPU分配不同任务

cd /workspace/yqm/NetKD
source .venv/bin/activate

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="results/full_experiment_v2/${TIMESTAMP}"
mkdir -p ${OUTPUT_BASE}

echo "Starting parallel experiments at ${TIMESTAMP}"
echo "Output: ${OUTPUT_BASE}"

# GPU 0: MAE预训练 - ConvNeXt系列
nohup python -c "
import sys
sys.path.insert(0, '.')
from scripts.run_full_pipeline import *

config = load_config('configs/full_experiment_v2.yaml')
output_dir = Path('${OUTPUT_BASE}')
output_dir.mkdir(parents=True, exist_ok=True)
log_file = output_dir / 'gpu0.log'

device = get_device(0)
train_loader, _, _ = get_data_loaders(config['experiment']['dataset_root'], batch_size=512)

mae_dir = output_dir / 'phase1_pretrain' / 'mae'
mae_dir.mkdir(parents=True, exist_ok=True)

results = []
for backbone in ['convnextv2_tiny', 'convnextv2_small', 'convnextv2_nano']:
    for mask_ratio in [0.6, 0.7, 0.8]:
        try:
            r = train_mae(backbone, mask_ratio, 100, train_loader, device, mae_dir, log_file)
            results.append(r)
            save_json(results, output_dir / 'mae_gpu0.json')
        except Exception as e:
            log_message(f'Failed {backbone}: {e}', log_file)

print('GPU0 MAE done')
" > ${OUTPUT_BASE}/gpu0_stdout.log 2>&1 &

echo "GPU 0 started (ConvNeXt MAE)"

# GPU 1: MAE预训练 - EfficientNet系列
nohup python -c "
import sys
sys.path.insert(0, '.')
from scripts.run_full_pipeline import *

config = load_config('configs/full_experiment_v2.yaml')
output_dir = Path('${OUTPUT_BASE}')
output_dir.mkdir(parents=True, exist_ok=True)
log_file = output_dir / 'gpu1.log'

device = get_device(1)
train_loader, _, _ = get_data_loaders(config['experiment']['dataset_root'], batch_size=512)

mae_dir = output_dir / 'phase1_pretrain' / 'mae'
mae_dir.mkdir(parents=True, exist_ok=True)

results = []
for backbone in ['efficientnetv2_rw_s', 'efficientnetv2_rw_m']:
    for mask_ratio in [0.6, 0.7, 0.8]:
        try:
            r = train_mae(backbone, mask_ratio, 100, train_loader, device, mae_dir, log_file)
            results.append(r)
            save_json(results, output_dir / 'mae_gpu1.json')
        except Exception as e:
            log_message(f'Failed {backbone}: {e}', log_file)

print('GPU1 MAE done')
" > ${OUTPUT_BASE}/gpu1_stdout.log 2>&1 &

echo "GPU 1 started (EfficientNet MAE)"

# GPU 2: MAE预训练 - GhostNet/RepViT系列
nohup python -c "
import sys
sys.path.insert(0, '.')
from scripts.run_full_pipeline import *

config = load_config('configs/full_experiment_v2.yaml')
output_dir = Path('${OUTPUT_BASE}')
output_dir.mkdir(parents=True, exist_ok=True)
log_file = output_dir / 'gpu2.log'

device = get_device(2)
train_loader, _, _ = get_data_loaders(config['experiment']['dataset_root'], batch_size=512)

mae_dir = output_dir / 'phase1_pretrain' / 'mae'
mae_dir.mkdir(parents=True, exist_ok=True)

results = []
for backbone in ['ghostnetv2_100', 'ghostnetv3_100', 'repvit_m1_0']:
    for mask_ratio in [0.6, 0.7, 0.8]:
        try:
            r = train_mae(backbone, mask_ratio, 100, train_loader, device, mae_dir, log_file)
            results.append(r)
            save_json(results, output_dir / 'mae_gpu2.json')
        except Exception as e:
            log_message(f'Failed {backbone}: {e}', log_file)

print('GPU2 MAE done')
" > ${OUTPUT_BASE}/gpu2_stdout.log 2>&1 &

echo "GPU 2 started (GhostNet/RepViT MAE)"

echo ""
echo "All experiments launched!"
echo "Monitor with: tail -f ${OUTPUT_BASE}/gpu*.log"
echo "Check GPU: nvidia-smi -l 5"

# 保存输出目录
echo "${OUTPUT_BASE}" > /tmp/current_experiment_dir.txt
