"""正确的评估脚本 - 使用训练时相同的预处理"""
import torch
import timm
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import json, csv
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

DATASET_PATH = '/workspace/yqm/Dataset/ISCXVPN2016'
NUM_CLASSES = 7
IMG_SIZE = 40  # 原始训练使用40x40

BEST_MODELS = {
    'repvit_m0_9_frac1.0': ('repvit_m0_9', 'results/kd_ablation_20251208_191348/repvit_m0_9_ce_only_frac1.0/best_model.pth'),
    'repvit_m0_9_frac0.5': ('repvit_m0_9', 'results/kd_ablation_20251208_191348/repvit_m0_9_ce_only_frac0.5/best_model.pth'),
    'repvit_m0_9_frac0.2': ('repvit_m0_9', 'results/kd_ablation_20251208_191348/repvit_m0_9_ce_only_frac0.2/best_model.pth'),
    'mobilenetv3_small_frac1.0': ('mobilenetv3_small_100', 'results/kd_ablation_20251208_191348/mobilenetv3_small_ce_only_frac1.0/best_model.pth'),
    'mobilenetv3_small_frac0.5': ('mobilenetv3_small_100', 'results/kd_ablation_20251208_191348/mobilenetv3_small_ce_only_frac0.5/best_model.pth'),
    'mobilenetv3_small_frac0.2': ('mobilenetv3_small_100', 'results/kd_ablation_20251208_191348/mobilenetv3_small_ce_only_frac0.2/best_model.pth'),
}

class TrafficDataset(Dataset):
    def __init__(self, root_dir, split='test'):
        self.root_dir = Path(root_dir) / split
        self.samples, self.labels = [], []
        
        self.class_names = sorted([d for d in os.listdir(self.root_dir) 
                                   if os.path.isdir(self.root_dir / d)])
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        
        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            if class_dir.is_dir():
                for img_path in class_dir.glob('*.png'):
                    self.samples.append(str(img_path))
                    self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img = Image.open(self.samples[idx])
        
        # 转灰度图
        if img.mode != 'L':
            img = img.convert('L')
        
        # 转numpy并归一化到[0,1]
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # 添加channel维度 [H,W] -> [1,H,W]
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)
        
        return img_tensor, self.labels[idx]

def load_model(arch_name, ckpt_path, device):
    """加载模型"""
    model = timm.create_model(arch_name, pretrained=False, num_classes=NUM_CLASSES, in_chans=1)
    
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt.get('model_state_dict', ckpt)
    
    # 移除model.前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('model.', '') if k.startswith('model.') else k
        new_state_dict[new_key] = v
    
    model.load_state_dict(new_state_dict, strict=True)
    return model.to(device)

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Evaluating', leave=False):
            outputs = model(images.to(device))
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels)
    
    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1_macro': f1_score(all_labels, all_preds, average='macro'),
    }

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Dataset: ISCXVPN2016 (7 classes, 40x40 grayscale)")
    
    dataset = TrafficDataset(DATASET_PATH, 'test')
    print(f"Test samples: {len(dataset)}")
    print(f"Classes: {dataset.class_names}\n")
    
    all_results = []
    
    for model_name, (arch, ckpt_path) in BEST_MODELS.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")
        
        if not os.path.exists(ckpt_path):
            print(f"  Model file not found")
            continue
        
        model = load_model(arch, ckpt_path, device)
        
        # 3轮测试
        for run_id in range(1, 4):
            loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
            metrics = evaluate(model, loader, device)
            
            result = {'model': model_name, 'run': run_id, **metrics}
            all_results.append(result)
            print(f"  Run {run_id}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f}")
    
    # 保存结果
    out_dir = Path('results/iscxvpn_correct_eval')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / 'results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    
    # 汇总
    print(f"\n{'='*60}")
    print("Results Summary (3 runs)")
    print(f"{'='*60}\n")
    
    from collections import defaultdict
    stats = defaultdict(list)
    for r in all_results:
        stats[r['model']].append((r['accuracy'], r['f1_macro']))
    
    print(f"| Model | Accuracy | F1-Macro |")
    print(f"|-------|----------|----------|")
    for model, runs in stats.items():
        accs = [r[0] for r in runs]
        f1s = [r[1] for r in runs]
        print(f"| {model} | {np.mean(accs):.4f}±{np.std(accs):.4f} | {np.mean(f1s):.4f}±{np.std(f1s):.4f} |")
    
    print(f"\nResults saved to: {out_dir}/")

if __name__ == '__main__':
    main()
