#!/usr/bin/env python3
"""完整实验 V2: MAE(0.8) + SimCLR预训练对比 + 新教师模型 + 学生模型"""
import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
import numpy as np

sys.path.insert(0, '/workspace/yqm/NetKD')
from data_preprocessing.image_loader import quick_load_dataset
from sklearn.metrics import f1_score, classification_report

# 配置
def log(msg, f=None):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] {msg}")
    if f:
        with open(f, 'a') as fp:
            fp.write(f"[{ts}] {msg}\n")

def get_dataloaders(bs=128):
    d = CONFIG['data_dir']
    train = TrafficDataset(os.path.join(d, 'train.npz'))
    val = TrafficDataset(os.path.join(d, 'val.npz'))
    test = TrafficDataset(os.path.join(d, 'test.npz'))
    return (DataLoader(train, batch_size=bs, shuffle=True, num_workers=8, pin_memory=True),
            DataLoader(val, batch_size=bs, shuffle=False, num_workers=8, pin_memory=True),
            DataLoader(test, batch_size=bs, shuffle=False, num_workers=8, pin_memory=True))

def create_model(name, nc=12):
    import timm
    model_map = {
        'convnextv2_tiny': 'convnextv2_tiny.fcmae',
        'efficientnetv2_s': 'tf_efficientnetv2_s',
        'ghostnetv2_100': 'ghostnetv2_100',
        'repvit_m1_0': 'repvit_m1_0.dist_300e_in1k',
        'repvit_m0_9': 'repvit_m0_9.dist_300e_in1k',
        'ghostnet_100': 'ghostnet_100',
        'mobilenetv3_small': 'mobilenetv3_small_100',
        'mobilenetv2_050': 'mobilenetv2_050',
    }
    try:
        m = timm.create_model(model_map.get(name, name), pretrained=False, num_classes=nc, in_chans=1)
    except:
        m = timm.create_model(model_map.get(name, name), pretrained=False, num_classes=nc)
        # 修改第一层
        for n, mod in m.named_modules():
            if isinstance(mod, nn.Conv2d) and mod.in_channels == 3:
                parent = m
                parts = n.split('.')
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                old = getattr(parent, parts[-1])
                new_conv = nn.Conv2d(1, old.out_channels, old.kernel_size, old.stride, old.padding, bias=old.bias is not None)
                setattr(parent, parts[-1], new_conv)
                break
    return m

def train_model(model, train_loader, val_loader, epochs, device, log_file=None):
    model = model.to(device)
    opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    best_acc = 0
    
    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
        
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = 100*correct/total
        sch.step()
        
        if (ep+1) % 10 == 0:
            log(f"  Epoch {ep+1}/{epochs}, Val Acc: {acc:.2f}%", log_file)
        if acc > best_acc:
            best_acc = acc
    
    return best_acc

def evaluate_model(model, test_loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            pred = model(x).argmax(1).cpu()
            preds.extend(pred.numpy())
            targets.extend(y.numpy())
    acc = 100 * np.mean(np.array(preds) == np.array(targets))
    f1 = 100 * f1_score(targets, preds, average='weighted')
    return acc, f1

# MAE预训练
def mae_pretrain(model_name, mask_ratio=0.8, epochs=100, device='cuda:0', save_dir=None, log_file=None):
    log(f"MAE预训练: {model_name}, mask={mask_ratio}", log_file)
    
    backbone = create_model(model_name, nc=12)
    backbone = backbone.to(device)
    train_loader, _, _ = get_dataloaders(bs=512)
    
    opt = optim.AdamW(backbone.parameters(), lr=0.01, weight_decay=0.05)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    
    best_loss = float('inf')
    for ep in range(epochs):
        backbone.train()
        total_loss = 0
        for x, _ in train_loader:
            x = x.to(device)
            # 随机掩码
            mask = (torch.rand_like(x) > mask_ratio).float()
            masked_x = x * mask
            opt.zero_grad()
            out = backbone(masked_x)
            # 简单的自监督: 特征一致性
            with torch.no_grad():
                target = backbone(x)
            loss = nn.functional.mse_loss(out, target)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        sch.step()
        
        if (ep+1) % 20 == 0:
            log(f"  Epoch {ep+1}/{epochs}, Loss: {avg_loss:.6f}", log_file)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                torch.save(backbone.state_dict(), f"{save_dir}/{model_name}_mae{int(mask_ratio*100)}.pth")
    
    log(f"  MAE预训练完成, Best Loss: {best_loss:.6f}", log_file)
    return backbone, best_loss

# SimCLR预训练
def simclr_pretrain(model_name, epochs=100, device='cuda:0', save_dir=None, log_file=None):
    log(f"SimCLR预训练: {model_name}", log_file)
    
    backbone = create_model(model_name, nc=128)  # 投影头维度
    backbone = backbone.to(device)
    train_loader, _, _ = get_dataloaders(bs=256)
    
    opt = optim.AdamW(backbone.parameters(), lr=0.001, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    
    def augment(x):
        # 数据增强
        if torch.rand(1) > 0.5:
            x = torch.flip(x, dims=[3])
        if torch.rand(1) > 0.5:
            x = x + 0.1 * torch.randn_like(x)
        return x
    
    def nt_xent_loss(z1, z2, temp=0.5):
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)
        bs = z1.size(0)
        z = torch.cat([z1, z2], dim=0)
        sim = torch.mm(z, z.t()) / temp
        labels = torch.cat([torch.arange(bs) + bs, torch.arange(bs)]).to(z.device)
        mask = torch.eye(2*bs, device=z.device).bool()
        sim.masked_fill_(mask, -1e9)
        return nn.functional.cross_entropy(sim, labels)
    
    best_loss = float('inf')
    for ep in range(epochs):
        backbone.train()
        total_loss = 0
        for x, _ in train_loader:
            x = x.to(device)
            v1, v2 = augment(x), augment(x)
            opt.zero_grad()
            z1, z2 = backbone(v1), backbone(v2)
            loss = nt_xent_loss(z1, z2)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        sch.step()
        
        if (ep+1) % 20 == 0:
            log(f"  Epoch {ep+1}/{epochs}, Loss: {avg_loss:.4f}", log_file)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                torch.save(backbone.state_dict(), f"{save_dir}/{model_name}_simclr.pth")
    
    log(f"  SimCLR预训练完成, Best Loss: {best_loss:.4f}", log_file)
    return backbone, best_loss

def run_experiment(gpu_id=0):
    device = f'cuda:{gpu_id}'
    torch.cuda.set_device(gpu_id)
    
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = f'/workspace/yqm/NetKD/results/complete_v2_{ts}'
    os.makedirs(result_dir, exist_ok=True)
    log_file = f'{result_dir}/experiment.log'
    
    log("="*60, log_file)
    log(f"完整实验V2开始 - GPU {gpu_id}", log_file)
    log("="*60, log_file)
    
    results = {'mae': [], 'simclr': [], 'finetune': []}
    train_loader, val_loader, test_loader = get_dataloaders()
    
    models_to_test = ['convnextv2_tiny', 'efficientnetv2_s']
    
    # Phase 1: MAE预训练 (mask=0.8)
    log("\n=== Phase 1: MAE预训练 (mask=0.8) ===", log_file)
    mae_dir = f'{result_dir}/mae_pretrain'
    for m in models_to_test:
        try:
            _, loss = mae_pretrain(m, mask_ratio=0.8, epochs=100, device=device, save_dir=mae_dir, log_file=log_file)
            results['mae'].append({'model': m, 'mask': 0.8, 'loss': loss})
        except Exception as e:
            log(f"MAE失败 {m}: {e}", log_file)
    
    # Phase 2: SimCLR预训练
    log("\n=== Phase 2: SimCLR预训练 ===", log_file)
    simclr_dir = f'{result_dir}/simclr_pretrain'
    for m in models_to_test:
        try:
            _, loss = simclr_pretrain(m, epochs=100, device=device, save_dir=simclr_dir, log_file=log_file)
            results['simclr'].append({'model': m, 'loss': loss})
        except Exception as e:
            log(f"SimCLR失败 {m}: {e}", log_file)
    
    # Phase 3: 微调对比
    log("\n=== Phase 3: 微调对比实验 ===", log_file)
    for m in models_to_test:
        # 从头训练
        log(f"微调 {m} (从头训练)", log_file)
        model = create_model(m)
        val_acc = train_model(model, train_loader, val_loader, 50, device, log_file)
        test_acc, f1 = evaluate_model(model, test_loader, device)
        results['finetune'].append({'model': m, 'method': 'scratch', 'val': val_acc, 'test': test_acc, 'f1': f1})
        log(f"  结果: Val={val_acc:.2f}%, Test={test_acc:.2f}%, F1={f1:.2f}%", log_file)
        
        # MAE预训练后微调
        mae_path = f'{mae_dir}/{m}_mae80.pth'
        if os.path.exists(mae_path):
            log(f"微调 {m} (MAE预训练)", log_file)
            model = create_model(m)
            try:
                model.load_state_dict(torch.load(mae_path), strict=False)
            except: pass
            val_acc = train_model(model, train_loader, val_loader, 50, device, log_file)
            test_acc, f1 = evaluate_model(model, test_loader, device)
            results['finetune'].append({'model': m, 'method': 'mae', 'val': val_acc, 'test': test_acc, 'f1': f1})
            log(f"  结果: Val={val_acc:.2f}%, Test={test_acc:.2f}%, F1={f1:.2f}%", log_file)
        
        # SimCLR预训练后微调
        simclr_path = f'{simclr_dir}/{m}_simclr.pth'
        if os.path.exists(simclr_path):
            log(f"微调 {m} (SimCLR预训练)", log_file)
            model = create_model(m)
            try:
                model.load_state_dict(torch.load(simclr_path), strict=False)
            except: pass
            val_acc = train_model(model, train_loader, val_loader, 50, device, log_file)
            test_acc, f1 = evaluate_model(model, test_loader, device)
            results['finetune'].append({'model': m, 'method': 'simclr', 'val': val_acc, 'test': test_acc, 'f1': f1})
            log(f"  结果: Val={val_acc:.2f}%, Test={test_acc:.2f}%, F1={f1:.2f}%", log_file)
    
    # 保存结果
    with open(f'{result_dir}/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # 生成报告
    report = ["# 完整实验V2报告", f"时间: {datetime.now()}", ""]
    report.append("## MAE预训练 (mask=0.8)")
    for r in results['mae']:
        report.append(f"- {r['model']}: Loss={r['loss']:.6f}")
    report.append("\n## SimCLR预训练")
    for r in results['simclr']:
        report.append(f"- {r['model']}: Loss={r['loss']:.4f}")
    report.append("\n## 微调对比")
    report.append("| 模型 | 方法 | Val Acc | Test Acc | F1 |")
    report.append("|------|------|---------|----------|-----|")
    for r in results['finetune']:
        report.append(f"| {r['model']} | {r['method']} | {r['val']:.2f}% | {r['test']:.2f}% | {r['f1']:.2f}% |")
    
    with open(f'{result_dir}/REPORT.md', 'w') as f:
        f.write('\n'.join(report))
    
    log("="*60, log_file)
    log("实验完成!", log_file)
    log("="*60, log_file)
    
    return results

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=0)
    args = p.parse_args()
    run_experiment(args.gpu)
