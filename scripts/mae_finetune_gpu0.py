import sys
sys.path.insert(0, '/workspace/yqm/NetKD')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import json
from datetime import datetime
import timm
from data.dataset import get_dataloaders
from sklearn.metrics import accuracy_score, f1_score

output_dir = 'results/full_pipeline_20251208_004257/gpu0_mae_finetune'
os.makedirs(output_dir, exist_ok=True)
log_file = f'{output_dir}/experiment.log'

def log(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    with open(log_file, 'a') as f:
        f.write(line + '\\n')

log('Loading dataset...')
train_loader, val_loader, test_loader = get_dataloaders(
    '/workspace/yqm/Dataset/ISCXVPN2016',
    batch_size=128, num_workers=8, img_size=40
)
log(f'Dataset loaded')

mae_checkpoints = {
    'mae_0.75': 'results/full_pipeline_20251208_004257/gpu0_full_fixed/pretrain/convnextv2_tiny_mae_mask0.75_encoder.pth',
    'mae_0.8': 'results/full_pipeline_20251208_004257/gpu0_full_fixed/pretrain/convnextv2_tiny_mae_mask0.8_encoder.pth',
}

results = []

for mae_name, mae_path in mae_checkpoints.items():
    if os.path.exists(mae_path):
        log(f'Testing {mae_name} pretrained model...')
        model = timm.create_model('convnextv2_tiny', pretrained=False, num_classes=12, in_chans=1)
        state_dict = torch.load(mae_path, map_location='cpu', weights_only=True)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        log(f'Loaded {mae_name}: missing={len(missing)}, unexpected={len(unexpected)}')
        model = model.cuda()
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
        criterion = torch.nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        scaler = torch.amp.GradScaler('cuda')
        
        best_val = 0
        for epoch in range(50):
            model.train()
            for x, y in train_loader:
                x, y = x.cuda(), y.cuda()
                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    out = model(x)
                    loss = criterion(out, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            scheduler.step()
            
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.cuda(), y.cuda()
                    out = model(x)
                    correct += (out.argmax(1) == y).sum().item()
                    total += y.size(0)
            val_acc = correct / total * 100
            if val_acc > best_val: best_val = val_acc
            if (epoch + 1) % 10 == 0:
                log(f'  [{mae_name}] Epoch {epoch+1}/50, Val: {val_acc:.2f}%')
        
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()
                all_preds.extend(model(x).argmax(1).cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        
        test_acc = accuracy_score(all_labels, all_preds) * 100
        f1 = f1_score(all_labels, all_preds, average='weighted') * 100
        results.append({'name': f'convnextv2_tiny_{mae_name}_finetune', 'val_acc': best_val, 'test_acc': test_acc, 'f1': f1})
        log(f'[{mae_name}] Completed: Val={best_val:.2f}%, Test={test_acc:.2f}%, F1={f1:.2f}%')
    else:
        log(f'{mae_name} checkpoint not found: {mae_path}')

with open(f'{output_dir}/mae_finetune_results.json', 'w') as f:
    json.dump(results, f, indent=2)
log('MAE finetune experiment completed!')
log(f'Results: {results}')
