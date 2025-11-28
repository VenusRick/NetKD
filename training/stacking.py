"""
Stacking Ensemble Training Module
Combines predictions from multiple teacher models using a meta-learner
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
# sklearn replaced with custom impl
from typing import List, Tuple, Dict
# Custom metrics to avoid sklearn dependency
def accuracy_score(y_true, y_pred):
    return (y_true == y_pred).mean()

def f1_score(y_true, y_pred, average='macro'):
    from collections import defaultdict
    classes = np.unique(np.concatenate([y_true, y_pred]))
    f1s = []
    for c in classes:
        tp = ((y_true == c) & (y_pred == c)).sum()
        fp = ((y_true != c) & (y_pred == c)).sum()
        fn = ((y_true == c) & (y_pred != c)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        f1s.append(f1)
    return np.mean(f1s)

def confusion_matrix(y_true, y_pred):
    n = max(y_true.max(), y_pred.max()) + 1
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

import json
from pathlib import Path


class StackingMetaLearner(nn.Module):
    """
    Meta-learner for stacking ensemble.
    Takes concatenated teacher predictions and learns optimal combination.
    """
    def __init__(self, num_teachers: int, num_classes: int, hidden_dim: int = 64):
        super().__init__()
        input_dim = num_teachers * num_classes
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)


def extract_teacher_predictions(
    teachers: List[nn.Module],
    data_loader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract soft predictions from all teacher models."""
    all_predictions = []
    all_labels = []
    
    for teacher in teachers:
        teacher.eval()
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            
            batch_preds = []
            for teacher in teachers:
                logits = teacher(inputs)
                probs = torch.softmax(logits, dim=1)
                batch_preds.append(probs.cpu().numpy())
            
            batch_preds = np.concatenate(batch_preds, axis=1)
            all_predictions.append(batch_preds)
            all_labels.append(labels.numpy())
    
    predictions = np.vstack(all_predictions)
    labels = np.concatenate(all_labels)
    
    return predictions, labels


def train_stacking_model(
    teachers: List[nn.Module],
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    num_classes: int,
    device: torch.device,
    epochs: int = 50,
    lr: float = 0.001,
    batch_size: int = 256,
    weight_decay: float = 1e-4,
    save_path: str = None,
) -> Dict:
    """Train stacking ensemble model."""
    print("🔄 Extracting teacher predictions...")
    
    train_preds, train_labels = extract_teacher_predictions(teachers, train_loader, device)
    val_preds, val_labels = extract_teacher_predictions(teachers, val_loader, device)
    test_preds, test_labels = extract_teacher_predictions(teachers, test_loader, device)
    
    print(f"  Train: {train_preds.shape}, Val: {val_preds.shape}, Test: {test_preds.shape}")
    
    train_dataset = TensorDataset(torch.FloatTensor(train_preds), torch.LongTensor(train_labels))
    val_dataset = TensorDataset(torch.FloatTensor(val_preds), torch.LongTensor(val_labels))
    test_dataset = TensorDataset(torch.FloatTensor(test_preds), torch.LongTensor(test_labels))
    
    train_meta_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_meta_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_meta_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    num_teachers = len(teachers)
    meta_model = StackingMetaLearner(num_teachers=num_teachers, num_classes=num_classes, hidden_dim=64).to(device)
    
    print(f"\n📊 Meta-learner: {sum(p.numel() for p in meta_model.parameters())/1e3:.2f}K parameters")
    
    optimizer = optim.AdamW(meta_model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5, min_lr=1e-6)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    best_state = None
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print(f"\n🚀 Training stacking meta-learner for {epochs} epochs...")
    print("=" * 70)
    
    for epoch in range(1, epochs + 1):
        meta_model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_meta_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = meta_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        meta_model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_meta_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = meta_model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, preds = outputs.max(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        if epoch <= 5 or epoch % 5 == 0:
            print(f"Epoch {epoch:3d}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} ({val_acc*100:.2f}%)")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = meta_model.state_dict()
            if epoch > 5:
                print(f"  ✅ New best: {best_val_acc*100:.2f}%")
        
        scheduler.step(val_acc)
    
    print("=" * 70)
    
    if best_state is not None:
        meta_model.load_state_dict(best_state)
        if save_path:
            torch.save(best_state, save_path)
            print(f"✅ Best model saved to {save_path}")
    
    meta_model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_meta_loader:
            inputs = inputs.to(device)
            outputs = meta_model(inputs)
            _, preds = outputs.max(1)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    test_acc = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average='weighted')
    test_cm = confusion_matrix(all_labels, all_preds)
    
    results = {
        'history': history,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'confusion_matrix': test_cm.tolist(),
        'num_teachers': num_teachers,
        'num_classes': num_classes,
        'epochs': epochs,
        'lr': lr,
        'batch_size': batch_size
    }
    
    print(f"\n{'='*70}")
    print(f"FINAL STACKING RESULTS")
    print(f"{'='*70}")
    print(f"Best Val Accuracy:  {best_val_acc*100:.2f}%")
    print(f"Test Accuracy:      {test_acc*100:.2f}%")
    print(f"Test F1 Score:      {test_f1:.4f}")
    print(f"{'='*70}\n")
    
    return results


def compare_ensemble_vs_individuals(
    teachers: List[nn.Module],
    test_loader: DataLoader,
    stacking_model: nn.Module,
    device: torch.device,
    num_classes: int
) -> Dict:
    """Compare stacking ensemble with individual teachers and voting."""
    print("\n" + "="*70)
    print("ENSEMBLE COMPARISON")
    print("="*70)
    
    test_preds, test_labels = extract_teacher_predictions(teachers, test_loader, device)
    
    teacher_accs = []
    num_teachers = len(teachers)
    for i in range(num_teachers):
        teacher_pred = test_preds[:, i*num_classes:(i+1)*num_classes].argmax(axis=1)
        acc = accuracy_score(test_labels, teacher_pred)
        teacher_accs.append(acc)
        print(f"Teacher {i+1}: {acc*100:.2f}%")
    
    voting_preds = []
    for i in range(len(test_labels)):
        votes = [test_preds[i, j*num_classes:(j+1)*num_classes].argmax() for j in range(num_teachers)]
        voting_preds.append(np.bincount(votes).argmax())
    voting_acc = accuracy_score(test_labels, voting_preds)
    print(f"\nSimple Voting:  {voting_acc*100:.2f}%")
    
    stacking_model.eval()
    stacking_preds = []
    
    test_dataset = TensorDataset(torch.FloatTensor(test_preds), torch.LongTensor(test_labels))
    test_meta_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    with torch.no_grad():
        for inputs, _ in test_meta_loader:
            inputs = inputs.to(device)
            outputs = stacking_model(inputs)
            _, preds = outputs.max(1)
            stacking_preds.append(preds.cpu().numpy())
    
    stacking_preds = np.concatenate(stacking_preds)
    stacking_acc = accuracy_score(test_labels, stacking_preds)
    print(f"Stacking:       {stacking_acc*100:.2f}%")
    
    print("\nImprovement:")
    print(f"  vs Best Teacher:  {(stacking_acc - max(teacher_accs))*100:+.2f}pp")
    print(f"  vs Voting:        {(stacking_acc - voting_acc)*100:+.2f}pp")
    print("="*70)
    
    return {
        'teacher_accs': teacher_accs,
        'voting_acc': voting_acc,
        'stacking_acc': stacking_acc
    }
