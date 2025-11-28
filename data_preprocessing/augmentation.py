"""数据增强模块 - 专门针对类别不平衡问题"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import random
from pathlib import Path
import os
from tqdm import tqdm


class TrafficImageAugmenter:
    """流量图像数据增强器"""
    
    def __init__(self, rotation_range=5, brightness_range=(0.9, 1.1),
                 contrast_range=(0.9, 1.1), noise_level=0.02, blur_prob=0.3):
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_level = noise_level
        self.blur_prob = blur_prob
    
    def augment_image(self, image_path, output_path, num_augments=5):
        """对单张图像生成多个增强版本"""
        img = Image.open(image_path).convert('RGB')
        base_name = Path(image_path).stem
        
        for i in range(num_augments):
            aug_img = self._apply_augmentation(img)
            save_path = os.path.join(output_path, f"{base_name}_aug{i+1}.png")
            aug_img.save(save_path)
    
    def _apply_augmentation(self, img):
        """应用随机增强变换"""
        aug_img = img.copy()
        
        # 小角度旋转
        if random.random() > 0.5:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            aug_img = aug_img.rotate(angle, resample=Image.BILINEAR)
        
        # 亮度调整
        if random.random() > 0.5:
            factor = random.uniform(*self.brightness_range)
            enhancer = ImageEnhance.Brightness(aug_img)
            aug_img = enhancer.enhance(factor)
        
        # 对比度调整
        if random.random() > 0.5:
            factor = random.uniform(*self.contrast_range)
            enhancer = ImageEnhance.Contrast(aug_img)
            aug_img = enhancer.enhance(factor)
        
        # 高斯噪声
        if random.random() > 0.5:
            aug_img = self._add_gaussian_noise(aug_img)
        
        # 轻微模糊
        if random.random() > (1 - self.blur_prob):
            aug_img = aug_img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        return aug_img
    
    def _add_gaussian_noise(self, img):
        """添加高斯噪声"""
        img_array = np.array(img).astype(np.float32) / 255.0
        noise = np.random.normal(0, self.noise_level, img_array.shape)
        noisy_img = np.clip(img_array + noise, 0, 1)
        return Image.fromarray((noisy_img * 255).astype(np.uint8))


class MixupAugmenter:
    """Mixup数据增强"""
    
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def mixup_data(self, x, y):
        """Mixup增强"""
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        """Mixup损失函数"""
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def balance_dataset(dataset_path, target_samples_per_class=None, augment_multiplier=3):
    """平衡数据集类别分布"""
    augmenter = TrafficImageAugmenter()
    
    class_dirs = sorted([d for d in Path(dataset_path).iterdir() if d.is_dir()])
    class_counts = {}
    
    print("=== 当前类别分布 ===")
    for class_dir in class_dirs:
        class_name = class_dir.name
        images = list(class_dir.glob("*.png"))
        class_counts[class_name] = len(images)
        print(f"{class_name}: {len(images)} 样本")
    
    if target_samples_per_class is None:
        target_samples_per_class = max(class_counts.values())
    
    print(f"\n目标每类样本数: {target_samples_per_class}")
    
    augmentation_summary = {}
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        current_count = class_counts[class_name]
        
        if current_count < target_samples_per_class:
            deficit = target_samples_per_class - current_count
            augments_per_image = min(augment_multiplier, 
                                    int(np.ceil(deficit / current_count)))
            
            print(f"\n增强 {class_name}:")
            print(f"  当前: {current_count}, 目标: {target_samples_per_class}")
            print(f"  每张图增强: {augments_per_image} 次")
            
            images = list(class_dir.glob("*.png"))
            
            for img_path in tqdm(images[:deficit//augments_per_image + 1], 
                                desc=f"  处理中"):
                augmenter.augment_image(
                    str(img_path), 
                    str(class_dir),
                    num_augments=augments_per_image
                )
            
            new_count = len(list(class_dir.glob("*.png")))
            augmentation_summary[class_name] = {
                'original': current_count,
                'augmented': new_count - current_count,
                'total': new_count
            }
    
    return augmentation_summary


def create_balanced_sampler(dataset, target_balance='uniform'):
    """创建平衡采样器"""
    from torch.utils.data import WeightedRandomSampler
    
    targets = np.array([dataset[i][1] for i in range(len(dataset))])
    class_counts = np.bincount(targets)
    
    if target_balance == 'uniform':
        weights = 1.0 / class_counts[targets]
    elif target_balance == 'sqrt':
        weights = 1.0 / np.sqrt(class_counts[targets])
    else:
        raise ValueError(f"Unknown balance strategy: {target_balance}")
    
    weights = weights / weights.sum()
    
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(dataset),
        replacement=True
    )
    
    return sampler
