#!/usr/bin/env python
"""快速测试: 增强Class 5并评估效果"""

import sys
import os
sys.path.insert(0, '/walnut_data/yqm/NetKD')

from pathlib import Path
from data_preprocessing.augmentation import TrafficImageAugmenter
import random


def quick_test_augmentation():
    print("=" * 70)
    print("快速测试: Class 5 (streaming) 数据增强")
    print("=" * 70)
    
    class_dir = Path('/walnut_data/yqm/Dataset/ISCXVPN2016/train/streaming')
    test_output = Path('/walnut_data/yqm/NetKD/augmentation_test_samples')
    test_output.mkdir(exist_ok=True)
    
    if not class_dir.exists():
        print(f"错误: 目录不存在 - {class_dir}")
        return
    
    original_images = [img for img in class_dir.glob("*.png") 
                      if '_aug' not in img.name]
    
    print(f"\n当前Class 5统计:")
    print(f"  原始样本数: {len(original_images)}")
    print(f"  总样本数: {len(list(class_dir.glob('*.png')))}")
    
    test_samples = random.sample(original_images, min(5, len(original_images)))
    
    print(f"\n生成测试样本:")
    print(f"  选择图像数: {len(test_samples)}")
    print(f"  每张生成: 3个增强版本")
    print(f"  保存位置: {test_output}")
    
    augmenter = TrafficImageAugmenter(
        rotation_range=5,
        brightness_range=(0.9, 1.1),
        contrast_range=(0.9, 1.1),
        noise_level=0.02,
        blur_prob=0.3
    )
    
    print("\n开始生成...")
    for i, img_path in enumerate(test_samples, 1):
        print(f"  [{i}/{len(test_samples)}] {img_path.name}")
        augmenter.augment_image(str(img_path), str(test_output), num_augments=3)
    
    generated_files = list(test_output.glob("*.png"))
    print(f"\n生成完成!")
    print(f"  总文件数: {len(generated_files)}")
    
    print(f"\n示例文件:")
    for f in sorted(generated_files)[:6]:
        size_kb = f.stat().st_size / 1024
        print(f"  - {f.name} ({size_kb:.1f} KB)")
    
    print("\n" + "=" * 70)
    print("下一步: python scripts/balance_dataset.py --classes streaming --multiplier 3")
    print("=" * 70)


if __name__ == "__main__":
    quick_test_augmentation()
