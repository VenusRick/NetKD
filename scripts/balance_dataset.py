#!/usr/bin/env python
"""
数据集平衡脚本
针对Class 2, 4, 5等样本不足的类别进行数据增强
"""

import sys
import os
sys.path.insert(0, '/walnut_data/yqm/NetKD')

from data_preprocessing.augmentation import balance_dataset
import argparse


def main():
    parser = argparse.ArgumentParser(description='平衡数据集类别分布')
    parser.add_argument('--dataset', type=str, 
                       default='/walnut_data/yqm/Dataset/ISCXVPN2016/train',
                       help='数据集路径')
    parser.add_argument('--target', type=int, default=None,
                       help='目标每类样本数(默认使用最大类样本数)')
    parser.add_argument('--multiplier', type=int, default=3,
                       help='增强倍数(每张原图最多生成N个增强版本)')
    parser.add_argument('--classes', type=str, default=None,
                       help='仅增强指定类别,逗号分隔(如: streaming,email,p2p)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("数据集平衡工具 - ISCXVPN2016")
    print("=" * 60)
    
    # 如果指定了特定类别,只增强这些类别
    if args.classes:
        target_classes = args.classes.split(',')
        print(f"\n目标类别: {target_classes}")
        
        from pathlib import Path
        from data_preprocessing.augmentation import TrafficImageAugmenter
        from tqdm import tqdm
        import numpy as np
        
        augmenter = TrafficImageAugmenter()
        
        for class_name in target_classes:
            class_dir = Path(args.dataset) / class_name.strip()
            if not class_dir.exists():
                print(f"警告: 类别目录不存在 - {class_dir}")
                continue
            
            images = list(class_dir.glob("*.png"))
            original_count = len([img for img in images if '_aug' not in img.name])
            
            print(f"\n处理类别: {class_name}")
            print(f"原始样本数: {original_count}")
            
            # 只对原始图像进行增强(跳过已增强的)
            original_images = [img for img in images if '_aug' not in img.name]
            
            for img_path in tqdm(original_images, desc=f"增强{class_name}"):
                augmenter.augment_image(
                    str(img_path),
                    str(class_dir),
                    num_augments=args.multiplier
                )
            
            new_count = len(list(class_dir.glob("*.png")))
            print(f"增强后样本数: {new_count} (+{new_count - len(images)})")
    
    else:
        # 全局平衡
        summary = balance_dataset(
            args.dataset,
            target_samples_per_class=args.target,
            augment_multiplier=args.multiplier
        )
        
        print("\n" + "=" * 60)
        print("增强总结")
        print("=" * 60)
        
        if summary:
            for class_name, stats in summary.items():
                increase_pct = (stats['total'] - stats['original']) / stats['original'] * 100
                print(f"\n{class_name}:")
                print(f"  原始样本: {stats['original']:>5}")
                print(f"  新增样本: {stats['augmented']:>5}")
                print(f"  总计样本: {stats['total']:>5} (+{increase_pct:.1f}%)")
        else:
            print("\n所有类别已平衡,无需增强")


if __name__ == "__main__":
    main()
