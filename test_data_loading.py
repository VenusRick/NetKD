"""
示例脚本：加载和使用预处理图像数据集

展示如何使用 image_loader 模块加载数据集。
"""

import torch
from data_preprocessing import quick_load_dataset


def test_dataset_loading(dataset_name="ISCXVPN2016"):
    """测试数据集加载"""
    
    print(f"{'='*60}")
    print(f"测试加载数据集: {dataset_name}")
    print(f"{'='*60}\n")
    
    try:
        # 加载数据集
        train_loader, val_loader, test_loader, metadata = quick_load_dataset(
            dataset_name=dataset_name,
            dataset_root="/walnut_data/yqm/Dataset",
            batch_size=32,
            num_workers=0,  # Windows下建议设为0
            pin_memory=False,
            val_ratio=0.15,
            test_ratio=0.15,
        )
        
        print(f"\n{'='*60}")
        print("数据集信息:")
        print(f"{'='*60}")
        for key, value in metadata.items():
            if isinstance(value, list) and len(value) > 5:
                print(f"  {key}: {value[:5]}... (共 {len(value)} 项)")
            else:
                print(f"  {key}: {value}")
        
        # 测试获取一个batch
        print(f"\n{'='*60}")
        print("测试数据加载:")
        print(f"{'='*60}")
        
        # 从训练集获取一个batch
        train_batch = next(iter(train_loader))
        images, labels = train_batch
        
        print(f"  训练集 batch shape: {images.shape}")
        print(f"  标签 shape: {labels.shape}")
        print(f"  图像值范围: [{images.min():.3f}, {images.max():.3f}]")
        print(f"  标签示例: {labels[:10].tolist()}")
        
        # 统计各类别样本数
        print(f"\n{'='*60}")
        print("类别分布:")
        print(f"{'='*60}")
        
        train_labels_all = []
        for _, labels in train_loader:
            train_labels_all.extend(labels.tolist())
        
        from collections import Counter
        label_counts = Counter(train_labels_all)
        for label_idx in sorted(label_counts.keys()):
            class_name = metadata['class_names'][label_idx]
            count = label_counts[label_idx]
            print(f"  类别 {label_idx} ({class_name}): {count} 样本")
        
        print(f"\n✅ 数据集 {dataset_name} 加载成功！")
        return True
        
    except Exception as e:
        print(f"\n❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """测试多个数据集"""
    
    # 可用的数据集列表
    available_datasets = [
        "ISCXVPN2016",      # VPN流量分类 (7类)
        "ISCXTor2016",      # Tor流量分类 (8类)
        "USTC-TFC2016",     # 恶意软件和应用流量 (18类)
        # "CrossPlatform-Android",  # Android应用流量（类别较多）
        # "CrossPlatform-iOS",      # iOS应用流量（类别较多）
        # "CICIoT2022",             # IoT设备流量
    ]
    
    print("NetKD 数据集加载测试")
    print("="*60)
    print("数据集根目录: /walnut_data/yqm/Dataset")
    print("="*60)
    
    # 测试第一个数据集
    dataset_to_test = "ISCXVPN2016"  # 可以修改为其他数据集
    
    success = test_dataset_loading(dataset_to_test)
    
    if success:
        print(f"\n\n{'='*60}")
        print("如何在训练中使用:")
        print(f"{'='*60}")
        print("""
from data_preprocessing import quick_load_dataset

# 加载数据
train_loader, val_loader, test_loader, metadata = quick_load_dataset(
    dataset_name="ISCXVPN2016",
    batch_size=32,
)

# 获取类别数
num_classes = metadata['num_classes']

# 开始训练
for epoch in range(num_epochs):
    for images, labels in train_loader:
        # 训练代码...
        pass
        """)


if __name__ == "__main__":
    main()
