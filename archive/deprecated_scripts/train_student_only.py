"""
只训练学生模型,使用已有的98%+高精度教师模型
教师模型路径: checkpoints/{resnet50,mbv3,densenet121}_teacher.pth (无ECA)
Stacking路径: checkpoints/stacking_model.pth
"""
from experiments.sd_mkd import ExperimentConfig, run_experiment

print("="*80)
print("训练学生模型 - 使用已有98%+教师")
print("="*80)
print("\n检查教师模型:")
import os
for name in ['resnet50_teacher.pth', 'mbv3_teacher.pth', 'densenet121_teacher.pth', 'stacking_model.pth']:
    path = f'checkpoints/{name}'
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  ✅ {name}: {size_mb:.1f} MB")
    else:
        print(f"  ❌ {name}: 不存在")

print("\n开始训练学生模型 (100 epochs)...")
print("-"*80)

# 配置 - 只训练学生
config = ExperimentConfig(
    dataset="ISCXVPN2016",
    data_root="/walnut_data/yqm/Dataset",
    batch_size=256,
    num_workers=0,
    epochs_teacher=0,      # 跳过教师训练
    epochs_stacking=0,     # 跳过Stacking训练
    epochs_student=100,    # 只训练学生100轮
    lr=1e-3,
    weight_decay=1e-4,
    temperature=4.0,
    output_dir="./checkpoints",
    mode="student_only",
    student_ckpt="student_baseline_fixed.pth",
    disable_monitor=False
)

# 运行训练
results = run_experiment(config)

print("\n" + "="*80)
print("训练完成!")
print("="*80)
if 'test_accuracy' in results:
    acc = results['test_accuracy'] * 100
    print(f"学生模型测试精度: {acc:.2f}%")
    print(f"目标达成: {'✅' if acc >= 98.5 else '❌'} (目标≥98.5%)")
else:
    print(f"结果: {results}")
