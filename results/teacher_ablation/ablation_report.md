# 教师模型消融实验报告

生成时间: 2025-12-16 22:45:52

## 实验概述

本实验探究不同教师模型组合对知识蒸馏性能的影响。

最优教师组合: DenseNet121-ECA + MobileNetV3-Large-ECA + ConvNeXtV2-Tiny-ECA

## 实验结果

| 组合名称 | 教师数量 | 教师模型 | 集成测试准确率 | 训练时间 |
|----------|----------|----------|----------------|----------|
| full_3teachers | 3 | DenseNet121-ECA, MobileNetV3-Large-ECA, ConvNeXtV2-Tiny-ECA | 98.55% | 15.0min |
| remove_convnext | 2 | DenseNet121-ECA, MobileNetV3-Large-ECA | 98.34% | 13.4min |
| only_densenet | 1 | DenseNet121-ECA | 98.27% | 8.8min |
| remove_densenet | 2 | MobileNetV3-Large-ECA, ConvNeXtV2-Tiny-ECA | 98.19% | 9.0min |
| only_mobilenet | 1 | MobileNetV3-Large-ECA | 97.98% | 5.1min |
| only_convnext | 1 | ConvNeXtV2-Tiny-ECA | 97.98% | 5.3min |
| remove_mobilenet | 2 | DenseNet121-ECA, ConvNeXtV2-Tiny-ECA | 97.90% | 14.0min |

## 消融分析

### 删除单个教师的影响

- 删除 DenseNet121: 98.19% (相比完整组合 -0.36%)
- 删除 MobileNetV3: 97.90% (相比完整组合 -0.65%)
- 删除 ConvNeXtV2: 98.34% (相比完整组合 -0.22%)

### 单个教师的性能

- DenseNet121-ECA: 98.27%
- MobileNetV3-Large-ECA: 97.98%
- ConvNeXtV2-Tiny-ECA: 97.98%

## 结论

(实验完成后根据结果填写)