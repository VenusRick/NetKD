# 教师模型搜索实验报告

生成时间: 2025-12-01 21:58:48

## 1. ECA消融实验结果

| 教师模型 | 无ECA准确率 | 有ECA准确率 | 提升幅度 |
|----------|-------------|-------------|----------|
| resnet50 | 0.9870 | 0.9834 | -0.36% |
| densenet121 | 0.9863 | 0.9863 | +0.00% |
| mobilenetv3 | 0.9754 | 0.9783 | +0.29% |
| convnextv2_tiny | 0.9870 | 0.9863 | -0.07% |
| efficientnetv2_s | 0.9855 | 0.9848 | -0.07% |

## 2. Stacking组合实验结果

| 组合名称 | 教师列表 | 测试准确率 | F1-macro | 不一致率 | Oracle准确率 |
|----------|----------|------------|----------|----------|--------------|
| replace_resnet | convnextv2_tiny, densenet121, mobilenetv3 | 0.9906 | 0.9865 | 0.0246 | 0.9935 |
| new_trio | convnextv2_tiny, densenet121, efficientnetv2_s | 0.9884 | 0.9841 | 0.0210 | 0.9942 |
| full_ensemble | resnet50, densenet121, mobilenetv3, convnextv2_tiny, efficientnetv2_s | 0.9884 | 0.9843 | 0.0303 | 0.9949 |
| baseline | resnet50, densenet121, mobilenetv3 | 0.9877 | 0.9833 | 0.0224 | 0.9913 |
| replace_mobilenet | resnet50, densenet121, efficientnetv2_s | 0.9877 | 0.9833 | 0.0188 | 0.9913 |
| quad_ensemble | resnet50, densenet121, convnextv2_tiny, efficientnetv2_s | 0.9863 | 0.9815 | 0.0238 | 0.9942 |

## 3. 推荐配置

**最佳教师组合**: replace_resnet
- 教师列表: convnextv2_tiny, densenet121, mobilenetv3
- 测试准确率: 0.9906
- F1-macro: 0.9865