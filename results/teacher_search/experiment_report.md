# 教师模型搜索实验报告

生成时间: 2025-12-01 20:45:48

## 1. ECA消融实验结果

| 教师模型 | 无ECA准确率 | 有ECA准确率 | 提升幅度 |
|----------|-------------|-------------|----------|
| resnet50 | 0.9863 | 0.9827 | -0.36% |
| densenet121 | 0.9848 | 0.9841 | -0.07% |
| mobilenetv3 | 0.9769 | 0.9754 | -0.14% |
| convnextv2_tiny | 0.9812 | 0.9855 | +0.43% |
| efficientnetv2_s | 0.9834 | 0.9834 | +0.00% |

## 2. Stacking组合实验结果

| 组合名称 | 教师列表 | 测试准确率 | F1-macro | 不一致率 | Oracle准确率 |
|----------|----------|------------|----------|----------|--------------|
| new_trio | convnextv2_tiny, densenet121, efficientnetv2_s | 0.9877 | 0.9849 | 0.0202 | 0.9913 |
| quad_ensemble | resnet50, densenet121, convnextv2_tiny, efficientnetv2_s | 0.9877 | 0.9848 | 0.0238 | 0.9935 |
| full_ensemble | resnet50, densenet121, mobilenetv3, convnextv2_tiny, efficientnetv2_s | 0.9870 | 0.9831 | 0.0303 | 0.9935 |
| replace_resnet | convnextv2_tiny, densenet121, mobilenetv3 | 0.9855 | 0.9821 | 0.0246 | 0.9906 |
| baseline | resnet50, densenet121, mobilenetv3 | 0.9834 | 0.9785 | 0.0246 | 0.9899 |
| replace_mobilenet | resnet50, densenet121, efficientnetv2_s | 0.9827 | 0.9767 | 0.0181 | 0.9913 |

## 3. 推荐配置

**最佳教师组合**: new_trio
- 教师列表: convnextv2_tiny, densenet121, efficientnetv2_s
- 测试准确率: 0.9877
- F1-macro: 0.9849