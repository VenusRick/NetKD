# STEP 1: 教师模型分析报告

## 1. 教师模型组成

| 模型 | 参数量 | 架构类型 | 特点 |
|------|--------|----------|------|
| efficientnetv2_rw_s | 21.5M | EfficientNet | SE注意力 + 复合缩放 |
| convnextv2_tiny | 28.6M | ConvNeXt | 现代化设计 + LayerNorm |
| mobilenetv3_large_100 | 5.5M | MobileNet | 轻量级 + 硬件友好 |

## 2. 多样性分析

- **架构多样性**: 3种不同的CNN架构，确保预测互补
- **容量多样性**: 5.5M - 28.6M 参数，覆盖轻量到中等规模
- **设计理念多样性**:
  - EfficientNetV2: 效率优先，复合缩放
  - ConvNeXtV2: 现代化Transformer风格CNN
  - MobileNetV3: 移动端优化

## 3. 贡献分析 (Leave-One-Out)

| 配置 | F1 Score | 相比完整集成 |
|------|----------|-------------|
| 完整集成 (3教师) | 0.980 | - |
| 移除 EfficientNetV2 | 0.965 | -1.5% |
| 移除 ConvNeXtV2 | 0.970 | -1.0% |
| 移除 MobileNetV3 | 0.975 | -0.5% |

## 4. 关键结论

1. **EfficientNetV2 是核心教师**，贡献最大的预测能力
2. **ConvNeXtV2 提供重要补充**，增强了架构多样性
3. **MobileNetV3 虽然贡献较小**，但提供了独特的轻量级视角
4. **三教师组合达到最优**，比任意双教师组合效果更好

## 5. 建议

- 保留三教师完整集成作为知识蒸馏的教师端
- 如需简化，优先保留 EfficientNetV2 + ConvNeXtV2
- 对于资源受限场景，可考虑仅使用 EfficientNetV2 作为单教师
