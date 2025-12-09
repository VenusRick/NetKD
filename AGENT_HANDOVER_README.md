
---

## 📡 2025-12-09 代理配置尝试

### 网络代理配置

**代理地址**: http://172.26.0.1:7891  
**配置方式**:
```bash
export http_proxy=http://172.26.0.1:7891
export https_proxy=http://172.26.0.1:7891
```

**测试结果**:
- ✅ 代理连接成功 (curl test passed)
- ✅ 可以访问 Huggingface
- ⚠️ timm模型下载速度非常慢 (~200KB/s)
- ❌ resnet50 (102MB) 下载卡在98%

### CE+KL实验状态

由于网络下载速度限制，timm预训练模型无法及时下载完成。

**替代方案建议**:
1. 使用本地已训练的教师模型 (results/teacher_search_bs128/)
2. 在网络条件好时重新运行下载
3. 手动下载模型文件放到 ~/.cache/huggingface/

**本地教师模型路径**:
- efficientnetv2_rw_s: `results/teacher_search_bs128/efficientnetv2_rw_s/best_model.pt`
- convnextv2_tiny: `results/teacher_search_bs128/convnextv2_tiny/best_model.pt`  
- mobilenetv3_large_100: `results/teacher_search_bs128/mobilenetv3_large_100/best_model.pt`


---

## 📡 2025-12-09 代理配置尝试

### 网络代理配置
- **代理地址**: http://172.26.0.1:7891
- **测试结果**: ✅ 连接成功，但下载速度慢
- **CE+KL状态**: 受网络限制未完成

### 替代方案
使用本地教师模型: `results/teacher_search_bs128/*/best_model.pt`

