## Date: 25 July, 2026

---

### 需求
编写一个模型训练程序，基于现有的图片迁移训练一个预训练模型，分类识别铅笔、橡皮、积木块。

### 交付要求
一个 Python 程序，可以在预训练架构上针对训练集进行进一步训练。

### 部署位置
代码部署在台式机上，具有 NVIDIA GeForce RTX 4070 显卡（12GB 显存）。

### 参考程序
`Baseline/train_cnn_v2.py` 程序，用于训练可以区分五种猫的模型。

### 训练集、验证集图片来源和划分
- 图片来源：`Project/request_recogniser/collected_images` 目录，需预先通过 `auto_marker.py` 程序进行图片标注。
- 比例划分：训练集 85%，验证集 15%。

### 采用的架构和预训练权重（与 Baseline 相同）
ConvNeXt

### 程序实现详细要求
1. 需要加载预训练模型。
2. 需要按 [训练集、验证集图片来源和划分] 中的划分比例划分训练集和验证集。
3. 参考 [参考程序] 的实现，采用合适的迭代轮次数，并通过一定的指标触发早停、防止过拟合。

---
---

# 任务目标
编写一个基于 PyTorch 的图像分类模型训练程序（`train_classifier.py`），利用迁移学习（Transfer Learning）对预训练 ConvNeXt 模型进行微调，实现小车搭载摄像头抓拍物品的类型识别。

---

### 1. 部署与硬件环境
- **硬件平台**: Desktop (NVIDIA GeForce RTX 4070, 12GB VRAM)
- **性能优化要求**: 
  - 必须启用自动混合精度训练（`torch.cuda.amp` / `torch.amp`），以充分利用 RTX 4070 的 Tensor Cores 并降低显存开销。
  - 开启 `torch.backends.cudnn.benchmark = True` 提升卷积计算效率。
  - Batch Size 建议设置为 32 或 64（根据 12GB 显存动态调整）。

---

### 2. 参考程序与目标类别
- **参考基线**: 复用 `Baseline/train_cnn_v2.py` 的代码架构习惯（如配置读取、训练/验证循环逻辑）。
- **模型架构与权重**: ConvNeXt（加载 ImageNet 预训练权重）。
- **目标分类**: 3 类，对应标签名：
  1. `pencil` (铅笔)
  2. `eraser` (橡皮)
  3. `building_block` (积木块)

---

### 3. 数据集划分与处理
- **数据路径**: `Project/request_recogniser/collected_images`
- **标注来源**: 来源于 `auto_marker.py` 的标注结果（需支持自动读取其生成的子目录或标注文件）。
- **数据集划分**: 训练集 85%，验证集 15%（设置固定 `random_seed` 以保证划分可复现）。
- **车载抓拍数据增强 (Data Augmentation)**:
  - 考虑到小车运动及摄像头抽帧的特点（可能有轻微模糊、变焦、光照变化），数据增强需包含：
    - `RandomResizedCrop` / `Resize`（适配 ConvNeXt 输入尺寸，如 224x224）
    - `ColorJitter`（随机亮度、对比度抖动，模拟环境光变化）
    - `RandomHorizontalFlip` 与轻微 `RandomRotation`（±15 度，模拟视角倾斜）

---

### 4. 详细代码实现要求

#### 4.1 模型构建
- 加载预训练 ConvNeXt 权重，要求可以选择权重参数规模（Tiny, Small, Base 等）。
- 修改最后的分类头（Classifier Head），使其输出维度为 3。

#### 4.2 训练策略与防止过拟合
- **优化器与学习率**: 使用 `AdamW` 优化器，配合余弦退火调度器 (`CosineAnnealingLR`) 或按需衰减。
- **防止过拟合**:
  - 结合 Label Smoothing (如 0.1) 和 Weight Decay (如 1e-2)。
  - 实现基于 **Validation Loss** 的早停机制（Early Stopping，设置 `patience=5~8` 轮未改进即停止）。

#### 4.3 评估、日志与模型保存
- **过程记录**: 每个 Epoch 打印 `Train Loss`, `Train Acc`, `Val Loss`, `Val Acc` 及当前 Learning Rate。
- **最佳模型**: 自动保存验证集 Acc 最高的权重至 `weights/best_convnext.pth`。
- **模型评估**: 训练结束后，在验证集上计算并打印分类报告（Precision, Recall, F1-score）与混淆矩阵（Confusion Matrix）。
- **推理准备**: 在训练完成后，提供将最佳模型自动导出为 `model.onnx` 格式的逻辑，方便后续部署至小车边缘端。
