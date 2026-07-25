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

# 任务：基于 ConvNeXt 迁移学习的近距离小物体（笔/橡皮/积木）图像分类模型训练脚本编写

## 1. 项目背景与场景定义
* **应用场景**：小车搭载 800 万像素、160° 鱼眼摄像头，在近距离视角下识别孩子手中拿取的物品（铅笔、橡皮、积木块）。
* **场景挑战**：图片存在鱼眼边缘畸变、目标较小（特别是笔和橡皮）、手部遮挡较多以及复杂背景干扰。
* **任务目标**：编写一个健壮、高效率的 Python 迁移学习训练脚本，对现有的标注数据集进行微调训练，实现对这三类小高精度分类。

---

## 2. 运行环境与硬件约束
* **计算设备**：桌面端台式机
* **GPU 规格**：NVIDIA GeForce RTX 4070 (12GB 显存)
* **优化要求**：支持 PyTorch 自动混合精度（AMP / `torch.cuda.amp`），以充分利用 Tensor Cores 并节省显存；数据加载需配置合适数量的 `num_workers` 和 `pin_memory=True`。

---

## 3. 数据集规范与划分
* **原始数据目录**：`Project/request_recogniser/collected_images`（已通过 `auto_marker.py` 生成对应标注信息）。
* **分类类别**（共 3 类）：
  1. `pencil` (铅笔)
  2. `eraser` (橡皮)
  3. `building_block` (积木块)
* **数据集划分比例**：
  * 训练集（Training Set）：85%
  * 验证集（Validation Set）：15%
  * **划分要求**：采用分层抽样（Stratified Split）或固定随机种子（`seed=42`），确保验证集中各类别的比例与总数据集一致，保证划分可复现。

---

## 4. 模型架构与预训练权重
* **Backbone 架构**：ConvNeXt（优先采用 `convnext_small` 或 `convnext_base`，适配 12GB 显存）。
* **权重来源**：加载 ImageNet 预训练权重（如 `ConvNeXt_Small_Weights.DEFAULT` 或 `timm` 库对应的预训练权重）。
* **Head 修改**：替换顶层分类头（Classification Head），将最终输出维度修改为 3（对应上述 3 个类别）。

---

## 5. 参考代码结构
* **参考脚本**：`Baseline/train_cnn_v2.py`（原用于五种猫的分类训练）。
* **重构规范**：继承参考脚本的编码风格和模块化划分，但在训练循环、数据增强、评价指标和早停机制上进行专业升级。

---

## 6. 具体程序功能与代码实现细节要求

### A. 针对小物体与畸变的数据增强（Data Augmentation）
针对鱼眼与手拿小物体的特殊性，设计 PyTorch `transforms` 或 `albumentations` 流水线：
1. **训练集增强**：
   * 尺寸缩放：输入分辨率设定为 $384 \times 384$ 或 $448 \times 448$（利用 RTX 4070 显存优势保留小物体高频细节，切勿盲目缩放到 $224 \times 224$）。
   * 随机裁剪与缩放（`RandomResizedCrop`，scale 在 0.6~1.0 之间）。
   * 随机水平翻转、随机旋转（±15°，模拟摄像头/手捏角度偏差）。
   * 颜色抖动（`ColorJitter`，轻微改变亮度、对比度，应对小车移动时的环境光照变化）。
   * 归一化（ImageNet mean/std）。
2. **验证集处理**：
   * Resize 到固定分辨率（如 $384 \times 384$）+ Center Crop + 归一化。

### B. 训练策略与损失函数
1. **优化器**：推荐使用 `AdamW` 优化器，基础学习率设为 `1e-4`，设置权重衰减（`weight_decay=1e-2`）。
2. **学习率调度**：使用余弦退火调度器（`CosineAnnealingLR` 或 `ReduceLROnPlateau`）。
3. **损失函数**：使用带标签平滑（Label Smoothing，如 `label_smoothing=0.1`）的 `CrossEntropyLoss`，降低由于手部遮挡或标注噪声导致的过拟合风险。

### C. 早停机制（Early Stopping）与模型保存
1. **监控指标**：监控验证集 Loss（`val_loss`）和 Macro F1-score。
2. **早停逻辑**：设置 `patience=10`，若连续 10 个 Epoch 验证集 Loss 未降低，则自动触发 Early Stopping 中止训练。
3. **checkpoint 保存**：
   * 训练过程中仅保存 Validation F1-score 最高的那一套最佳权重（`best_model.pth`）。
   * 保存包含分类映射字典（`class_to_idx`）、最佳 Epoch、优化器状态的完整字典，方便后续推理与部署。

### D. 日志记录与可视化
1. 打印清晰的 Epoch 进度日志（显示 Train Loss/Acc, Val Loss/Acc/F1, 当前学习率, 耗时）。
2. 将训练过程中的指标曲线绘制导出为本地图片 `training_metrics.png`，或集成 `TensorBoard` / `wandb` 记录。

---

## 7. 交付产物
1. 一个完整的、注释详尽的 Python 脚本文件 `train_item_recogniser.py`。
2. 脚本需要包含可调参数配置区域（如 Batch Size、Epochs、Learning Rate、Image Size、Data Path 等），便于后期调优。
3. 包含命令行参数解析（`argparse`）或清晰的结构化配置类（`Config` class）。
