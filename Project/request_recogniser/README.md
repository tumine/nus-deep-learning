# 物品请求识别模型训练

本目录的 `train_classifier.py` 使用 ImageNet 预训练 ConvNeXt 进行迁移学习，识别小车摄像头抓拍的三类物品：

- `pencil`：铅笔
- `eraser`：橡皮
- `building_block`：积木块

训练时会对每个类别独立地按 $85\%$ 训练集和 $15\%$ 验证集划分。默认随机种子为 `42`，所以同一批输入数据会得到可复现的划分。

## 环境准备

建议在配备 NVIDIA RTX 4070 的桌面环境中，安装支持 CUDA 的 PyTorch。项目依赖定义在仓库根目录的 `pyproject.toml` 中；其中 CUDA 版 PyTorch 需要从 PyTorch 官方索引安装：

```powershell
pip install -e .
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

训练脚本在检测到 CUDA 后会自动启用 `torch.amp` 自动混合精度、`GradScaler`、cuDNN benchmark 和 TF32。请先确认：

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 训练数据

默认数据目录为 `Project/request_recogniser/collected_images`。可通过 `--data` 指定其他目录。标注程序完成标注后，推荐将图片整理为下面的类别目录结构：

```text
Project/request_recogniser/
├── collected_images/
│   ├── pencil/
│   │   ├── pencil_001.jpg
│   │   └── ...
│   ├── eraser/
│   │   ├── eraser_001.jpg
│   │   └── ...
│   └── building_block/
│       ├── block_001.jpg
│       └── ...
└── train_classifier.py
```

脚本会递归查找名称为 `pencil`、`eraser` 和 `building_block` 的目录，并接受 `.jpg`、`.jpeg`、`.png`、`.bmp`、`.webp` 图片。每个类别至少需要两张图片，且三类均不能缺失。

如果 `auto_marker.py` 输出的是标注清单而非类别目录，也可在数据根目录放置以下任一文件：`labels.csv`、`annotations.csv`、`labels.json`、`annotations.json`。每条记录必须有：

- 图片路径字段：`path`、`file`、`image_path` 或 `image`
- 类别字段：`label`、`class` 或 `category`

例如 `labels.csv`：

```csv
path,label
raw/pencil_001.jpg,pencil
raw/eraser_001.jpg,eraser
raw/block_001.jpg,building_block
```

图片路径相对于清单所在的数据根目录。类别目录优先于清单读取。

## 开始训练

在仓库根目录执行：

```powershell
python Project/request_recogniser/train_classifier.py
```

默认使用 ConvNeXt-Tiny、40 个 epoch、学习率 `3e-4`、权重衰减 `1e-2`、标签平滑 `0.1` 和耐心值 6。Tiny 和 Small 默认 batch size 为 64，Base 默认 batch size 为 32；显存不足时可手动减小：

```powershell
python Project/request_recogniser/train_classifier.py --variant small --epochs 50 --batch-size 32
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--data PATH` | 数据集根目录，默认 `collected_images` |
| `--variant {tiny,small,base}` | ConvNeXt 模型规模，默认 `tiny` |
| `--epochs N` | 最大训练轮数，默认 40 |
| `--batch-size N` | 每批图片数；不设置时根据模型规模选 64 或 32 |
| `--learning-rate RATE` | AdamW 初始学习率，默认 `3e-4` |
| `--weight-decay RATE` | AdamW 权重衰减，默认 `1e-2` |
| `--label-smoothing RATE` | 交叉熵标签平滑，默认 `0.1` |
| `--patience N` | 验证损失连续不改善时的早停轮数，默认 6 |
| `--seed N` | 划分及训练随机种子，默认 42 |
| `--num-workers N` | DataLoader 工作进程数，默认 4 |
| `--no-pretrained` | 不加载 ImageNet 预训练权重 |
| `--no-export-onnx` | 训练完成后不导出 ONNX |

## 训练过程与输出

训练集采用随机裁剪、亮度/对比度/饱和度抖动、水平翻转及 $\pm15^\circ$ 旋转，模拟小车运动、光照和视角变化；验证集只做 Resize、中心裁剪和 ImageNet 归一化。

每个 epoch 会输出训练损失、训练准确率、验证损失、验证准确率和学习率。优化器为 AdamW，学习率由 `CosineAnnealingLR` 按余弦退火调整。早停依据验证损失，最佳部署模型则依据验证准确率保存。

训练结束后会生成：

```text
Project/request_recogniser/
├── weights/best_convnext.pth  # 验证准确率最高的 PyTorch 权重及类别元数据
├── model.onnx                 # 动态 batch 维度的 ONNX 推理模型
└── model.pt                   # TorchScript 推理模型，可通过 torch.jit.load 加载
```

控制台还会输出验证集的 Precision、Recall、F1-score 和混淆矩阵。ONNX 输入名为 `image`，形状为 `[batch, 3, 224, 224]`；输出名为 `logits`，形状为 `[batch, 3]`，类别顺序固定为 `pencil`、`eraser`、`building_block`。