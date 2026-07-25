# 小物体检测训练流程

本目录用于训练一个识别手持细小物体的 YOLOv11n-P2 检测模型。目标类别固定为笔、橡皮和积木块，模型在 YOLO11 Nano 的基础上增加了 P2/4 检测头，以改善高分辨率画面中小目标的召回率。

| 类别 ID | 类别名称 | 含义 |
| --- | --- | --- |
| `0` | `pen` | 笔 |
| `1` | `eraser` | 橡皮 |
| `2` | `building_block` | 积木块 |

## 文件说明

| 文件 | 用途 |
| --- | --- |
| [collect_resource_images.py](collect_resource_images.py) | 从 Bing Images 搜集训练素材。 |
| [auto_marker.py](auto_marker.py) | 使用 Grounding DINO 为图片生成初始 YOLO 标注。 |
| [train_yolo_v11_p2.py](train_yolo_v11_p2.py) | 验证数据、划分训练/验证集并执行迁移学习训练。 |
| [yolov11n-p2.yaml](yolov11n-p2.yaml) | YOLO11 Nano + P2 小目标检测头的网络定义。 |

## 训练流程概览

1. 准备包含图片和 YOLO 标注的源数据。
2. 检查或修正自动标注结果，尤其是手部遮挡、鱼眼畸变和极小目标。
3. 安装 CUDA 版 PyTorch 与项目依赖。
4. 运行训练脚本。脚本会校验标注、以固定随机种子划分 85% 训练集和 15% 验证集、生成 `dataset.yaml`，然后加载 COCO 预训练权重进行微调。
5. 检查 `best.pt`、验证指标和训练曲线；用真实小车摄像头画面做独立测试。

## 1. 数据准备

### 1.1 推荐的数据布局

训练脚本接受以下两种布局，并递归扫描 `.jpg`、`.jpeg`、`.png`、`.bmp`、`.webp` 图片。

**布局 A：图片和标签同级**

```text
collected_images/
  frame_0001.jpg
  frame_0001.txt
  frame_0002.jpg
  frame_0002.txt
```

**布局 B：标准 YOLO `images/` 与 `labels/` 镜像目录**

```text
yolo_dataset/
  images/
    image_0001.jpg
    image_0002.jpg
  labels/
    image_0001.txt
    image_0002.txt
```

`auto_marker.py` 默认把 Grounding DINO 的输出写入 `Project/request_recogniser/yolo_dataset`，因此运行自动标注后，应使用布局 B 对应的路径作为 `--data-path`。

### 1.2 YOLO 标签格式

每张图片应有一个同名 `.txt` 文件。每个目标占一行，字段顺序如下：

```text
<class_id> <x_center> <y_center> <width> <height>
```

坐标必须相对于图片宽高归一化到 $[0, 1]$。例如，一支笔的标注可以是：

```text
0 0.5125 0.4800 0.3300 0.1200
```

脚本会在训练前检查：

- 每行是否恰好包含 5 个数值字段；
- 类别是否只在 `0`、`1`、`2` 之间；
- 中心点、宽度和高度是否在合法归一化范围内；
- 每张图片是否能找到匹配的标签文件。

空的 `.txt` 文件代表没有目标的背景图，是合法数据。没有 `.txt` 的图片默认会令训练停止，避免无意中丢失标注；若这些图片确实是背景图，可加入 `--allow-unlabeled`，脚本会为它们生成空标签。

### 1.3 标注质量建议

自动标注只适合作为起点。训练前建议使用标注工具逐张抽查，并优先修正以下情况：

- 目标被孩子手指遮挡时，框应覆盖仍可见的物体区域，而不是整只手。
- 笔细长且倾斜时，边界框要覆盖笔尖和笔尾，避免将其裁成很短的局部。
- 鱼眼边缘处的物体应按畸变后的实际像素范围标框。
- 同一帧中多个物体分别标注；不要把笔和橡皮合成一个框。
- 保留一部分无目标、模糊、逆光和强遮挡画面作为背景样本，降低误报。

训练集和验证集都应包含来自真实 160 度鱼眼摄像头的画面。不要只用网络搜集的白底商品图，否则模型在小车摄像头中的距离、畸变、手部遮挡和背景条件下往往泛化较差。

## 2. 环境安装

项目需要 Python 3.11+、NVIDIA 驱动、CUDA 可用的 PyTorch，以及 `ultralytics`。在仓库根目录执行以下命令。

```powershell
# 创建并激活虚拟环境（首次使用）
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 安装与 NVIDIA 驱动匹配的 CUDA 版 PyTorch。
# 项目 pyproject.toml 当前建议使用 cu126；根据实际环境调整版本。
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 安装项目其余依赖（包括 Ultralytics、PyYAML、OpenCV 等）
pip install -e .
```

安装后检查 GPU 是否可用：

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python -c "import ultralytics; print(ultralytics.__version__)"
```

第一条命令应显示 `True` 和 RTX 4070 的名称。若为 `False`，先不要启动训练；应检查 NVIDIA 驱动、CUDA PyTorch wheel 与当前 Python 环境是否一致。

## 3. 开始训练

在仓库根目录执行。默认数据路径为 `Project/request_recogniser/collected_images`，默认将拆分后的数据写入 `Project/request_recogniser/prepared_yolo_dataset`。

```powershell
python Project/request_recogniser/train_yolo_v11_p2.py --epochs 150 --batch-size 16 --imgsz 800
```

若数据由 `auto_marker.py` 生成：

```powershell
python Project/request_recogniser/train_yolo_v11_p2.py `
  --data-path Project/request_recogniser/yolo_dataset `
  --epochs 150 `
  --batch-size 16 `
  --imgsz 800
```

### 3.1 RTX 4070 推荐配置

RTX 4070 有 12 GB 显存，建议先从以下配置开始：

```powershell
python Project/request_recogniser/train_yolo_v11_p2.py `
  --data-path Project/request_recogniser/yolo_dataset `
  --epochs 150 `
  --imgsz 800 `
  --batch-size 16 `
  --workers 8
```

- `imgsz=800`：默认配置，兼顾小目标细节和显存压力。
- `imgsz=1024`：小目标极小且显存允许时使用；建议先将 `--batch-size` 降至 `8` 或 `12`。
- `batch-size=16`：12 GB 显存的保守起点。若发生 CUDA out-of-memory，依次尝试 `12`、`8`。
- `batch-size=-1`：委托 Ultralytics 自动估算批次大小。
- `workers=8`：可按 CPU 核心数、存储速度与 Windows 稳定性调整；数据加载卡顿或 worker 异常时尝试 `4`。

首次启动时，Ultralytics 会下载默认的 COCO 预训练权重 `yolo11n.pt`。如网络受限，请先下载该文件并使用绝对或相对路径指定：

```powershell
python Project/request_recogniser/train_yolo_v11_p2.py --weights .\weights\yolo11n.pt
```

## 4. 脚本执行内容

每次运行时，脚本会执行以下操作：

1. 递归发现源图片及其标签，并验证类别和归一化框坐标。
2. 用 `seed=42` 打乱样本，并按默认 85%/15% 分成训练集和验证集。
3. 删除并重新生成 `prepared_yolo_dataset`，其中包含 `images/train`、`images/val`、`labels/train`、`labels/val` 与 `dataset.yaml`。由于该目录会被重建，不要在其中手工维护唯一副本的原始数据。
4. 从 [yolov11n-p2.yaml](yolov11n-p2.yaml) 构建模型，再将 `yolo11n.pt` 中形状匹配的 COCO 权重迁移到自定义网络。
5. 在 GPU `0` 上使用 AMP 混合精度训练，终端由 Ultralytics 实时输出损失、Precision、Recall、`mAP50` 和 `mAP50-95`。
6. 训练结束后自动加载 `best.pt` 重新验证，并在日志中打印 `mAP50` 和 `mAP50-95`。

## 5. 训练策略

脚本默认采用面向小目标和遮挡的配置：

| 配置 | 默认值 | 作用 |
| --- | --- | --- |
| AMP | 启用 | 使用 FP16 自动混合精度，降低显存占用并加快 RTX 4070 训练。 |
| `cos_lr` | 启用 | Cosine 学习率衰减。 |
| `lr0` | `0.01` | 初始学习率。 |
| `lrf` | `0.01` | 训练结束时的学习率比例。 |
| `weight_decay` | `0.0005` | 抑制过拟合。 |
| `mosaic` | `1.0` | 拼接多张图片，增加目标尺度和背景组合多样性。 |
| `scale` | `0.5` | 限制随机尺度增强强度，避免小目标经常缩小到不可见。 |
| `copy_paste` | `0.3` | 增加物体组合与遮挡变化。 |
| `close_mosaic` | `10` | 训练最后 10 个 epoch 关闭 Mosaic，使模型适应自然图片分布。 |
| `patience` | `30` | 验证指标连续 30 个 epoch 无改进时早停。 |

可按数据量调整：数据较少且验证指标波动大时，尝试降低 `--lr0` 至 `0.003`；若验证集持续优于训练集，不要立即提高模型规模，应先检查训练增强是否过强以及验证集是否过于简单。

## 6. 常用参数

```powershell
python Project/request_recogniser/train_yolo_v11_p2.py --help
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--data-path` | `collected_images` | 原始图片与标注目录。 |
| `--dataset-dir` | `prepared_yolo_dataset` | 自动生成的标准化数据集目录。该目录会被覆盖。 |
| `--model-yaml` | `yolov11n-p2.yaml` | 模型网络定义文件。 |
| `--weights` | `yolo11n.pt` | COCO 预训练权重路径或名称。 |
| `--epochs` | `150` | 最大训练轮数。 |
| `--batch-size` | `16` | 批次大小，`-1` 使用自动估算。 |
| `--imgsz` | `800` | 训练分辨率，可选 `640`、`800`、`1024`。 |
| `--val-ratio` | `0.15` | 验证集比例。 |
| `--seed` | `42` | 数据拆分和训练随机种子。 |
| `--patience` | `30` | 早停耐心值。 |
| `--allow-unlabeled` | 关闭 | 将缺少 `.txt` 的图片作为无目标背景图。 |
| `--project` | `runs/train` | 训练结果父目录。 |
| `--name` | `yolov11n_p2_small_objects` | 当前实验输出目录名。 |
| `--exist-ok` | 关闭 | 允许复用同名训练目录。 |

## 7. 查看输出与结果

默认训练结果位于：

```text
runs/train/yolov11n_p2_small_objects/
  weights/
    best.pt
    last.pt
  results.csv
  results.png
  confusion_matrix.png
  PR_curve.png
  F1_curve.png
  args.yaml
```

- `weights/best.pt`：验证阶段表现最佳的模型，部署和后续测试优先使用它。
- `weights/last.pt`：最后一个 epoch 的模型，可用于继续训练或排查早停前的状态。
- `results.csv`：逐 epoch 的损失与指标数据。
- `results.png`：损失、Precision、Recall 和 mAP 曲线。
- `confusion_matrix.png`：类别混淆情况；检查笔、橡皮、积木块是否常被彼此误判。
- `PR_curve.png`、`F1_curve.png`：选择部署置信度阈值时的参考曲线。

数据准备和最终评估信息也会写入 `Project/request_recogniser/train_yolo_v11_p2.log`。需要区分多次实验时，指定不同名称：

```powershell
python Project/request_recogniser/train_yolo_v11_p2.py `
  --data-path Project/request_recogniser/yolo_dataset `
  --name fish_eye_1024_trial `
  --imgsz 1024 `
  --batch-size 8
```

## 8. 常见问题

### 找不到图片或标签

确认 `--data-path` 指向的是数据集根目录，而不是单个类别目录。对于自动标注输出，应指向 `yolo_dataset`。默认情况下，缺失标签会中止训练并列出最多 10 个问题图片；补齐标注后重试，或确认其为背景图后使用 `--allow-unlabeled`。

### 标签类别超出范围或坐标非法

本任务仅接受类别 `0` 至 `2`。标签中的中心点应为 $[0, 1]$，宽度和高度应为 $(0, 1]$。常见原因是像素坐标未归一化，或自动标注工具使用了不一致的类别映射。

### CUDA 不可用

训练脚本会拒绝在 CPU 上运行，因为高分辨率 P2 模型训练耗时过长。运行本 README 的 GPU 检查命令，并安装与驱动匹配的 CUDA PyTorch。不要只安装 PyPI 的 CPU 版 PyTorch。

### CUDA out of memory

先降低 `--batch-size`，再将 `--imgsz` 从 `1024` 调整到 `800` 或 `640`。确认没有其他应用占用 GPU 显存，并通过 `nvidia-smi` 检查进程。

### 验证集 mAP 很低或波动很大

优先检查图片和标签质量、三个类别的样本数量是否失衡，以及验证集是否包含真实鱼眼和手部遮挡场景。数据量少时，单次 15% 随机划分可能造成较大波动；保持 `--seed` 不变以便公平比较实验，或在数据量增长后重新训练。

## 9. 部署前检查

在将 `best.pt` 集成进小车系统前，应使用从未参与训练或验证的真实摄像头录像进行测试，并记录每类物体在不同距离、光照、遮挡程度和鱼眼位置下的漏检与误检。部署阈值应根据验证集 PR/F1 曲线和实际误报成本确定，而不是固定沿用默认置信度。