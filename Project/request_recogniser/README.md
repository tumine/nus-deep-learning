# 小物体无标注检测

本目录的主方案是 **YOLO-World 零样本开放词汇检测**：为预训练模型设置英文文字提示词 `pen`、`eraser`、`building block`，即可检测图片中的笔、橡皮和积木块。此流程不需要本地图片标注、训练集划分、`dataset.yaml` 或模型微调。

与原先的 YOLOv11n-P2 监督训练相比，这是一种“预训练模型 + 文字词表绑定”的部署准备过程，并不从本地图片学习。因此它能立刻用于未标注的小车鱼眼画面，但无法自动适应特定镜头的畸变、手部遮挡或非常细小的笔。

| 类别 ID | 默认提示词 | 检测对象 |
| --- | --- | --- |
| `0` | `pen` | 笔 |
| `1` | `eraser` | 橡皮 |
| `2` | `building block` | 积木块 |

## 文件说明

| 文件 | 用途 |
| --- | --- |
| [prepare_yolo_world.py](prepare_yolo_world.py) | 主脚本：绑定提示词、保存 YOLO-World 模型，并可直接预测未标注图片或视频。 |
| [collect_resource_images.py](collect_resource_images.py) | 可选：搜集测试图片或日后微调素材。 |
| [auto_marker.py](auto_marker.py) | 可选：用 Grounding DINO 生成伪标注。 |
| [train_yolo_v11_p2.py](train_yolo_v11_p2.py) | 可选精度升级：只有拥有真实或伪标签时才使用的监督微调脚本。 |
| [yolov11n-p2.yaml](yolov11n-p2.yaml) | 可选监督微调使用的 P2 小目标检测网络定义。 |

## 原理与限制

YOLO-World 是开放词汇检测模型。它在大规模图文和检测数据上预训练，可以将图像区域与文字语义匹配；运行时调用 `set_classes()` 绑定目标文字，不需重新训练。Ultralytics 官方文档说明该模型支持动态自定义提示词而无需重训练：<https://docs.ultralytics.com/models/yolo-world/>。

必须区分以下两件事：

- **无标注零样本检测**：本项目默认方案。无需标签、无需 epoch、无需 loss，也没有本地 `mAP50` 或 `mAP50-95`。
- **监督微调**：若希望模型适应 160 度鱼眼、孩子手部遮挡和极小目标，需要边界框标注或高质量伪标注；这时使用 `train_yolo_v11_p2.py`。

没有任何模型能仅从完全未标注的本地图片中学习“哪个框对应笔、橡皮或积木块”并可靠计算检测精度。零样本模型依赖其已有的预训练知识；如需量化评估，至少应保留一小批人工框标注的独立测试集。

## 1. 环境安装

在仓库根目录安装 CUDA 版 PyTorch 和项目依赖。RTX 4070 必须被 PyTorch 正确识别。

```powershell
# 首次使用时创建环境
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 安装 CUDA PyTorch。项目当前建议 cu126，请按实际驱动环境调整。
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 安装 Ultralytics 与其余项目依赖
pip install -e .
```

检查环境：

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python -c "from ultralytics import YOLOWorld; print('YOLOWorld ready')"
```

第一条命令必须显示 `True` 和 RTX 4070 名称。若 CUDA 不可用，脚本会停止而不会退回 CPU 推理。

## 2. 准备零样本模型

首次运行以下命令。Ultralytics 会下载预训练的 `yolov8s-worldv2.pt`，脚本将三个默认文字提示词写入模型并保存为本地部署文件。此命令不读取图片，不读取 `.txt` 标签，也不运行训练。

```powershell
python Project/request_recogniser/prepare_yolo_world.py
```

输出模型和日志位于：

```text
Project/request_recogniser/zero_shot_output/
  yolov8s_world_school_objects.pt
  prepare_yolo_world.log
```

默认使用 `yolov8s-worldv2.pt`，在精度、显存与速度之间比较适合 RTX 4070。若模型文件已在本地，可通过 `--weights` 指定路径：

```powershell
python Project/request_recogniser/prepare_yolo_world.py `
  --weights .\weights\yolov8s-worldv2.pt
```

## 3. 使用未标注图片或视频检测

`--source` 可以是单张图片、图片目录或视频文件。不需要任何同名 `.txt` 文件。

```powershell
python Project/request_recogniser/prepare_yolo_world.py `
  --source Project/request_recogniser/collected_images `
  --imgsz 800 `
  --conf 0.15
```

预测结果保存在：

```text
Project/request_recogniser/zero_shot_output/predictions/
```

该目录包含带检测框的图片或视频，以及 Ultralytics 输出的预测文本文件。预测文本仅是模型的输出，不是训练所需标注。

### RTX 4070 参数建议

- `--imgsz 800`：默认值，适合 12 GB 显存，保留更多小目标细节。
- `--imgsz 1024`：远距离笔或积木很小时可试用，推理更慢且显存占用更高。
- `--conf 0.15`：零样本场景的起点；漏检多时尝试 `0.10`，误检多时尝试 `0.25` 或更高。
- 默认模型为 `yolov8s-worldv2.pt`。如果速度不足，可改用 `--weights yolov8s-world.pt`；如果效果不足，应先优化提示词并检查真实画面，再考虑有标注微调。

## 4. 调整提示词

提示词顺序定义输出的类别 ID。可以一次传入多个英文提示词；含空格的提示词需要加引号：

```powershell
python Project/request_recogniser/prepare_yolo_world.py `
  --prompts pencil "rubber eraser" "toy building block" `
  --source Project/request_recogniser/collected_images
```

建议以同一组真实小车画面比较不同提示词组合。可尝试：

| 目标 | 可比较的英文提示词 |
| --- | --- |
| 笔 | `pen`、`pencil`、`ballpoint pen`、`marker pen` |
| 橡皮 | `eraser`、`rubber eraser`、`pencil eraser` |
| 积木 | `building block`、`toy building block`、`toy brick` |

避免同时放入语义重叠过强的词，例如既放 `pen` 又放 `pencil` 并把它们当成不同类别；这会使相同物体在多个类别之间竞争。应选择与实际目标类别一一对应的三个提示词。

## 5. 无标注数据准备与人工验收

准备真实小车摄像头的未标注图片或视频即可：

```text
collected_images/
  frame_0001.jpg
  frame_0002.jpg
  classroom_clip.mp4
```

优先收集以下情况并人工查看预测结果：

- 160 度鱼眼画面中央和边缘的物体；
- 手指遮挡不同程度的笔、橡皮和积木；
- 桌面、衣服、玩具等容易产生误检的背景；
- 远距离、逆光、运动模糊、无目标画面。

在没有标注的前提下，验收应记录典型漏检和误检案例，而不是报告 mAP。若模型无法识别具体物体，先比较提示词、分辨率和置信度；这三项不改变模型参数，也不会从本地图片学习。

## 6. 何时进入可选的 YOLOv11-P2 微调

若零样本方案在鱼眼或遮挡条件下效果不够，应切换到已有的监督微调方案。它需要边界框数据，不能满足“完全不提供图片标注”的约束：

1. 用 [auto_marker.py](auto_marker.py) 生成初始伪标注，或人工绘制边界框。
2. 抽查并修正伪标注，尤其是细笔、遮挡和鱼眼边缘目标。
3. 运行 [train_yolo_v11_p2.py](train_yolo_v11_p2.py)。该脚本才会进行 85/15 划分、AMP 训练、早停和 mAP 验证。

该升级路径的价值是把真实相机域知识写入模型；其成本是必须维护标签质量。不要把自动生成的预测文本误认为独立的真实评估标签。

## 7. 常见问题

### `YOLOWorld` 无法导入

升级 Ultralytics：

```powershell
pip install --upgrade ultralytics
```

然后重新执行环境检查。项目中旧版 `ultralytics>=8.0.0` 依赖范围较宽；YOLO-World 功能需要使用包含 `YOLOWorld` 类的版本。

### CUDA 不可用

安装与 NVIDIA 驱动匹配的 CUDA PyTorch，并确认运行的是同一个虚拟环境。可用 `nvidia-smi` 检查显卡和显存占用。

### 模型未检测到细小物体

先使用 `--imgsz 1024` 和较低的 `--conf 0.10` 复测，再替换为更贴近物体外观的英文提示词。若仍然漏检，根本原因通常是零样本模型没有充分见过该摄像头域；需要少量高质量标注并切换到监督微调，而不是增加无标注图片数量。

### 误检较多

提高 `--conf`，缩小提示词语义范围，例如用 `rubber eraser` 替代宽泛的 `eraser`，并用无目标背景画面检查效果。无标注模式无法自动从这些误检中更新权重。