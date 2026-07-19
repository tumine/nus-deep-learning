# Cat Recognition Model for Laptop Deployment

## 概述

在教室内巡游小车的视觉系统中，训练一个可运行在笔记本电脑（Intel UHD Graphics + NVIDIA RTX A2000 Laptop GPU）上的小猫识别模型，满足毫秒级推理延迟。

## 硬件约束分析

| 硬件 | 规格 | 影响 |
|------|------|------|
| **NVIDIA RTX A2000 Laptop GPU** | 4GB VRAM, Ampere, FP16/TF32 加速 | 可运行量化后的轻量模型（<2GB VRAM 占用） |
| **Intel UHD Graphics** | 集成显卡 | 仅用于显示，不参与推理 |
| **CPU** | 典型笔记本 i7/i9 | 预处理/后处理 |
| **内存** | 典型 16-32GB | 足够缓存视频帧 |

## 毫秒级延迟方案

针对 RTX A2000（4GB VRAM）的推理延迟预估：

| 模型 | 输入尺寸 | 参数量 | FP16 推理延迟 | VRAM 占用 |
|------|---------|--------|-------------|----------|
| YOLOv8n | 640×640 | 3.2M | **~3-5ms** | ~500MB |
| YOLOv8s | 640×640 | 11.2M | **~5-8ms** | ~800MB |
| YOLOv8m | 640×640 | 25.9M | **~10-15ms** | ~1.2GB |
| MobileNetV3-SSD | 320×320 | 5.4M | **~2-4ms** | ~400MB |

**推荐方案**：YOLOv8n（3.2M 参数），在 RTX A2000 上 FP16 推理约 3-5ms/帧，完全满足毫秒级延迟要求。

## 项目结构

```
cat_recognition/
├── README.md                    # 本文件
├── pyproject.toml               # 项目依赖
├── config.yaml                  # 训练与推理配置
├── dataset/
│   ├── prepare_dataset.py       # 数据集准备脚本
│   └── augment.py               # 数据增强
├── train.py                     # YOLOv8n 训练脚本
├── evaluate.py                  # 模型评估脚本
├── export.py                    # 模型导出（ONNX/TensorRT）
├── infer.py                     # 本地推理脚本（毫秒级延迟验证）
└── deploy/
    ├── server.py                # TCP 推理服务（接收树莓派帧，返回结果）
    └── benchmark.py             # 延迟基准测试
```

## 快速开始

```bash
# 1. 安装依赖
pip install -e .

# 2. 准备数据集（使用 COCO cat 子集 + 自定义数据）
python dataset/prepare_dataset.py

# 3. 训练模型
python train.py

# 4. 评估模型
python evaluate.py

# 5. 导出为 ONNX / TensorRT
python export.py

# 6. 推理延迟测试
python infer.py --image test.jpg --benchmark

# 7. 启动推理服务（接收树莓派视频帧）
python deploy/server.py --port 9001
```

## 架构设计

```
[Arduino 小车 + 摄像头]
        │
        │ (JPEG 视频帧, UDP/TCP)
        ▼
[树莓派] ──转发──► [笔记本 (RTX A2000)]
                         │
                         │ YOLOv8n 推理 (3-5ms)
                         ▼
                   检测结果 (bbox + 置信度)
                         │
                         │ TCP
                         ▼
                    [树莓派] ──► [Arduino 小车]
```
