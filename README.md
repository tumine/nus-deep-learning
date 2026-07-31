# Kinderguarder — 2026 NUS SoC SWS3009 Robotics & Deep Learning

本仓库是 2026 年新加坡国立大学计算机学院暑期学校（NUS SoC SWS 2026）**SWS3009** 课程的代码库，目标为开发一台**智能教室助手机器人（Kinderguarder）**。机器人在幼儿园教室中自主巡逻，通过计算机视觉检测举手求助的儿童，通过视觉或语音识别请求，并自动递送物品或通知教师协助。

---

## 目录结构

```
nus-deep-learning/
├── Baseline/                 # 基线任务：猫品种分类
├── card_detecter/            # ArUco 请求卡片检测模块
├── Notes/                    # 课程讲义笔记
├── Project/                  # 主项目：Kinderguarder 完整系统
└── pyproject.toml            # 项目依赖配置
```

---

## 各目录功能

### 1. `Baseline/` — 猫品种分类（基线任务）

课程第一阶段的"寻宝"任务。机器人移动过程中对猫图片进行品种分类，共五类：

- 兔狲 (Pallas's cat)、波斯猫 (Persian)、布偶猫 (Ragdoll)、新加坡猫 (Singapura)、斯芬克斯猫 (Sphynx)

| 子目录/文件 | 功能 |
|---|---|
| `train_cnn_v2.py` | 训练脚本，使用 **ConvNeXt-Tiny/Small** 或 EfficientNetV2 进行迁移学习（特征提取 + 微调），384×384 分辨率，MixUp/CutMix 增强，Cosine Warmup 调度器。适配 RTX 4070 (12GB)。 |
| `integrated_pipeline.py` | Flask HTTP 服务，从树莓派接收图片，调用分类器推理并输出格式化结果。 |
| `image_collector/` | Selenium 爬虫，从 Bing Images 收集各猫品种约 350 张图片。 |
| `offline_test/deploy/` | **生产部署**：多引擎推理（PyTorch/TorchScript/ONNX），TCP / FastAPI HTTP 通信，Docker 容器化。 |
| `deprecated/` | 旧版训练脚本（ResNet-50、ResNet-18）和早期模块化流水线。 |
| `network_choice.md` | 选用 ConvNeXt 的理由（深度可分离卷积、细粒度分类优势）。 |

---

### 2. `Notes/` — 课程讲义

包含课程 4 讲的笔记：`Lec1.md` ~ `Lec4.md`，涵盖深度学习基础、计算机视觉、机器人系统等内容。

---

### 3. `Project/` — 主项目：Kinderguarder 完整系统

主项目的核心代码和资源，整合了所有子系统。

| 子目录/文件 | 功能 |
|---|---|
| `hand_card_state_machine/` | **举手-卡片状态机**（核心控制逻辑）：管理机器人从检测举手 → 识别卡片 → 递送物品的完整状态流转。含 YOLO 姿态估计模型、ArUco 检测以及教师端网页。 |
| `request_recogniser/` | **请求识别模块**：训练 YOLOv11 模型进行举手姿态检测，包含 ~1,241 张训练图片和标注数据。 |
| `car_control_tracking/` | **小车运动控制与巡线**：基于 Arduino 的四轮差速驱动控制，含红外循线传感器循线逻辑。 |
| `camera_binding/` | **摄像头绑定与标定**：前后摄像头的配置和参数标定工具。 |
| `cry_detect/` | **哭声检测模块**：语音识别，检测儿童哭声并通过树莓派向教师端发送警报。 |
| `transfer_template/` | TCP 网络通信模板。 |
| `asset/` | 项目资源文件（图片、模型等）。 |
| `doc/` | 项目文档和设计说明。 |
| `prompts/` | AI 辅助开发提示词。 |
| `ref/` | 参考资料。 |
| `deprecated/` | 废弃的旧版代码。 |
| `ws_car_control_V1_4.py` | 小车控制 Websocket 服务端。 |
| `car_control_tracking.ino` | Arduino 端电机控制固件。 |

---

## 硬件架构

| 组件 | 说明 |
|---|---|
| **Raspberry Pi 5** | 主控制器，运行深度学习推理和系统逻辑 |
| **Arduino Mega 2560** | 从机，负责电机驱动和传感器读取 |
| **前置摄像头** | 举手姿态检测 + 卡片识别 |
| **下方红外巡线传感器** | 固定路线巡游（循线、交叉口识别） |
| **超声波传感器** | 前向避障 |

---

## 环境配置

```bash
# Python >= 3.11
pip install -e .                           # 安装基础依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126  # GPU 训练
pip install -e ".[tensorrt]"               # TensorRT 推理加速（可选）
pip install -e ".[dev]"                    # 开发工具（pytest, black, ruff）
```
