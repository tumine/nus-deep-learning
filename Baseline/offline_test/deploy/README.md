# 猫品种识别 — 模型部署模块

基于 ResNet-50 的 5 种猫品种分类模型的**生产级部署方案**。支持两种网络部署方式：

| 部署方式 | 协议 | 适用场景 |
|---------|------|---------|
| **TCP Socket 服务** (`tcp_server.py`) | 自定义二进制协议 | 嵌入式设备（树莓派/Arduino）、低延迟内网通信 |
| **HTTP REST 服务** (`server.py`) | FastAPI / JSON | Web 应用、微服务、跨语言调用 |

---

## 目录结构

```
Baseline/deploy/
├── inference.py        推理引擎（模型加载 + 预处理 + 推理，支持 3 种后端）
├── tcp_server.py       TCP 推理服务器（自定义二进制协议）
├── tcp_client.py       TCP 客户端（发送图片，接收分类结果）
├── server.py           FastAPI HTTP REST 推理服务
├── client.py           HTTP 客户端调用示例
├── demo.py             端到端本地演示（一键启动服务端+客户端）
├── Dockerfile          Docker 容器化（GPU + CPU 两个版本）
├── docker-compose.yml  Docker Compose 一键编排
└── requirements.txt    部署所需 Python 依赖
```

---

## 1. 准备工作

### 1.1 安装依赖

```bash
pip install -r deploy/requirements.txt
```

主要依赖：

| 包 | 用途 |
|----|------|
| `fastapi` / `uvicorn` | HTTP REST 服务（仅 HTTP 部署需要） |
| `torch` / `torchvision` | PyTorch 推理后端 |
| `onnxruntime` | ONNX 推理后端（CPU/GPU 通用，更轻量） |
| `pillow` / `numpy` | 图像处理 |
| `opencv-python` | 摄像头实时模式（客户端可选） |
| `requests` | HTTP 客户端 |

### 1.2 模型文件

训练脚本 `train_cnn.py` 在训练结束时会自动导出以下模型文件：

| 文件 | 格式 | 大小 | 后端 | 说明 |
|------|------|------|------|------|
| `best_model.pth` | PyTorch 检查点 | ~100MB | `pytorch` | 标准训练检查点 |
| `resnet50_cat_scripted.pt` | TorchScript | ~100MB | `torchscript` | 跟踪导出的脚本模型 |
| `resnet50_cat.onnx` | ONNX | ~100MB | `onnx` | **生产推荐**，跨平台 |

模型目录通常在：`outputs/resnet50_cat_YYYYMMDD_HHMMSS/`

```bash
ls outputs/resnet50_cat_20240715_120000/
# checkpoints/best_model.pth
# resnet50_cat_scripted.pt
# resnet50_cat.onnx
# test_image_paths.txt
```

---

## 2. TCP Socket 部署（推荐用于嵌入式场景）

### 2.1 通信协议

```
请求（客户端 → 服务器）:
┌──────────┬─────────────────────┐
│  4 bytes │      N bytes        │
│ 图像长度  │    JPEG 图像数据     │
│ (uint32) │                     │
└──────────┴─────────────────────┘

响应（服务器 → 客户端）:
┌──────────┬─────────────────────┐
│  4 bytes │      N bytes        │
│ JSON长度  │    JSON 分类结果     │
│ (uint32) │                     │
└──────────┴─────────────────────┘
```

**JSON 响应格式：**

```json
{
  "class_id": 0,
  "class_name": "ragdoll",
  "class_name_cn": "布偶猫",
  "confidence": 0.9521,
  "top5": [
    {"rank": 1, "class_name": "ragdoll", "class_name_cn": "布偶猫", "probability": 0.9521},
    {"rank": 2, "class_name": "persian", "class_name_cn": "波斯猫", "probability": 0.0312},
    ...
  ],
  "latency_ms": 3.2,
  "server_time_ms": 1690000000000,
  "request_id": 42
}
```

### 2.2 启动 TCP 服务器

```bash
# GPU 推理（PyTorch 后端，默认）
python -m deploy.tcp_server --model outputs/resnet50_cat_xxx/checkpoints/best_model.pth

# CPU 推理（ONNX 后端，更轻量）
python -m deploy.tcp_server \
    --model outputs/resnet50_cat_xxx/resnet50_cat.onnx \
    --backend onnx

# 自定义端口和最大连接数
python -m deploy.tcp_server --model best_model.pth --port 9527 --max-clients 20
```

服务器启动后输出：

```
============================================================
🐱 猫品种分类 TCP 推理服务
============================================================
  地址:     tcp://0.0.0.0:9000
  模型后端: pytorch
  计算设备: cuda:0
  最大连接: 10
  协议:     长度前缀 + JPEG 图像 → JSON 结果
============================================================
等待客户端连接...
```

### 2.3 客户端调用

```bash
# 单张图片
python deploy/tcp_client.py --image cat.jpg --host localhost --port 9000

# 整文件夹批量
python deploy/tcp_client.py --dir ./test_images/ --output results.json

# 摄像头实时识别（需安装 opencv-python）
python deploy/tcp_client.py --webcam

# 远程服务器
python deploy/tcp_client.py --image cat.jpg --host 192.168.1.100 --port 9000
```

### 2.4 Python 代码调用

```python
from deploy.tcp_client import TcpBreedClient

# ---- 短连接模式：每次请求自动建连/断连 ----
client = TcpBreedClient("localhost", 9000)
result = client.predict_file("cat.jpg")
print(f"{result['class_name_cn']} ({result['confidence']*100:.1f}%)")

# ---- 长连接模式：复用连接，批量发送更高效 ----
with TcpBreedClient("localhost", 9000) as client:
    for img in ["cat1.jpg", "cat2.jpg", "cat3.jpg"]:
        result = client.predict_file(img)
        print(result["class_name_cn"])

# ---- 发送原始字节（如来自摄像头的帧） ----
import cv2
frame = cv2.imread("cat.jpg")
result = client.predict_frame(frame)   # 自动 BGR→RGB→JPEG
```

---

## 3. HTTP REST 部署（推荐用于 Web/微服务）

### 3.1 启动 HTTP 服务器

```bash
python -m deploy.server --model outputs/resnet50_cat_xxx/checkpoints/best_model.pth
```

启动后访问：

- **API 文档（Swagger UI）**：http://localhost:8000/docs
- **健康检查**：http://localhost:8000/health
- **模型信息**：http://localhost:8000/info

### 3.2 API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查（K8s 存活探针） |
| `GET` | `/info` | 模型元信息（类别、输入尺寸等） |
| `POST` | `/predict` | 单张图片分类 |
| `POST` | `/predict/batch` | 批量图片分类（最多 20 张） |

### 3.3 调用示例

**curl：**

```bash
# 健康检查
curl http://localhost:8000/health

# 单张预测
curl -X POST -F "file=@cat.jpg" http://localhost:8000/predict

# 批量预测
curl -X POST -F "files=@cat1.jpg" -F "files=@cat2.jpg" http://localhost:8000/predict/batch
```

**Python：**

```python
from deploy.client import CatBreedClient

client = CatBreedClient("http://localhost:8000")

# 单张预测
result = client.predict_file("cat.jpg")
print(result["class_name_cn"], result["confidence"])

# 批量预测
results = client.predict_batch(["cat1.jpg", "cat2.jpg"])
for r in results["results"]:
    print(r["class_name_cn"])

# 从字节预测
with open("cat.jpg", "rb") as f:
    result = client.predict_bytes(f.read())
```

---

## 4. 端到端本地演示

无需手动启动服务器，一条命令自动完成"启动服务端 → 发送图片 → 打印结果"：

```bash
# 单张图片
python deploy/demo.py --model outputs/resnet50_cat_xxx/checkpoints/best_model.pth --image cat.jpg

# 批量图片
python deploy/demo.py --model best_model.pth --dir ./test_images/

# ONNX 后端
python deploy/demo.py --model resnet50_cat.onnx --backend onnx --image cat.jpg
```

输出示例：

```
=======================================================
🐱 端到端部署演示
=======================================================
  模型:     outputs/.../best_model.pth
  后端:     pytorch
  TCP 端口: 9000
TCP 服务已就绪 ✅

=======================================================
📷 单张推理演示
   图片: cat.jpg
=======================================================
已连接: 127.0.0.1:9000
发送: cat.jpg (45678 字节)

=======================================================
🐱 识别结果: 布偶猫 (ragdoll)
  📊 置信度:   95.21%
  ⏱️  推理耗时: 3.20 ms
  🕐 服务时间: 1690000000000

  Top-5 概率分布:
    1. 布偶猫      ████████████████████ 95.2% ←
    2. 波斯猫      ██ 3.1%
    3. 新加坡猫    █ 1.2%
    4. 斯芬克斯猫  ▌ 0.3%
    5. 兔狲        ▌ 0.1%
=======================================================
  🌐 端到端耗时（含网络）: 12.3ms
  ⚡ 纯推理耗时（服务端）: 3.2ms
  📡 网络开销:           9.1ms
```

---

## 5. Docker 容器化部署

### 5.1 构建镜像

```bash
# GPU 版本（PyTorch + CUDA）
docker build -t cat-breed-api --target gpu -f deploy/Dockerfile .

# CPU 版本（ONNX Runtime，更轻量）
docker build -t cat-breed-api --target cpu -f deploy/Dockerfile .
```

### 5.2 运行容器

```bash
# GPU 推理（TCP 服务）
docker run --gpus all -p 9000:9000 \
    -v $(pwd)/outputs:/models:ro \
    cat-breed-api \
    --model /models/resnet50_cat_xxx/checkpoints/best_model.pth

# CPU 推理（ONNX 后端）
docker run -p 9000:9000 \
    -v $(pwd)/outputs:/models:ro \
    cat-breed-api \
    --model /models/resnet50_cat_xxx/resnet50_cat.onnx \
    --backend onnx
```

### 5.3 Docker Compose 一键编排

```bash
# 启动 GPU 服务
docker compose -f deploy/docker-compose.yml up -d

# 同时启动 CPU 服务（端口 8001）
docker compose -f deploy/docker-compose.yml --profile cpu up -d
```

---

## 6. 推理后端选择

| 后端 | 格式 | 启动速度 | GPU | CPU | 跨平台 | 适用场景 |
|------|------|---------|-----|-----|--------|---------|
| `pytorch` | `.pth` | 慢 | ✅ | ❌ 慢 | ❌ | 开发调试、GPU 服务器 |
| `torchscript` | `.pt` | 中 | ✅ | ❌ | ✅ | 跨平台 PyTorch 部署 |
| `onnx` | `.onnx` | 快 | ✅ | ✅ | ✅ | **生产推荐** |

所有后端使用相同的推理接口 `CatBreedClassifier`，切换只需改 `--backend` 参数和模型路径。

---

## 7. 性能优化建议

| 优化项 | 说明 |
|--------|------|
| **使用 ONNX 后端** | 比 PyTorch 快 10-30%，CPU 推理首选 |
| **长连接复用** | TCP 客户端使用 `with` 语句复用连接，避免频繁握手 |
| **批量推理** | HTTP `/predict/batch` 或 `predict_batch()` 提升吞吐量 |
| **FP16 推理** | CUDA GPU 上自动启用 Tensor Cores 加速 |
| **多进程** | HTTP 服务用 `--workers N` 启动多进程（Linux/macOS） |
| **模型预热** | 服务启动时已加载模型并初始化，首请求无延迟惩罚 |

---

## 8. 常见问题

**Q: 客户端连接超时？**
- 检查服务器是否启动：`curl http://localhost:8000/health`（HTTP）或查看 TCP 日志
- 检查防火墙/端口：`telnet localhost 9000`

**Q: 推理结果一直置信度很低？**
- 确认预处理参数与训练一致（已内置，无需修改）
- 确认输入图片包含猫主体且清晰

**Q: GPU 不可用？**
- 确认安装了 CUDA 版本 PyTorch：`python -c "import torch; print(torch.cuda.is_available())"`
- 或使用 ONNX 后端在 CPU 上推理

**Q: 支持哪些图片格式？**
- JPEG / PNG / WebP / BMP（任意 PIL 可读格式）

**Q: 如何集成到树莓派 + Arduino 小车？**
- 树莓派用 `tcp_client.py` 的模式采集摄像头帧 → TCP 发送给笔记本 GPU 推理 → 接收 JSON 结果 → 通过串口控制 Arduino
- 参考 `Baseline/cat_recognition/deploy/server.py` 中的 UDP+TCP 混合架构
