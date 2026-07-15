# ============================================================
# 猫品种识别模型 — Docker 部署
# ============================================================
#
# 构建与运行：
#
#   1. 构建镜像
#      docker build -t cat-breed-api -f deploy/Dockerfile .
#
#   2. 运行容器（PyTorch 后端 + GPU）
#      docker run --gpus all -p 8000:8000 \
#        -v $(pwd)/outputs:/models \
#        cat-breed-api \
#        --model /models/resnet50_cat_xxx/checkpoints/best_model.pth
#
#   3. 运行容器（ONNX 后端 + CPU）
#      docker run -p 8000:8000 \
#        -v $(pwd)/outputs:/models \
#        cat-breed-api \
#        --model /models/resnet50_cat_xxx/resnet50_cat.onnx \
#        --backend onnx
#
#   4. 测试
#      curl http://localhost:8000/health
#      curl -X POST -F "file=@cat.jpg" http://localhost:8000/predict
#
#   5. 访问 API 文档
#      浏览器打开 http://localhost:8000/docs
# ============================================================

# ---- GPU 版本 (PyTorch CUDA) ----
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime AS gpu

WORKDIR /app

# 安装 Python 依赖
COPY pyproject.toml .
RUN pip install --no-cache-dir \
    torchvision \
    pillow \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    pydantic \
    requests \
    && rm pyproject.toml

# 复制部署代码
COPY Baseline/deploy/ /app/deploy/

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

ENTRYPOINT ["python", "-m", "deploy.server"]


# ---- CPU 版本 (ONNX Runtime, 更小更轻量) ----
FROM python:3.11-slim AS cpu

WORKDIR /app

# 安装 ONNX Runtime 和 Web 框架
RUN pip install --no-cache-dir \
    onnxruntime \
    pillow \
    numpy \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    pydantic \
    requests

# 复制部署代码
COPY Baseline/deploy/ /app/deploy/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

ENTRYPOINT ["python", "-m", "deploy.server"]
