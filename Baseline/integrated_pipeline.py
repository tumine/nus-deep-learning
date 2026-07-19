"""
集成图片接收与识别流水线
========================

整合三个模块的方法，完成以下流程：
  1. **receiver.py** 的 Flask HTTP 服务 ── 接收树莓派等设备发来的图片
  2. **tcp_client.py** 的 print_result() ── 格式化并打印分类结果到控制台
  3. **tcp_server.py 的 CatBreedClassifier** ── 加载模型识别图片

架构：
  ┌──────────────┐     HTTP POST /upload     ┌──────────────────────┐
  │  树莓派/客户端  │ ──────────────────────→ │  Flask (receiver.py)  │
  │  发送 JPEG 图片 │                          │  接收并保存图片          │
  └──────────────┘                          └──────────┬───────────┘
                                                       │ 图片字节
                                                       ▼
                                            ┌──────────────────────┐
                                            │  CatBreedClassifier   │
                                            │  (tcp_server.py 核心)  │
                                            │  模型推理 → 品种分类    │
                                            └──────────┬───────────┘
                                                       │ 推理结果
                                                       ▼
                                            ┌──────────────────────┐
                                            │  print_result()       │
                                            │  (tcp_client.py 工具)  │
                                            │  控制台打印分类结果     │
                                            └──────────────────────┘

用法：
    python integrated_pipeline.py --model best_model.pth
    python integrated_pipeline.py --model best_model.pth --port 5001 --host 0.0.0.0
    python integrated_pipeline.py --model best_model.pth --threshold 0.60
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from flask import Flask, request, jsonify
from PIL import Image
from torchvision import models, transforms

# 将 offline_test 目录加入 sys.path，以便导入 deploy.inference 模块
_script_dir = Path(__file__).resolve().parent
_deploy_parent = _script_dir / "offline_test"
if str(_deploy_parent) not in sys.path:
    sys.path.insert(0, str(_deploy_parent))

from deploy.inference import CatBreedClassifier, BREED_CN

# ============================================================
# 日志
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("integrated-pipeline")

# ============================================================
# ── 来自 receiver.py：Flask 图片接收服务 ──
# ============================================================

# 保存图片的目录
SAVE_DIR = str(Path(__file__).resolve().parent / "img_recv")

app = Flask(__name__)
os.makedirs(SAVE_DIR, exist_ok=True)

# 全局分类器引用（由 main 函数设置）
_classifier: CatBreedClassifier | None = None


@app.route("/upload", methods=["POST"])
def upload_image():
    """[来自 receiver.py] 接收图片并保存，同时触发模型预测。

    接收树莓派发来的图片（multipart/form-data, field name="image"），
    保存后立即调用模型识别，结果打印在控制台。
    """
    try:
        # 1. 检查是否有文件
        if "image" not in request.files:
            return jsonify({"status": "error", "msg": "No image file"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"status": "error", "msg": "Empty filename"}), 400

        # 2. 生成带时间戳的文件名并保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.jpg"
        save_path = os.path.join(SAVE_DIR, filename)
        file.save(save_path)
        logger.info(f"✅ 收到图片并保存: {save_path}")

        # 3. 读取图片字节（供后续模型推理使用）
        image_bytes = _read_image_bytes(save_path)

        # 4. 调用模型识别（来自 tcp_server.py 的 CatBreedClassifier）
        result = _run_inference(image_bytes)

        # 5. 在控制台打印识别结果（来自 tcp_client.py 的 print_result）
        print_result(result, filename)

        return jsonify({
            "status": "success",
            "msg": f"Saved as {filename}",
            "result": result.to_dict(),
        }), 200

    except Exception as e:
        logger.error(f"❌ 处理失败: {e}", exc_info=True)
        return jsonify({"status": "error", "msg": str(e)}), 500


# ============================================================
# ── 来自 tcp_client.py：读取图片 ──
# ============================================================

def _read_image_bytes(image_path: str) -> bytes:
    """[来自 tcp_client.py TcpBreedClient.predict_file()]
    从文件路径读取 JPEG 图片为字节数据。
    """
    with open(image_path, "rb") as f:
        return f.read()


# ============================================================
# ── 来自 tcp_server.py：模型推理（CatBreedClassifier）──
# ============================================================

def _run_inference(image_bytes: bytes):
    """[来自 tcp_server.py TcpBreedServer._handle_client() 中的推理逻辑]
    调用 CatBreedClassifier 对图片字节进行品种分类预测。
    """
    global _classifier
    if _classifier is None:
        raise RuntimeError("分类器未初始化，请先指定 --model 参数")

    return _classifier.predict_from_bytes(image_bytes)


# ============================================================
# ── 来自 tcp_client.py：格式化打印结果 ──
# ============================================================

def print_result(result, image_name: str = ""):
    """[来自 tcp_client.py print_result()]
    格式化打印分类结果到控制台。
    """
    result_dict = result.to_dict()

    if result.is_not_cat:
        print(f"\n{'=' * 55}")
        if image_name:
            print(f"📷 {image_name}")
        print(f"  🚫 Detected as non-cat (confidence below threshold)")
        print(f"  📊 Max probability: {result.confidence * 100:.2f}%")
        print(f"  ⏱️  Inference time: {result.latency_ms:.2f} ms")
        print(f"{'=' * 55}")
        return

    print(f"\n{'=' * 55}")
    if image_name:
        print(f"📷 {image_name}")
    print(f"  🐱 Result: {result.class_name_cn} ({result.class_name})")
    print(f"  📊 Confidence: {result.confidence * 100:.2f}%")
    print(f"  ⏱️  Inference time: {result.latency_ms:.2f} ms")
    print(f"\n  Top-5 Probability Distribution:")
    for item in result_dict.get("top5", []):
        bar = "█" * max(1, int(item["probability"] * 30))
        marker = " ←" if item["rank"] == 1 else ""
        print(f"    {item['rank']}. {item['class_name_cn']:<8} "
              f"{bar} {item['probability'] * 100:.1f}%{marker}")
    print(f"{'=' * 55}")


# ============================================================
# ── CLI 入口 ──
# ============================================================

def main():
    global SAVE_DIR, _classifier
    parser = argparse.ArgumentParser(
        description="集成图片接收与猫品种识别流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 基本用法
  python integrated_pipeline.py --model best_model.pth

  # 自定义端口和置信度阈值
  python integrated_pipeline.py --model best_model.pth --port 5001 --threshold 0.60

  # ONNX 后端
  python integrated_pipeline.py --model resnet50_cat.onnx --backend onnx --port 5000

  # 指定存储目录和监听地址
  python integrated_pipeline.py --model best_model.pth --save-dir ./my_images --host 0.0.0.0
        """,
    )
    parser.add_argument("--model", type=str, required=True,
                        help="模型文件路径（.pth / .pt / .onnx）")
    parser.add_argument("--backend", type=str, default="pytorch",
                        choices=["pytorch", "torchscript", "onnx"],
                        help="推理后端（默认 pytorch）")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Flask 监听地址（默认 0.0.0.0）")
    parser.add_argument("--port", type=int, default=5001,
                        help="Flask 监听端口（默认 5001）")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="图片保存目录")
    parser.add_argument("--threshold", type=float, default=0.50,
                        help="置信度阈值（0-1），低于此值判定为非猫（默认 0.50）")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="计算设备（默认 auto）")
    args = parser.parse_args()

    # 更新全局图片保存目录
    SAVE_DIR = args.save_dir
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 初始化分类器（来自 tcp_server.py 的方法）
    logger.info("=" * 60)
    logger.info("🚀 启动集成图片接收与识别流水线")
    logger.info("=" * 60)
    logger.info(f"  模型路径:     {args.model}")
    logger.info(f"  推理后端:     {args.backend}")
    logger.info(f"  计算设备:     {args.device}")
    logger.info(f"  置信度阈值:   {args.threshold:.2f}")
    logger.info(f"  图片保存目录: {SAVE_DIR}")
    logger.info(f"  HTTP 地址:    http://{args.host}:{args.port}/upload")
    logger.info("=" * 60)

    # 加载模型
    logger.info("加载模型中...")
    _classifier = CatBreedClassifier(
        model_path=args.model,
        backend=args.backend,
        device=args.device,
        confidence_threshold=args.threshold,
    )
    _classifier.load_model()
    logger.info(f"模型加载完成 (device={_classifier.device})")

    # 启动 Flask 服务
    logger.info(f"等待图片上传... (POST http://{args.host}:{args.port}/upload)")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
