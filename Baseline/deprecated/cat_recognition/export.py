"""模型导出脚本。

将训练好的 YOLOv8n 模型导出为多种格式，针对 RTX A2000 进行优化。

导出格式：
1. ONNX — 通用跨平台推理
2. TensorRT — NVIDIA GPU 最优推理引擎（FP16）
3. OpenVINO — Intel UHD Graphics 备选方案

在 RTX A2000 上，TensorRT FP16 可达到最低推理延迟。
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def export_onnx(model: YOLO, output_dir: Path, config: dict) -> Path:
    """导出 ONNX 格式。

    ONNX 优点：跨平台兼容、可用于 CPU 推理、生态成熟。
    """
    onnx_cfg = config["export"]["onnx"]
    output_path = output_dir / "cat_yolov8n.onnx"

    print(f"\n=== 导出 ONNX ===")
    model.export(
        format="onnx",
        imgsz=640,
        opset=onnx_cfg["opset"],
        dynamic=onnx_cfg["dynamic"],
        simplify=onnx_cfg["simplify"],
        half=True,  # FP16
    )

    print(f"ONNX 模型已导出: {output_path}")
    return output_path


def export_tensorrt(model: YOLO, output_dir: Path, config: dict) -> Path:
    """导出 TensorRT 引擎。

    TensorRT 优点：NVIDIA GPU 上最低延迟、FP16 原生加速。
    在 RTX A2000 上预期延迟 < 3ms。

    注意：TensorRT 引擎与 GPU 型号绑定，不能在跨 GPU 使用。
    """
    trt_cfg = config["export"]["tensorrt"]
    output_path = output_dir / "cat_yolov8n.engine"

    print(f"\n=== 导出 TensorRT (FP16) ===")
    print(f"目标 GPU: NVIDIA RTX A2000 Laptop GPU")

    try:
        model.export(
            format="engine",
            imgsz=640,
            half=trt_cfg["fp16"],
            workspace=trt_cfg["workspace"],
            batch=trt_cfg["batch_size"],
            device="cuda:0",
        )
        print(f"TensorRT 引擎已导出: {output_path}")
    except Exception as e:
        print(f"TensorRT 导出失败: {e}")
        print("请确保已安装 TensorRT 和 onnx-graphsurgeon:")
        print("  pip install tensorrt onnx-graphsurgeon")
        print("将回退到 ONNX 推理。")

    return output_path


def export_openvino(model: YOLO, output_dir: Path, config: dict) -> Path:
    """导出 OpenVINO IR 格式。

    作为备选方案，可在 Intel UHD Graphics 上进行推理。
    延迟会比 GPU 方案高，但在 GPU 不可用时提供 fallback。
    """
    output_dir = output_dir / "openvino"

    print(f"\n=== 导出 OpenVINO (备选) ===")
    try:
        model.export(
            format="openvino",
            imgsz=640,
            half=True,
        )
        print(f"OpenVINO 模型已导出: {output_dir}")
    except Exception as e:
        print(f"OpenVINO 导出失败（非必需）: {e}")

    return output_dir


def verify_export(model_path: Path) -> None:
    """验证导出的模型可以正常加载推理。"""
    print(f"\n=== 验证导出模型: {model_path} ===")
    import numpy as np

    model = YOLO(str(model_path))
    dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    results = model(dummy, verbose=False)

    detections = len(results[0].boxes) if results[0].boxes is not None else 0
    print(f"推理成功！检测到 {detections} 个目标（随机噪声输入，预期为 0 或极少）")


def print_usage_guide(output_dir: Path) -> None:
    """打印各格式的使用指南。"""
    print(f"\n{'='*60}")
    print("模型使用指南")
    print(f"{'='*60}")

    onnx_path = output_dir / "cat_yolov8n.onnx"
    engine_path = output_dir / "cat_yolov8n.engine"

    print(f"""
1. TensorRT 推理（推荐，RTX A2000 最低延迟）:
   from ultralytics import YOLO
   model = YOLO("{engine_path}")
   results = model("image.jpg")

2. ONNX Runtime 推理（跨平台兼容）:
   import onnxruntime as ort
   session = ort.InferenceSession("{onnx_path}")
   # 详见 infer.py

3. OpenVINO 推理（Intel UHD Graphics fallback）:
   from ultralytics import YOLO
   model = YOLO("{output_dir / 'openvino'}")
   results = model("image.jpg")
""")


def main() -> None:
    parser = argparse.ArgumentParser(description="导出猫识别模型")
    parser.add_argument("--model", default="runs/cat_detection/yolov8n_cat/weights/best.pt", help="训练好的模型路径")
    parser.add_argument("--output", default="./models", help="导出目录")
    parser.add_argument("--formats", nargs="+", default=["onnx", "tensorrt"], help="导出格式")
    parser.add_argument("--config", default="config.yaml", help="配置文件")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载训练好的模型
    if not Path(args.model).exists():
        print(f"错误: 找不到模型文件 {args.model}")
        print("请先运行 train.py 训练模型，或下载预训练权重。")
        sys.exit(1)

    print(f"加载模型: {args.model}")
    model = YOLO(args.model)

    # 导出各格式
    for fmt in args.formats:
        fmt = fmt.lower()
        if fmt == "onnx":
            onnx_path = export_onnx(model, output_dir, config)
            verify_export(onnx_path)
        elif fmt in ("tensorrt", "engine", "trt"):
            engine_path = export_tensorrt(model, output_dir, config)
            if engine_path.exists():
                verify_export(engine_path)
        elif fmt == "openvino":
            export_openvino(model, output_dir, config)
        else:
            print(f"不支持的格式: {fmt}")

    print_usage_guide(output_dir)
    print(f"\n所有模型已导出到: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
