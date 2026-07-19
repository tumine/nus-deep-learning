"""模型评估脚本。

评估指标：
- mAP50, mAP50-95（标准目标检测指标）
- 推理延迟（Preprocess + Inference + Postprocess）
- 在不同输入尺寸下的性能对比
- 在不同距离/角度下的检测鲁棒性
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def benchmark_inference(
    model: YOLO,
    image: np.ndarray,
    num_warmup: int = 20,
    num_runs: int = 100,
    imgsz: int = 640,
    half: bool = True,
) -> dict:
    """基准测试推理延迟。

    Args:
        model: YOLO 模型
        image: 输入图片
        num_warmup: 预热次数
        num_runs: 测试运行次数
        imgsz: 输入尺寸
        half: 是否使用 FP16

    Returns:
        延迟统计字典
    """
    # 预热 GPU
    for _ in range(num_warmup):
        _ = model(image, imgsz=imgsz, half=half, verbose=False)

    # 正式测试
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = model(image, imgsz=imgsz, half=half, verbose=False)
        latencies.append((time.perf_counter() - start) * 1000)  # ms

    latencies = np.array(latencies)
    return {
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
    }


def evaluate_multiple_sizes(
    model_path: str,
    test_image: np.ndarray,
    sizes: list[int] = [320, 416, 512, 640],
) -> None:
    """评估不同输入尺寸下的推理延迟。"""
    print("\n=== 不同输入尺寸延迟对比 ===")
    print(f"{'尺寸':>8}  {'平均延迟':>10}  {'P50':>10}  {'P95':>10}  {'P99':>10}")
    print("-" * 55)

    for size in sizes:
        model = YOLO(model_path)
        stats = benchmark_inference(model, test_image, imgsz=size)
        print(
            f"{size:>4}×{size:<4}"
            f"  {stats['mean_ms']:>8.2f}ms"
            f"  {stats['p50_ms']:>8.2f}ms"
            f"  {stats['p95_ms']:>8.2f}ms"
            f"  {stats['p99_ms']:>8.2f}ms"
        )


def evaluate_robustness(
    model: YOLO,
    test_image: np.ndarray,
) -> None:
    """评估模型在不同条件下的鲁棒性。

    模拟教室场景中的各种干扰：
    - JPEG 压缩（模拟网络传输）
    - 运动模糊（模拟摄像头移动）
    - 亮度变化（模拟教室灯光变化）
    - 高斯噪声（模拟传感器噪声）
    """
    print("\n=== 鲁棒性评估 ===")

    # 1. JPEG 压缩鲁棒性
    print("\n--- JPEG 压缩鲁棒性 ---")
    for quality in [90, 70, 50, 30]:
        _, enc = cv2.imencode(".jpg", test_image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        compressed = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        results = model(compressed, verbose=False)
        detections = len(results[0].boxes) if results[0].boxes is not None else 0
        conf = float(results[0].boxes.conf[0]) if detections > 0 else 0
        print(f"  JPEG quality={quality}: {detections} detections, max_conf={conf:.3f}")

    # 2. 模糊鲁棒性
    print("\n--- 运动模糊鲁棒性 ---")
    for kernel in [3, 5, 7, 9]:
        blurred = cv2.GaussianBlur(test_image, (kernel, kernel), 0)
        results = model(blurred, verbose=False)
        detections = len(results[0].boxes) if results[0].boxes is not None else 0
        conf = float(results[0].boxes.conf[0]) if detections > 0 else 0
        print(f"  Blur kernel={kernel}: {detections} detections, max_conf={conf:.3f}")

    # 3. 亮度变化鲁棒性
    print("\n--- 亮度变化鲁棒性 ---")
    for factor in [0.5, 0.75, 1.0, 1.25, 1.5]:
        adjusted = cv2.convertScaleAbs(test_image, alpha=factor, beta=0)
        results = model(adjusted, verbose=False)
        detections = len(results[0].boxes) if results[0].boxes is not None else 0
        conf = float(results[0].boxes.conf[0]) if detections > 0 else 0
        print(f"  Brightness ×{factor:.2f}: {detections} detections, max_conf={conf:.3f}")

    # 4. 噪声鲁棒性
    print("\n--- 高斯噪声鲁棒性 ---")
    for sigma in [5, 10, 15, 20]:
        noise = np.random.normal(0, sigma, test_image.shape).astype(np.uint8)
        noisy = cv2.add(test_image, noise)
        results = model(noisy, verbose=False)
        detections = len(results[0].boxes) if results[0].boxes is not None else 0
        conf = float(results[0].boxes.conf[0]) if detections > 0 else 0
        print(f"  Noise σ={sigma}: {detections} detections, max_conf={conf:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="评估猫识别模型")
    parser.add_argument("--model", default="runs/cat_detection/yolov8n_cat/weights/best.pt", help="模型路径")
    parser.add_argument("--data", default="./datasets/cat_yolo/dataset.yaml", help="数据集 YAML")
    parser.add_argument("--test-image", default=None, help="单张测试图片路径")
    parser.add_argument("--benchmark", action="store_true", help="运行延迟基准测试")
    parser.add_argument("--robustness", action="store_true", help="运行鲁棒性测试")
    args = parser.parse_args()

    model = YOLO(args.model)

    # 标准评估（在验证集上）
    print("=== 标准评估（验证集） ===")
    metrics = model.val(data=args.data, split="val")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")

    # 延迟基准测试
    if args.benchmark:
        print(f"\n=== 推理延迟基准测试（RTX A2000, FP16） ===")
        # 使用一张测试图片
        if args.test_image and Path(args.test_image).exists():
            test_img = cv2.imread(args.test_image)
        else:
            # 创建随机测试图片（640×640）
            test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            print("（使用随机图片进行测试）")

        stats = benchmark_inference(model, test_img)
        print(f"\n推理延迟统计（100 次运行）:")
        print(f"  平均:  {stats['mean_ms']:.2f} ms")
        print(f"  标准差: {stats['std_ms']:.2f} ms")
        print(f"  最小:  {stats['min_ms']:.2f} ms")
        print(f"  最大:  {stats['max_ms']:.2f} ms")
        print(f"  P50:   {stats['p50_ms']:.2f} ms")
        print(f"  P95:   {stats['p95_ms']:.2f} ms")
        print(f"  P99:   {stats['p99_ms']:.2f} ms")

        # 判断是否满足毫秒级要求
        if stats['p95_ms'] < 10:
            print(f"\n✅ P95 延迟 {stats['p95_ms']:.2f}ms < 10ms，满足毫秒级延迟要求！")
        elif stats['p95_ms'] < 50:
            print(f"\n⚠️  P95 延迟 {stats['p95_ms']:.2f}ms < 50ms，在可接受范围内。")
        else:
            print(f"\n❌ P95 延迟 {stats['p95_ms']:.2f}ms > 50ms，需要进一步优化。")

        # 不同输入尺寸对比
        evaluate_multiple_sizes(args.model, test_img)

    # 鲁棒性测试
    if args.robustness and args.test_image:
        test_img = cv2.imread(args.test_image)
        evaluate_robustness(model, test_img)


if __name__ == "__main__":
    main()
