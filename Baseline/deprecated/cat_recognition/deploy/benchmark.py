"""延迟基准测试工具。

专门针对 RTX A2000 Laptop GPU 的推理延迟进行全面测试。

测试维度：
1. 不同输入尺寸 (320, 416, 512, 640)
2. 不同推理后端 (TensorRT FP16, ONNX GPU, ONNX CPU)
3. 不同精度 (FP32 vs FP16)
4. 批量推理 vs 单帧推理
5. 预热影响
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from infer import CatDetector


def benchmark_input_sizes(
    model_path: str,
    backend: str = "tensorrt",
    num_runs: int = 100,
) -> dict:
    """测试不同输入尺寸的推理延迟。"""
    print(f"\n{'='*60}")
    print(f"输入尺寸延迟对比 ({backend})")
    print(f"{'='*60}")

    sizes = [320, 416, 512, 640]
    results = {}

    for size in sizes:
        detector = CatDetector(
            model_path=model_path,
            backend=backend,
            imgsz=size,
        )

        dummy = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)

        # 预热
        for _ in range(20):
            detector.detect(dummy)

        # 测试
        latencies = []
        for _ in range(num_runs):
            _, elapsed = detector.detect_with_timing(dummy)
            latencies.append(elapsed)

        latencies = np.array(latencies)
        results[size] = {
            "mean_ms": float(np.mean(latencies)),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
        }

        print(
            f"  {size}×{size}: "
            f"mean={results[size]['mean_ms']:.2f}ms, "
            f"P50={results[size]['p50_ms']:.2f}ms, "
            f"P95={results[size]['p95_ms']:.2f}ms"
        )

    return results


def benchmark_backends(
    model_paths: dict[str, str],
    num_runs: int = 100,
) -> dict:
    """测试不同推理后端的延迟对比。"""
    print(f"\n{'='*60}")
    print(f"推理后端延迟对比 (640×640)")
    print(f"{'='*60}")

    results = {}

    for backend, path in model_paths.items():
        if not Path(path).exists():
            print(f"  {backend}: 模型不存在 ({path})，跳过")
            continue

        try:
            detector = CatDetector(
                model_path=path,
                backend=backend,
            )

            dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

            # 预热
            for _ in range(20):
                detector.detect(dummy)

            # 测试
            latencies = []
            for _ in range(num_runs):
                _, elapsed = detector.detect_with_timing(dummy)
                latencies.append(elapsed)

            latencies = np.array(latencies)
            results[backend] = {
                "mean_ms": float(np.mean(latencies)),
                "p50_ms": float(np.percentile(latencies, 50)),
                "p95_ms": float(np.percentile(latencies, 95)),
                "p99_ms": float(np.percentile(latencies, 99)),
                "min_ms": float(np.min(latencies)),
                "max_ms": float(np.max(latencies)),
            }

            print(
                f"  {backend:>12}: "
                f"mean={results[backend]['mean_ms']:.2f}ms, "
                f"P95={results[backend]['p95_ms']:.2f}ms, "
                f"min/max={results[backend]['min_ms']:.1f}/{results[backend]['max_ms']:.1f}ms"
            )

        except Exception as e:
            print(f"  {backend}: 错误 - {e}")

    return results


def benchmark_precision(
    model_path: str,
    num_runs: int = 100,
) -> dict:
    """测试 FP32 vs FP16 精度下的延迟对比。

    注意：需要 TensorRT 引擎或支持 half 参数的模型。
    """
    print(f"\n{'='*60}")
    print(f"精度延迟对比 (640×640)")
    print(f"{'='*60}")

    results = {}
    dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # FP32（默认）
    try:
        from ultralytics import YOLO
        model_fp32 = YOLO(model_path)

        for _ in range(20):
            model_fp32(dummy, half=False, verbose=False)

        latencies = []
        for _ in range(num_runs):
            start = time.perf_counter()
            model_fp32(dummy, half=False, verbose=False)
            latencies.append((time.perf_counter() - start) * 1000)

        latencies = np.array(latencies)
        results["FP32"] = {
            "mean_ms": float(np.mean(latencies)),
            "p95_ms": float(np.percentile(latencies, 95)),
        }
        print(f"  FP32: mean={results['FP32']['mean_ms']:.2f}ms, P95={results['FP32']['p95_ms']:.2f}ms")
    except Exception as e:
        print(f"  FP32: 错误 - {e}")

    # FP16
    try:
        from ultralytics import YOLO
        model_fp16 = YOLO(model_path)

        for _ in range(20):
            model_fp16(dummy, half=True, verbose=False)

        latencies = []
        for _ in range(num_runs):
            start = time.perf_counter()
            model_fp16(dummy, half=True, verbose=False)
            latencies.append((time.perf_counter() - start) * 1000)

        latencies = np.array(latencies)
        results["FP16"] = {
            "mean_ms": float(np.mean(latencies)),
            "p95_ms": float(np.percentile(latencies, 95)),
        }
        print(f"  FP16: mean={results['FP16']['mean_ms']:.2f}ms, P95={results['FP16']['p95_ms']:.2f}ms")

        # 加速比
        if "FP32" in results:
            speedup = results["FP32"]["mean_ms"] / results["FP16"]["mean_ms"]
            print(f"  FP16 加速比: {speedup:.2f}×")
    except Exception as e:
        print(f"  FP16: 错误 - {e}")

    return results


def benchmark_real_image(
    model_path: str,
    image_path: str,
    backend: str = "tensorrt",
    num_runs: int = 100,
) -> dict:
    """使用真实图片测试推理延迟。"""
    print(f"\n{'='*60}")
    print(f"真实图片延迟测试")
    print(f"{'='*60}")

    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图片: {image_path}")
        return {}

    h, w = image.shape[:2]
    print(f"图片尺寸: {w}×{h}")

    detector = CatDetector(
        model_path=model_path,
        backend=backend,
    )

    # 预热
    for _ in range(20):
        detector.detect(image)

    # 测试
    latencies = []
    detection_counts = []
    for _ in range(num_runs):
        detections, elapsed = detector.detect_with_timing(image)
        latencies.append(elapsed)
        detection_counts.append(len(detections))

    latencies = np.array(latencies)
    results = {
        "image_size": f"{w}×{h}",
        "mean_ms": float(np.mean(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "avg_detections": float(np.mean(detection_counts)),
    }

    print(f"  平均延迟: {results['mean_ms']:.2f} ms")
    print(f"  P95延迟:  {results['p95_ms']:.2f} ms")
    print(f"  平均检测数: {results['avg_detections']:.1f}")

    return results


def generate_report(all_results: dict, output_path: Optional[str] = None) -> None:
    """生成延迟测试报告。"""
    print(f"\n{'='*60}")
    print("延迟测试报告")
    print(f"{'='*60}")

    # 汇总判断
    has_tensorrt = "tensorrt" in all_results.get("backends", {})
    has_onnx = "onnx" in all_results.get("backends", {})

    print(f"\n推荐方案:")
    if has_tensorrt:
        trt_p95 = all_results["backends"]["tensorrt"]["p95_ms"]
        print(f"  ✅ TensorRT FP16: P95={trt_p95:.2f}ms")
        if trt_p95 < 5:
            print(f"     → 极低延迟，完全满足毫秒级要求")
        elif trt_p95 < 10:
            print(f"     → 满足毫秒级延迟要求")
    if has_onnx:
        onnx_p95 = all_results["backends"]["onnx"]["p95_ms"]
        print(f"  ⚠️  ONNX GPU: P95={onnx_p95:.2f}ms")

    # 保存报告
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n报告已保存: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="延迟基准测试")
    parser.add_argument("--model", default="./models/cat_yolov8n.engine", help="主模型路径")
    parser.add_argument("--onnx-model", default="./models/cat_yolov8n.onnx", help="ONNX 模型路径")
    parser.add_argument("--pt-model", default="runs/cat_detection/yolov8n_cat/weights/best.pt", help="PyTorch 模型路径")
    parser.add_argument("--image", help="真实测试图片路径")
    parser.add_argument("--output", default="benchmark_report.json", help="报告输出路径")
    parser.add_argument("--runs", type=int, default=100, help="测试运行次数")
    args = parser.parse_args()

    all_results = {}

    # 1. 输入尺寸对比
    if Path(args.model).exists():
        all_results["input_sizes"] = benchmark_input_sizes(
            args.model,
            backend="tensorrt",
            num_runs=args.runs,
        )

    # 2. 推理后端对比
    model_paths = {}
    if Path(args.model).exists():
        model_paths["tensorrt"] = args.model
    if Path(args.onnx_model).exists():
        model_paths["onnx"] = args.onnx_model
    if Path(args.pt_model).exists():
        model_paths["ultralytics"] = args.pt_model

    if model_paths:
        all_results["backends"] = benchmark_backends(model_paths, num_runs=args.runs)

    # 3. 精度对比
    if Path(args.pt_model).exists():
        all_results["precision"] = benchmark_precision(args.pt_model, num_runs=args.runs)

    # 4. 真实图片测试
    if args.image and Path(args.model).exists():
        all_results["real_image"] = benchmark_real_image(
            args.model,
            args.image,
            backend="tensorrt",
            num_runs=args.runs,
        )

    # 生成报告
    generate_report(all_results, args.output)


if __name__ == "__main__":
    main()
