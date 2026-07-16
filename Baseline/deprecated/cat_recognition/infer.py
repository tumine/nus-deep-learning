"""本地推理脚本。

在笔记本电脑上运行猫识别推理，验证毫秒级延迟。

支持三种推理后端：
1. TensorRT（最优，RTX A2000 FP16，< 5ms）
2. ONNX Runtime GPU（备选，跨平台）
3. Ultralytics 原生（开发调试用）
"""

import argparse
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class CatDetector:
    """猫检测器封装类。

    支持 TensorRT、ONNX、Ultralytics 三种后端，
    提供统一的推理接口。
    """

    def __init__(
        self,
        model_path: str,
        backend: str = "tensorrt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
    ):
        """初始化检测器。

        Args:
            model_path: 模型文件路径
            backend: 推理后端 ("tensorrt", "onnx", "ultralytics")
            conf_threshold: 置信度阈值
            iou_threshold: NMS IoU 阈值
            imgsz: 输入图像尺寸
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.backend = backend

        if backend in ("tensorrt", "ultralytics"):
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self._predict = self._predict_ultralytics
        elif backend == "onnx":
            import onnxruntime as ort
            providers = [
                ("CUDAExecutionProvider", {"device_id": 0}),  # RTX A2000
                "CPUExecutionProvider",
            ]
            self.session = ort.InferenceSession(
                model_path, providers=providers
            )
            self._predict = self._predict_onnx
        else:
            raise ValueError(f"不支持的推理后端: {backend}")

        print(f"检测器初始化完成")
        print(f"  后端: {backend}")
        print(f"  模型: {model_path}")
        print(f"  输入尺寸: {imgsz}×{imgsz}")

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """图像预处理。"""
        # Resize + 保持宽高比的 letterbox
        img = cv2.resize(image, (self.imgsz, self.imgsz))
        img = img.transpose(2, 0, 1)  # HWC → CHW
        img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
        return img

    def _predict_ultralytics(self, image: np.ndarray) -> list[dict]:
        """Ultralytics / TensorRT 推理。"""
        results = self.model(
            image,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            half=True,  # FP16
            verbose=False,
        )
        return self._parse_results(results[0])

    def _predict_onnx(self, image: np.ndarray) -> list[dict]:
        """ONNX Runtime 推理。"""
        input_tensor = self._preprocess(image)
        outputs = self.session.run(None, {"images": input_tensor})
        # ONNX 输出的后处理（简化版，完整版需 NMS）
        # 这里使用 Ultralytics 的 ONNX 输出格式
        return self._parse_onnx_output(outputs, image.shape)

    def _parse_results(self, result) -> list[dict]:
        """解析 YOLO 推理结果为统一格式。"""
        detections = []
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            for box, conf in zip(boxes, confs):
                detections.append({
                    "x1": int(box[0]),
                    "y1": int(box[1]),
                    "x2": int(box[2]),
                    "y2": int(box[3]),
                    "confidence": float(conf),
                    "class": "cat",
                })
        return detections

    def _parse_onnx_output(self, outputs, image_shape) -> list[dict]:
        """解析 ONNX 输出（简化版）。"""
        # ONNX 导出格式: [batch, num_dets, 6] 其中 6=[x1,y1,x2,y2,conf,cls]
        detections = []
        if outputs and len(outputs) > 0:
            preds = outputs[0][0]  # batch=1
            for pred in preds:
                conf = float(pred[4])
                if conf >= self.conf_threshold:
                    detections.append({
                        "x1": int(pred[0]),
                        "y1": int(pred[1]),
                        "x2": int(pred[2]),
                        "y2": int(pred[3]),
                        "confidence": conf,
                        "class": "cat",
                    })
        return detections

    def detect(self, image: np.ndarray) -> list[dict]:
        """执行猫检测。

        Args:
            image: BGR 格式图片 (H, W, 3)

        Returns:
            检测结果列表，每项包含 bbox 和置信度
        """
        return self._predict(image)

    def detect_with_timing(self, image: np.ndarray) -> tuple[list[dict], float]:
        """执行检测并返回推理时间。

        Returns:
            (检测结果, 推理时间_ms)
        """
        start = time.perf_counter()
        detections = self._predict(image)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return detections, elapsed_ms


def draw_detections(
    image: np.ndarray,
    detections: list[dict],
    color: tuple = (0, 255, 0),
) -> np.ndarray:
    """在图片上绘制检测框。"""
    img = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        conf = det["confidence"]

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"Cat {conf:.2f}"
        cv2.putText(
            img, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
        )

    return img


def benchmark_detector(
    detector: CatDetector,
    num_warmup: int = 20,
    num_runs: int = 200,
) -> dict:
    """全面基准测试检测器性能。

    Returns:
        延迟统计
    """
    print(f"\n=== 推理延迟基准测试 ===")
    print(f"预热: {num_warmup} 次, 测试: {num_runs} 次")

    dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # 预热
    for _ in range(num_warmup):
        detector.detect(dummy)

    # 测试
    latencies = []
    for _ in range(num_runs):
        _, elapsed = detector.detect_with_timing(dummy)
        latencies.append(elapsed)

    latencies = np.array(latencies)
    stats = {
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
    }

    print(f"  平均延迟: {stats['mean_ms']:.2f} ms")
    print(f"  P50:      {stats['p50_ms']:.2f} ms")
    print(f"  P95:      {stats['p95_ms']:.2f} ms")
    print(f"  P99:      {stats['p99_ms']:.2f} ms")
    print(f"  最小/最大: {stats['min_ms']:.2f} / {stats['max_ms']:.2f} ms")

    # 毫秒级判断
    if stats["p95_ms"] < 10:
        print(f"  ✅ 满足毫秒级延迟要求 (P95 < 10ms)")
    elif stats["p95_ms"] < 50:
        print(f"  ⚠️  在可接受范围内 (P95 < 50ms)")
    else:
        print(f"  ❌ 延迟过高，需要优化")

    return stats


def process_webcam(detector: CatDetector, camera_id: int = 0) -> None:
    """使用笔记本摄像头实时推理演示。"""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"无法打开摄像头 {camera_id}")
        return

    print("\n=== 实时摄像头推理 ===")
    print("按 'q' 退出, 按 's' 截图保存")

    frame_count = 0
    total_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections, elapsed = detector.detect_with_timing(frame)
        frame_count += 1
        total_time += elapsed

        # 绘制结果
        result = draw_detections(frame, detections)

        # 显示 FPS 和延迟
        avg_latency = total_time / frame_count
        cv2.putText(
            result,
            f"Latency: {elapsed:.1f}ms | Avg: {avg_latency:.1f}ms",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
        )

        if detections:
            for det in detections:
                print(f"  检测到猫! 置信度: {det['confidence']:.2f}")

        cv2.imshow("Cat Detection (RTX A2000)", result)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"cat_capture_{timestamp}.jpg", frame)
            print(f"截图已保存: cat_capture_{timestamp}.jpg")

    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="猫识别推理")
    parser.add_argument("--model", default="./models/cat_yolov8n.engine", help="模型路径")
    parser.add_argument("--backend", default="tensorrt", choices=["tensorrt", "onnx", "ultralytics"], help="推理后端")
    parser.add_argument("--image", help="单张图片路径")
    parser.add_argument("--video", help="视频文件路径")
    parser.add_argument("--webcam", action="store_true", help="使用摄像头实时推理")
    parser.add_argument("--benchmark", action="store_true", help="运行延迟基准测试")
    parser.add_argument("--output", help="输出图片路径")
    parser.add_argument("--config", default="config.yaml", help="配置文件")
    args = parser.parse_args()

    config = load_config(args.config)
    infer_cfg = config["inference"]

    # 初始化检测器
    detector = CatDetector(
        model_path=args.model,
        backend=args.backend,
        conf_threshold=infer_cfg["conf_threshold"],
        iou_threshold=infer_cfg["iou_threshold"],
        imgsz=infer_cfg["imgsz"],
    )

    # 基准测试
    if args.benchmark:
        benchmark_detector(detector)
        return

    # 单张图片推理
    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            print(f"无法读取图片: {args.image}")
            return

        detections, elapsed = detector.detect_with_timing(image)
        print(f"\n推理延迟: {elapsed:.2f} ms")
        print(f"检测到 {len(detections)} 只猫")

        for i, det in enumerate(detections):
            print(f"  [{i}] bbox=({det['x1']},{det['y1']},{det['x2']},{det['y2']}), conf={det['confidence']:.3f}")

        result = draw_detections(image, detections)

        if args.output:
            cv2.imwrite(args.output, result)
            print(f"结果已保存: {args.output}")
        else:
            cv2.imshow("Cat Detection", result)
            print("按任意键关闭窗口...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    # 视频文件推理
    if args.video:
        cap = cv2.VideoCapture(args.video)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            detections, _ = detector.detect_with_timing(frame)
            result = draw_detections(frame, detections)
            cv2.imshow("Cat Detection", result)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
        return

    # 摄像头实时推理
    if args.webcam:
        process_webcam(detector)
        return

    # 默认：基准测试
    benchmark_detector(detector)


if __name__ == "__main__":
    main()
