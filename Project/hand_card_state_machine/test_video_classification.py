"""
test_video_classification.py

通过电脑摄像头实时流式检测 YOLO 模型的视频分类/检测能力。

与 CardDetector（单次截图分类 + 会话锁定）不同，本脚本：
- 对摄像头每一帧持续进行推理，不做会话锁定
- 检测模式（有 boxes）：在画面中画出目标边界框和标签
- 分类模式（只有 probs）：在画面中央显示 top-N 分类结果
- 显示实时 FPS，验证流式推理性能

用法：
    python test_video_classification.py

按键：
    Q / ESC  → 退出
    M        → 切换 detection / classification 模式
    D        → 切换调试信息
    S        → 截图保存
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# 关键：在 import ultralytics 之前保存原始 OpenCV GUI 函数
# ultralytics 会 monkey-patch cv2.imshow / cv2.namedWindow 等，
# 其 unicode_escape 编码在 Windows 上会导致 OpenCV GUI 崩溃。
# 我们在导入后立即恢复原始函数。
# ──────────────────────────────────────────────────────────────
import cv2

_ORIG_IMSHOW = cv2.imshow
_ORIG_WAITKEY = cv2.waitKey
_ORIG_DESTROY_ALL = cv2.destroyAllWindows


def _check_opencv_gui_available():
    """检测 OpenCV 是否具备 GUI 支持（非 headless 版本）。"""
    import numpy as np
    try:
        # 仅检测函数是否存在 — headless 版中 imshow 存在但运行时失败
        dummy = np.zeros((10, 10, 3), dtype=np.uint8)
        cv2.imshow("_probe_", dummy)
        cv2.destroyWindow("_probe_")
        return True
    except cv2.error as e:
        if "not implemented" in str(e).lower():
            return False
        return True  # 其他错误可能是窗口系统的暂时问题
    except Exception:
        return False


import torch
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────
# 恢复原始 OpenCV GUI 函数，绕过 ultralytics 的 monkey-patch
# ──────────────────────────────────────────────────────────────
cv2.imshow = _ORIG_IMSHOW
cv2.waitKey = _ORIG_WAITKEY
cv2.destroyAllWindows = _ORIG_DESTROY_ALL

from config import (
    OBJECT_MODEL_PATH,
    OBJECT_CONFIDENCE,
    OBJECT_IMGSZ,
)


# ---- 绘制颜色 ----
PALETTE = [
    (0, 255, 0),     # 绿
    (255, 0, 0),     # 蓝
    (0, 0, 255),     # 红
    (255, 255, 0),   # 青
    (255, 0, 255),   # 紫
    (0, 255, 255),   # 黄
    (128, 255, 0),   # 黄绿
    (255, 128, 0),   # 橙
]


def draw_detection_boxes(frame, boxes_result, class_names):
    """在帧上绘制检测框（detection 模式）。"""
    if boxes_result is None or boxes_result.boxes is None:
        return frame

    boxes = boxes_result.boxes
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i])
        conf = float(boxes.conf[i])
        xyxy = boxes.xyxy[i].cpu().numpy()
        x1, y1, x2, y2 = map(int, xyxy)

        color = PALETTE[cls_id % len(PALETTE)]
        name = class_names.get(cls_id, f"cls_{cls_id}")

        # 框
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # 标签背景
        label = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame


def draw_classification_overlay(frame, probs_result, class_names, top_n=5):
    """在帧中央绘制分类结果（classification 模式）。"""
    if probs_result is None or probs_result.probs is None:
        return frame

    probs = probs_result.probs
    top5_indices = probs.top5
    top5_confs = probs.top5conf

    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    # 半透明背景面板
    panel_w = 280
    panel_h = 30 + 25 * top_n
    px1 = cx - panel_w // 2
    py1 = cy - panel_h // 2 - 60
    px2 = px1 + panel_w
    py2 = py1 + panel_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (px1, py1), (px2, py2), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.rectangle(frame, (px1, py1), (px2, py2), (100, 100, 100), 1)

    # 标题
    cv2.putText(frame, "Classification (Top-5)", (px1 + 10, py1 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    for rank in range(min(top_n, len(top5_indices))):
        idx = int(top5_indices[rank])
        conf = float(top5_confs[rank])
        name = class_names.get(idx, f"cls_{idx}")
        bar_w = int((panel_w - 80) * conf)

        y = py1 + 48 + rank * 25
        cv2.putText(frame, f"#{rank + 1} {name}", (px1 + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

        # 置信度条
        bar_color = (0, int(255 * conf), int(255 * (1 - conf)))
        cv2.rectangle(frame, (px1 + 130, y - 10), (px1 + 130 + bar_w, y + 2),
                      bar_color, -1)
        cv2.putText(frame, f"{conf:.3f}", (px1 + 135 + bar_w, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    return frame


def draw_hud(frame, fps, mode, model_name):
    """绘制 HUD 信息条。"""
    h, w = frame.shape[:2]

    # 顶部半透明条
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 38), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    cv2.putText(frame, f"FPS: {fps:.1f}", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, f"Mode: {mode}", (120, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
    cv2.putText(frame, f"Model: {model_name}", (300, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # 底部提示
    bot_y = h - 12
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 32), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay2, 0.4, frame, 0.6, 0, frame)
    cv2.putText(frame, "Q:Quit  M:Switch Mode  D:Toggle HUD  S:Screenshot",
                (8, bot_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    return frame


def main():
    print("=" * 60)
    print("  YOLO 流式视频检测测试")
    print("=" * 60)

    # ---- 检测 OpenCV GUI 可用性 ----
    if not _check_opencv_gui_available():
        print()
        print("=" * 60)
        print("  ERROR: OpenCV GUI 功能不可用！")
        print("=" * 60)
        print()
        print("  当前环境安装的是 opencv-python-headless")
        print("  （无头版本，不含 imshow / namedWindow 等 GUI 功能）。")
        print()
        print("  请执行以下命令修复：")
        print()
        print("    pip uninstall opencv-python-headless -y")
        print("    pip install opencv-python")
        print()
        print("  或使用 opencv-contrib-python：")
        print()
        print("    pip uninstall opencv-python-headless -y")
        print("    pip install opencv-contrib-python")
        print()
        print("=" * 60)
        sys.exit(1)

    # ------------------------------------------------------------
    # 加载模型
    # ------------------------------------------------------------
    model_path = Path(OBJECT_MODEL_PATH)
    if not model_path.exists():
        print(f"[ERROR] 模型文件不存在: {model_path.resolve()}")
        sys.exit(1)

    device = 0 if torch.cuda.is_available() else "cpu"
    device_name = "GPU" if torch.cuda.is_available() else "CPU"
    print(f"[INFO] 设备: {device_name}")
    print(f"[INFO] 模型: {model_path.name}")
    print(f"[INFO] 置信度阈值: {OBJECT_CONFIDENCE}")
    print(f"[INFO] 推理尺寸: {OBJECT_IMGSZ}")

    model = YOLO(str(model_path))

    # 自动检测模型类型
    has_detection = False
    has_classification = False
    try:
        dummy = model.predict(
            source="ultralytics/assets/bus.jpg" if not Path("ultralytics/assets/bus.jpg").exists()
            else "ultralytics/assets/bus.jpg",
            imgsz=OBJECT_IMGSZ,
            device=device,
            verbose=False,
        )[0]
        if dummy.boxes is not None:
            has_detection = True
        if dummy.probs is not None:
            has_classification = True
    except Exception:
        import numpy as np
        dummy = model.predict(
            source=np.zeros((OBJECT_IMGSZ, OBJECT_IMGSZ, 3), dtype=np.uint8),
            imgsz=OBJECT_IMGSZ,
            device=device,
            verbose=False,
        )[0]
        if dummy.boxes is not None:
            has_detection = True
        if dummy.probs is not None:
            has_classification = True

    print(f"[INFO] 检测模式 (boxes):  {'可用' if has_detection else '不可用'}")
    print(f"[INFO] 分类模式 (probs):  {'可用' if has_classification else '不可用'}")

    if not has_detection and not has_classification:
        print("[ERROR] 模型不支持检测也不支持分类，无法进行流式推理。")
        sys.exit(1)

    mode = "detection" if has_detection else "classification"
    print(f"[INFO] 初始模式: {mode}")

    class_names = model.names
    print(f"[INFO] 模型类别: {class_names}")

    # ------------------------------------------------------------
    # 打开摄像头
    # ------------------------------------------------------------
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] 无法打开摄像头。")
        sys.exit(1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] 摄像头分辨率: {actual_w}x{actual_h}")

    # ------------------------------------------------------------
    # 状态变量
    # ------------------------------------------------------------
    show_hud = True
    fps = 0.0
    frame_count = 0
    last_fps_time = time.time()

    WINDOW_NAME = "VideoDetection"

    print("\n[INFO] 流式检测已启动，按 Q 退出...\n")

    # ------------------------------------------------------------
    # 流式检测主循环
    # ------------------------------------------------------------
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        frame_count += 1

        # FPS
        now = time.time()
        elapsed = now - last_fps_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            last_fps_time = now

        # ---- 流式推理 ----
        t0 = time.time()
        results = model.predict(
            source=frame,
            imgsz=OBJECT_IMGSZ,
            conf=OBJECT_CONFIDENCE,
            device=device,
            verbose=False,
        )
        inference_ms = (time.time() - t0) * 1000

        result = results[0]

        # ---- 绘制结果 ----
        if mode == "detection" and result.boxes is not None:
            n_objects = len(result.boxes)
            frame = draw_detection_boxes(frame, result, class_names)
            if n_objects > 0:
                print(f"  [DETECT] {n_objects} objects | {inference_ms:.0f}ms", end="")
                for i in range(min(n_objects, 5)):
                    cls_id = int(result.boxes.cls[i])
                    conf = float(result.boxes.conf[i])
                    name = class_names.get(cls_id, "?")
                    print(f"  {name}:{conf:.2f}", end="")
                print()

        elif mode == "classification" and result.probs is not None:
            top1_id = int(result.probs.top1)
            top1_conf = float(result.probs.top1conf)
            top1_name = class_names.get(top1_id, "?")
            frame = draw_classification_overlay(frame, result, class_names)
            print(f"  [CLASSIFY] #{1} {top1_name}:{top1_conf:.3f} | {inference_ms:.0f}ms")

        # ---- HUD ----
        if show_hud:
            frame = draw_hud(frame, fps, mode, model_path.name)

        # ---- 显示 ----
        cv2.imshow(WINDOW_NAME, frame)

        # ---- 按键 ----
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            print("\n[INFO] 用户退出。")
            break
        elif key == ord("m"):
            if has_detection and has_classification:
                mode = "classification" if mode == "detection" else "detection"
                print(f"[INFO] 切换模式: {mode}")
        elif key == ord("d"):
            show_hud = not show_hud
            print(f"[INFO] HUD: {'ON' if show_hud else 'OFF'}")
        elif key == ord("s"):
            ss_dir = Path("screenshots")
            ss_dir.mkdir(exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = ss_dir / f"stream_detect_{ts}.jpg"
            cv2.imwrite(str(fname), frame)
            print(f"[INFO] 截图已保存: {fname}")

    # ------------------------------------------------------------
    # 清理
    # ------------------------------------------------------------
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] 测试结束。")


if __name__ == "__main__":
    main()
