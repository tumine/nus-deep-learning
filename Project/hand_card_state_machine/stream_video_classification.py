"""
stream_video_classification.py

通过 HTTP MJPEG 视频流实时显示画面并叠加 YOLO 模型的检测/分类结果。

与 test_video_classification.py（使用本地摄像头）不同，本脚本：
- 从 http://100.84.2.68:5000/video_feed 拉取远端视频流
- 对每一帧持续进行 YOLO 推理
- 检测模式（有 boxes）：在画面中画出目标边界框和标签
- 分类模式（只有 probs）：在画面中央显示 top-N 分类结果
- 显示实时 FPS，验证流式推理性能
- 支持断线自动重连

用法：
    python stream_video_classification.py

按键：
    Q / ESC  → 退出
    M        → 切换 detection / classification 模式
    D        → 切换调试信息（HUD）
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


# ================================================================
# 配置
# ================================================================
STREAM_URL = "http://100.84.2.68:5000/video_feed"

# 每隔多少帧做一次推理（1 = 每帧，2 = 隔一帧，可提高流畅度）
FRAME_STRIDE = 1

# 重连最大尝试次数
MAX_RECONNECT_ATTEMPTS = 10
# 重连间隔（秒）
RECONNECT_DELAY = 2.0


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


def compute_iou(box1, box2):
    """计算两个 xyxy 格式框的 IoU（交并比）。"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return inter_area / (area1 + area2 - inter_area + 1e-6)


def filter_overlapping_by_priority(boxes_result, class_names, iou_threshold=0.3):
    """
    对重叠位置上的 pencil / eraser / block 检测框进行优先级过滤。

    优先级：block > eraser > pencil
    返回: 需要丢弃的框索引集合（skip_set）。
    """
    if boxes_result is None or boxes_result.boxes is None:
        return set()

    boxes = boxes_result.boxes
    n = len(boxes)
    if n == 0:
        return set()

    priority_map: dict[int, int] = {}
    for cls_id, cls_name in class_names.items():
        cls_id = int(cls_id)
        name = cls_name.lower().strip()
        if name == "block":
            priority_map[cls_id] = 3
        elif name == "eraser":
            priority_map[cls_id] = 2
        elif name == "pencil":
            priority_map[cls_id] = 1

    if not priority_map:
        return set()

    xyxy = boxes.xyxy.cpu().numpy()
    cls_ids = [int(boxes.cls[i]) for i in range(n)]

    remove_indices: set[int] = set()

    for i in range(n):
        if i in remove_indices:
            continue
        if cls_ids[i] not in priority_map:
            continue

        for j in range(i + 1, n):
            if j in remove_indices:
                continue
            if cls_ids[j] not in priority_map:
                continue

            iou = compute_iou(xyxy[i], xyxy[j])
            if iou >= iou_threshold:
                pri_i = priority_map[cls_ids[i]]
                pri_j = priority_map[cls_ids[j]]
                if pri_i >= pri_j:
                    remove_indices.add(j)
                else:
                    remove_indices.add(i)
                    break

    return remove_indices


def draw_detection_boxes(frame, boxes_result, class_names, skip_set=None):
    """在帧上绘制检测框（detection 模式）。"""
    if boxes_result is None or boxes_result.boxes is None:
        return frame

    if skip_set is None:
        skip_set = set()

    boxes = boxes_result.boxes
    for i in range(len(boxes)):
        if i in skip_set:
            continue
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


def draw_hud(frame, fps, mode, model_name, stream_url, reconnect_info=""):
    """绘制 HUD 信息条。"""
    h, w = frame.shape[:2]

    # 顶部半透明条
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 48), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    cv2.putText(frame, f"FPS: {fps:.1f}", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, f"Mode: {mode}", (120, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
    cv2.putText(frame, f"Model: {model_name}", (300, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # 流地址（截断显示）
    short_url = stream_url if len(stream_url) <= 40 else stream_url[:37] + "..."
    cv2.putText(frame, f"Stream: {short_url}", (8, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 180, 255), 1)

    # 重连状态
    if reconnect_info:
        cv2.putText(frame, reconnect_info, (300, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

    # 底部提示
    bot_y = h - 12
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 32), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay2, 0.4, frame, 0.6, 0, frame)
    cv2.putText(frame, "Q:Quit  M:Switch Mode  D:Toggle HUD  S:Screenshot",
                (8, bot_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    return frame


def open_stream(url: str) -> cv2.VideoCapture | None:
    """打开视频流，返回 cv2.VideoCapture 对象，失败返回 None。"""
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        return None

    # 预热：读取几帧确保连接稳定
    for _ in range(3):
        cap.read()

    return cap


def main():
    print("=" * 60)
    print("  YOLO 远端视频流检测/分类")
    print("=" * 60)
    print(f"  流地址: {STREAM_URL}")
    print()

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
    print(f"[INFO] 帧步长: {FRAME_STRIDE}（每 {FRAME_STRIDE} 帧推理一次）")

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
    # 打开远端视频流
    # ------------------------------------------------------------
    print(f"\n[INFO] 正在连接视频流: {STREAM_URL}")
    cap = open_stream(STREAM_URL)
    if cap is None:
        print(f"[ERROR] 无法连接到视频流: {STREAM_URL}")
        print("        请确认服务器已启动且地址正确。")
        sys.exit(1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] 视频流分辨率: {actual_w}x{actual_h}")

    # ------------------------------------------------------------
    # 状态变量
    # ------------------------------------------------------------
    show_hud = True
    fps = 0.0
    frame_count = 0
    last_fps_time = time.time()
    reconnect_attempts = 0
    reconnect_info = ""
    last_inference_frame = None  # 缓存的上一帧推理结果

    WINDOW_NAME = "StreamDetection"

    print("\n[INFO] 流式检测已启动，按 Q 退出...\n")

    # ------------------------------------------------------------
    # 流式检测主循环
    # ------------------------------------------------------------
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("  [WARN] 读取帧失败，尝试重连...")
            reconnect_attempts += 1

            if reconnect_attempts > MAX_RECONNECT_ATTEMPTS:
                print(f"  [ERROR] 重连失败已达上限 ({MAX_RECONNECT_ATTEMPTS} 次)，退出。")
                break

            reconnect_info = f"Reconnecting... ({reconnect_attempts}/{MAX_RECONNECT_ATTEMPTS})"
            print(f"  {reconnect_info}")

            # 如果还有上一帧缓存，显示断线画面
            if last_inference_frame is not None:
                display_frame = last_inference_frame.copy()
                # 叠加断线提示
                h, w = display_frame.shape[:2]
                cv2.putText(display_frame, "STREAM LOST - RECONNECTING...",
                            (w // 2 - 200, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow(WINDOW_NAME, display_frame)
                cv2.waitKey(1)

            cap.release()
            time.sleep(RECONNECT_DELAY)
            cap = open_stream(STREAM_URL)
            if cap is not None:
                print("  [INFO] 重连成功！")
                reconnect_attempts = 0
                reconnect_info = ""
            else:
                reconnect_info = f"Reconnect failed ({reconnect_attempts}/{MAX_RECONNECT_ATTEMPTS})"
            continue

        # 重置重连计数（成功读取到帧）
        reconnect_attempts = 0
        reconnect_info = ""

        frame_count += 1

        # FPS
        now = time.time()
        elapsed = now - last_fps_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            last_fps_time = now

        # ---- 帧步长控制：每隔 FRAME_STRIDE 帧做一次推理 ----
        should_infer = (frame_count % FRAME_STRIDE == 0)

        if should_infer:
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
                skip_set = filter_overlapping_by_priority(result, class_names)
                n_filtered = n_objects - len(skip_set)
                frame = draw_detection_boxes(frame, result, class_names, skip_set)
                if n_filtered > 0:
                    det_str = f"  [DETECT] {n_filtered} objects | {inference_ms:.0f}ms"
                    printed = 0
                    for i in range(n_objects):
                        if i in skip_set:
                            continue
                        cls_id = int(result.boxes.cls[i])
                        conf = float(result.boxes.conf[i])
                        name = class_names.get(cls_id, "?")
                        det_str += f"  {name}:{conf:.2f}"
                        printed += 1
                        if printed >= 5:
                            break
                    print(det_str)

            elif mode == "classification" and result.probs is not None:
                top1_id = int(result.probs.top1)
                top1_conf = float(result.probs.top1conf)
                top1_name = class_names.get(top1_id, "?")
                frame = draw_classification_overlay(frame, result, class_names)
                print(f"  [CLASSIFY] #{1} {top1_name}:{top1_conf:.3f} | {inference_ms:.0f}ms")

            # 缓存当前帧（含推理结果），用于断线时显示
            last_inference_frame = frame.copy()
        else:
            # 跳过推理的帧：复用上一次的检测框绘制 + 本帧原始画面
            if last_inference_frame is not None:
                # 如果上一帧有推理结果，可以简单保持原始帧用于流畅显示
                pass

        # ---- HUD ----
        if show_hud:
            frame = draw_hud(frame, fps, mode, model_path.name, STREAM_URL, reconnect_info)
        else:
            # 即使关闭 HUD，也显示重连信息
            if reconnect_info:
                h, _ = frame.shape[:2]
                cv2.putText(frame, reconnect_info, (8, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        # ---- 显示 ----
        cv2.imshow(WINDOW_NAME, frame)

        # ---- 按键 ----
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:  # Q / ESC
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
    print("[INFO] 程序结束。")


if __name__ == "__main__":
    main()
