"""
yolo_stream_detect.py

从 HTTP MJPEG 视频流中每 15 帧抽取一帧，使用 YOLO 模型 (best.pt) 进行目标检测，
并将检测到的帧与 YOLO 格式标签保存为预标记数据集：

输出结构 (与 yolo detect predict 一致)：
    Project/request_recogniser/prelabels/handheld/
    ├── frame_0001.jpg
    ├── frame_0002.jpg
    └── labels/
        ├── frame_0001.txt
        └── frame_0002.txt

标签格式：class_id x_center y_center width height confidence（归一化坐标）

用法：
    python yolo_stream_detect.py
"""

import sys
import os

import cv2
from ultralytics import YOLO

# ------------------------------------------------------------
# 配置
# ------------------------------------------------------------
STREAM_URL = "http://100.84.2.68:5000/video_feed"

# best.pt 路径（相对于本文件所在目录的 hand_card_state_machine/best.pt）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "hand_card_state_machine", "best.pt")

CONFIDENCE_THRESHOLD = 0.2
FRAME_STRIDE = 15
IMGSZ = 1024

# 输出目录（与 yolo detect predict 的 project/name 参数一致）
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "request_recogniser", "prelabels", "handheld")

# ------------------------------------------------------------
# 准备输出目录
# ------------------------------------------------------------
IMAGES_DIR = OUTPUT_DIR
LABELS_DIR = os.path.join(OUTPUT_DIR, "labels")
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)

# 已有文件数量，用于接续编号
existing_images = [f for f in os.listdir(IMAGES_DIR) if f.endswith(".jpg")]
save_index = len(existing_images)

# ------------------------------------------------------------
# 加载模型
# ------------------------------------------------------------
print(f"正在加载模型: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# ------------------------------------------------------------
# 打开视频流
# ------------------------------------------------------------
print(f"正在连接视频流: {STREAM_URL}")
cap = cv2.VideoCapture(STREAM_URL)
if not cap.isOpened():
    print(f"❌ 无法打开视频流: {STREAM_URL}")
    sys.exit(1)

# 预热：读取几帧确保连接稳定
for _ in range(5):
    cap.read()

print(f"已连接，开始检测（每 {FRAME_STRIDE} 帧检测一次，置信度 ≥ {CONFIDENCE_THRESHOLD}）")
print(f"检测结果将保存到: {OUTPUT_DIR}")
print("按 Ctrl+C 终止。\n")

frame_count = 0
saved_this_session = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠ 读取帧失败，尝试重连...")
            cap.release()
            cap = cv2.VideoCapture(STREAM_URL)
            continue

        frame_count += 1

        # 每隔 FRAME_STRIDE 帧处理一次
        if frame_count % FRAME_STRIDE != 0:
            continue

        # YOLO 推理
        results = model(frame, conf=CONFIDENCE_THRESHOLD, imgsz=IMGSZ, verbose=False)

        detections = results[0]
        boxes = detections.boxes

        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        print(f"--- 帧 #{frame_count} (流时间: {timestamp:.1f}s) ---")

        if boxes is None or len(boxes) == 0:
            print("  (未检测到目标)")
        else:
            # 终端输出检测结果
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                class_name = model.names.get(cls_id, f"class_{cls_id}")

                print(
                    f"  {class_name:<12}  conf={conf:.3f}  "
                    f"box=({xyxy[0]:.0f},{xyxy[1]:.0f},{xyxy[2]:.0f},{xyxy[3]:.0f})"
                )

            # --- 保存为 YOLO 预标记数据集 ---
            save_index += 1
            img_h, img_w = frame.shape[:2]

            # 保存图像
            image_filename = f"frame_{save_index:04d}.jpg"
            image_path = os.path.join(IMAGES_DIR, image_filename)
            cv2.imwrite(image_path, frame)

            # 保存标签 (YOLO 格式: class_id x_center y_center width height confidence)
            label_filename = f"frame_{save_index:04d}.txt"
            label_path = os.path.join(LABELS_DIR, label_filename)

            with open(label_path, "w") as f:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    # 像素坐标 → 归一化 xywh
                    x_center = ((x1 + x2) / 2) / img_w
                    y_center = ((y1 + y2) / 2) / img_h
                    width = (x2 - x1) / img_w
                    height = (y2 - y1) / img_h

                    f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {conf:.6f}\n")

            saved_this_session += 1
            print(f"  💾 已保存 → {image_filename} / labels/{label_filename}")

        print()

except KeyboardInterrupt:
    print(f"\n用户终止程序。本次会话共保存 {saved_this_session} 帧。")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("资源已释放，程序退出。")
