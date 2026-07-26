#!/usr/bin/env python3
"""
手动截取小车摄像头视频帧

从指定 URL 拉取 MJPEG 视频流，
实时预览，按空格键截取当前帧并保存到 screenshots/ 目录。
"""

import os
import sys
from datetime import datetime

import cv2

# ============ 配置 ============
CAMERA_URL = "http://100.84.2.68:5000/video_feed"   # 依据实际 URL 设置
SAVE_DIR = os.path.join(os.path.dirname(__file__), "screenshots")
WINDOW_NAME = "Car Camera - Press SPACE to capture, Q to quit"

# ============ 初始化 ============
os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(CAMERA_URL)
if not cap.isOpened():
    print(f"[ERROR] 无法连接摄像头: {CAMERA_URL}")
    sys.exit(1)

print(f"[INFO] 已连接到 {CAMERA_URL}")
print(f"[INFO] 按 空格键 截取当前帧")
print(f"[INFO] 按 Q 退出")
print(f"[INFO] 截图保存路径: {SAVE_DIR}")

# ============ 主循环 ============
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] 读取帧失败，正在重试...")
            cap.release()
            cap = cv2.VideoCapture(CAMERA_URL)
            continue

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("[INFO] 退出")
            break

        elif key == ord(" "):  # 空格键截帧
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"capture_{timestamp}.jpg"
            filepath = os.path.join(SAVE_DIR, filename)
            cv2.imwrite(filepath, frame)
            print(f"[CAPTURE] 已保存: {filename}")

except KeyboardInterrupt:
    print("\n[INFO] 用户中断")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] 资源已释放")
