"""
hand_detector.py

Detect raised hands using YOLO.

Author: DL-V1
"""

import cv2
import torch
from ultralytics import YOLO

# ==========================
# Config
# ==========================

HAND_RAISE_HEIGHT = 0.4      # 手进入画面上40%区域认为举手
CONFIRM_FRAMES = 5           # 连续检测多少帧才确认


class HandDetector:

    def __init__(self,
                 model_path="yolov8_hand_raise.pt",
                 conf=0.5):

        self.model = YOLO(model_path)
        self.conf = conf

        print(f"[HandDetector] Device : {'GPU' if torch.cuda.is_available() else 'CPU'}")

        # 连续检测计数
        self.detect_count = 0

        # 是否已经确认过
        self.confirmed = False

        # 当前检测框（用于draw）
        self.last_boxes = []

    # ============================================================
    # Detect hand raise
    # ============================================================

    def detect(self, frame):

        events = []

        self.last_boxes = []

        results = self.model(
            frame,
            conf=self.conf,
            verbose=False
        )

        frame_height = frame.shape[0]

        detected = False

        for r in results:

            boxes = r.boxes

            if boxes is None:
                continue

            for box in boxes:

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                confidence = float(box.conf[0])

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                # 保存画框信息
                self.last_boxes.append({

                    "bbox": (x1, y1, x2, y2),

                    "center": (center_x, center_y),

                    "confidence": confidence

                })

                # 判断是否举手
                if y1 < frame_height * HAND_RAISE_HEIGHT:

                    detected = True

                    self.detect_count += 1

                    # 已经确认过，不重复发送
                    if self.confirmed:
                        continue

                    # 连续检测成功
                    if self.detect_count >= CONFIRM_FRAMES:

                        self.confirmed = True

                        events.append({

                            "type": "hand_raise",

                            "target": (center_x, center_y),

                            "confidence": confidence

                        })

        # 本帧没有检测到举手
        if not detected:

            self.detect_count = 0
            self.confirmed = False

        return events

    # ============================================================
    # Draw result
    # ============================================================

    def draw(self, frame):

        for info in self.last_boxes:

            x1, y1, x2, y2 = info["bbox"]

            center_x, center_y = info["center"]

            confidence = info["confidence"]

            cv2.rectangle(

                frame,

                (x1, y1),

                (x2, y2),

                (0, 255, 0),

                2

            )

            cv2.circle(

                frame,

                (center_x, center_y),

                5,

                (0, 0, 255),

                -1

            )

            if self.confirmed:

                text = f"Hand Raise ✓"

            else:

                text = f"Hand ({self.detect_count}/{CONFIRM_FRAMES})"

            cv2.putText(

                frame,

                text,

                (x1, y1 - 10),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.6,

                (255, 0, 0),

                2

            )

        return frame