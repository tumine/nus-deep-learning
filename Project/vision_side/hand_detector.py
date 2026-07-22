"""
hand_detector.py

Detect raised hands using a pretrained YOLO pose model.

The detector considers a hand raised when either wrist is above
its corresponding shoulder for several consecutive frames.
"""

import cv2
import torch
from ultralytics import YOLO


# ============================================================
# Configuration
# ============================================================

CONFIRM_FRAMES = 5

# 手腕至少高于肩膀多少像素，避免轻微抖动造成误判
WRIST_SHOULDER_MARGIN = 15

# COCO 人体关键点编号
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_WRIST = 9
RIGHT_WRIST = 10


class HandDetector:

    def __init__(
        self,
        model_path="yolov8n-pose.pt",
        conf=0.5
    ):
        """
        Initialize the pose-based hand-raise detector.

        Parameters
        ----------
        model_path : str
            Path or name of the YOLO pose model.

        conf : float
            Minimum person detection confidence.
        """

        self.model = YOLO(model_path)
        self.conf = conf

        self.device = 0 if torch.cuda.is_available() else "cpu"

        print(
            f"[HandDetector] Device: "
            f"{'GPU' if torch.cuda.is_available() else 'CPU'}"
        )

        self.detect_count = 0
        self.confirmed = False

        # 保存最后一帧检测结果，供 draw() 使用
        self.last_people = []

    # ============================================================
    # Detect raised hand
    # ============================================================

    def detect(self, frame):
        """
        Detect whether a person is raising either hand.

        Parameters
        ----------
        frame : numpy.ndarray
            Current video frame.

        Returns
        -------
        list
            Confirmed hand-raise events.
        """

        events = []
        self.last_people = []

        results = self.model.predict(
            source=frame,
            conf=self.conf,
            device=self.device,
            verbose=False
        )

        raised_person = None

        for result in results:

            if result.boxes is None or result.keypoints is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()

            keypoints_xy = result.keypoints.xy.cpu().numpy()

            # 部分版本会提供关键点置信度
            keypoints_conf = None

            if result.keypoints.conf is not None:
                keypoints_conf = result.keypoints.conf.cpu().numpy()

            for index, person_keypoints in enumerate(keypoints_xy):

                if index >= len(boxes):
                    continue

                x1, y1, x2, y2 = map(int, boxes[index])
                confidence = float(confidences[index])

                left_shoulder = person_keypoints[LEFT_SHOULDER]
                right_shoulder = person_keypoints[RIGHT_SHOULDER]
                left_wrist = person_keypoints[LEFT_WRIST]
                right_wrist = person_keypoints[RIGHT_WRIST]

                left_raised = self._is_wrist_raised(
                    wrist=left_wrist,
                    shoulder=left_shoulder,
                    keypoints_conf=keypoints_conf,
                    person_index=index,
                    wrist_index=LEFT_WRIST,
                    shoulder_index=LEFT_SHOULDER
                )

                right_raised = self._is_wrist_raised(
                    wrist=right_wrist,
                    shoulder=right_shoulder,
                    keypoints_conf=keypoints_conf,
                    person_index=index,
                    wrist_index=RIGHT_WRIST,
                    shoulder_index=RIGHT_SHOULDER
                )

                hand_raised = left_raised or right_raised

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                person_info = {
                    "bbox": (x1, y1, x2, y2),
                    "center": (center_x, center_y),
                    "confidence": confidence,
                    "left_shoulder": tuple(map(int, left_shoulder)),
                    "right_shoulder": tuple(map(int, right_shoulder)),
                    "left_wrist": tuple(map(int, left_wrist)),
                    "right_wrist": tuple(map(int, right_wrist)),
                    "left_raised": left_raised,
                    "right_raised": right_raised,
                    "hand_raised": hand_raised
                }

                self.last_people.append(person_info)

                # 当前先取第一个检测到举手的人
                if hand_raised and raised_person is None:
                    raised_person = person_info

        # ========================================================
        # Multi-frame confirmation
        # ========================================================

        if raised_person is not None:

            self.detect_count += 1

            if (
                self.detect_count >= CONFIRM_FRAMES
                and not self.confirmed
            ):
                self.confirmed = True

                events.append({
                    "type": "hand_raise",
                    "target": raised_person["center"],
                    "confidence": raised_person["confidence"],
                    "bbox": raised_person["bbox"],
                    "left_raised": raised_person["left_raised"],
                    "right_raised": raised_person["right_raised"]
                })

        else:

            self.detect_count = 0
            self.confirmed = False

        return events

    def _is_wrist_raised(
        self,
        wrist,
        shoulder,
        keypoints_conf,
        person_index,
        wrist_index,
        shoulder_index
    ):
        """
        Check whether one wrist is visibly above its shoulder.
        """

        wrist_x, wrist_y = wrist
        shoulder_x, shoulder_y = shoulder

        # 坐标为 0 通常表示关键点没有检测到
        if wrist_x <= 0 or wrist_y <= 0:
            return False

        if shoulder_x <= 0 or shoulder_y <= 0:
            return False

        # 如果模型提供关键点置信度，则过滤低置信度关键点
        if keypoints_conf is not None:

            wrist_conf = keypoints_conf[person_index][wrist_index]
            shoulder_conf = keypoints_conf[person_index][shoulder_index]

            if wrist_conf < 0.3 or shoulder_conf < 0.3:
                return False

        return wrist_y < shoulder_y - WRIST_SHOULDER_MARGIN

    # ============================================================
    # Draw detection result
    # ============================================================

    def draw(self, frame):
        """
        Draw person boxes, shoulders, wrists and hand-raise status.
        """

        for person in self.last_people:

            x1, y1, x2, y2 = person["bbox"]

            left_shoulder = person["left_shoulder"]
            right_shoulder = person["right_shoulder"]
            left_wrist = person["left_wrist"]
            right_wrist = person["right_wrist"]

            hand_raised = person["hand_raised"]

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # 画肩膀
            cv2.circle(frame, left_shoulder, 5, (255, 0, 0), -1)
            cv2.circle(frame, right_shoulder, 5, (255, 0, 0), -1)

            # 画手腕
            cv2.circle(frame, left_wrist, 6, (0, 0, 255), -1)
            cv2.circle(frame, right_wrist, 6, (0, 0, 255), -1)

            cv2.line(
                frame,
                left_shoulder,
                left_wrist,
                (255, 255, 0),
                2
            )

            cv2.line(
                frame,
                right_shoulder,
                right_wrist,
                (255, 255, 0),
                2
            )

            if hand_raised:

                if self.confirmed:
                    text = "HAND RAISED - CONFIRMED"
                else:
                    text = (
                        f"HAND RAISED "
                        f"({self.detect_count}/{CONFIRM_FRAMES})"
                    )

            else:
                text = "NO HAND RAISE"

            cv2.putText(
                frame,
                text,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

        return frame