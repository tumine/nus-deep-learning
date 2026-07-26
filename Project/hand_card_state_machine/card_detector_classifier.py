"""
card_detector_classifier_strict.py

Strict one-shot classification detector for the classroom robot.

Design goals:
1. Keep the original CardDetector API:
       detector = CardDetector()
       results = detector.detect(frame)
       frame = detector.draw(frame)
2. Return only ONE confirmed request in each request session.
3. Require consecutive stable predictions, not loose voting.
4. Never auto-unlock when the object leaves the ROI.
5. Only main.py may start a new session by calling reset_session().
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from config import (
    BOX_COLOR,
    CENTER_COLOR,
    TEXT_COLOR,
    OBJECT_MODEL_PATH,
    OBJECT_CONFIDENCE,
    OBJECT_IMGSZ,
    CLASS_NAMES,
    OBJECT_INPUT_SIZE,
    CLASSIFIER_MIN_AVG_CONFIDENCE,
    CLASSIFIER_FRAME_STRIDE,
    CLASSIFIER_ROI_LEFT,
    CLASSIFIER_ROI_RIGHT,
    CLASSIFIER_ROI_TOP,
    CLASSIFIER_ROI_BOTTOM,
    CLASSIFIER_DEVICE,
    CONFIRM_FRAMES,
)


CLASS_NAME_TO_REQUEST = {
    "block": "blocks",
    "eraser": "eraser",
    "pencil": "pencil",
}

REQUEST_TO_ID = {
    "blocks": 0,
    "pencil": 1,
    "eraser": 2,
}


class CardDetector:
    """
    Strict classifier-based request detector.

    A request is confirmed only after CONFIRM_FRAMES consecutive valid
    predictions of the same class, all above OBJECT_CONFIDENCE, and with
    average confidence above CLASSIFIER_MIN_AVG_CONFIDENCE.

    After confirmation, detect() returns [] forever until reset_session()
    is explicitly called.
    """

    # ImageNet 标准化参数（必须与训练时一致）
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self) -> None:
        model_path = Path(OBJECT_MODEL_PATH)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Classification model not found: {model_path.resolve()}"
            )

        # ---- 设备选择 ----
        if CLASSIFIER_DEVICE == 0 or CLASSIFIER_DEVICE == "0":
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
        elif isinstance(CLASSIFIER_DEVICE, str) and CLASSIFIER_DEVICE.lower() == "cpu":
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(CLASSIFIER_DEVICE)

        # ---- 使用 torch.jit.load 加载 TorchScript 模型 ----
        self.model = torch.jit.load(str(model_path), map_location=self.device)
        self.model.eval()
        print(f"[CardDetector] TorchScript model loaded: {model_path}")
        print(f"[CardDetector] device: {self.device}")

        # ---- 类别名称（来自 config.py，顺序必须与训练时一致） ----
        self.class_names: list[str] = list(CLASS_NAMES)
        self.class_to_idx: dict[str, int] = {
            name: i for i, name in enumerate(self.class_names)
        }
        print(f"[CardDetector] classes ({len(self.class_names)}): {self.class_names}")

        # ---- 预处理 transform（与训练时 Resize+Crop+Normalize 一致） ----
        self._transform = transforms.Compose([
            transforms.Resize(int(OBJECT_INPUT_SIZE * 1.15)),
            transforms.CenterCrop(OBJECT_INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.IMAGENET_MEAN,
                std=self.IMAGENET_STD,
            ),
        ])

        self.frame_index = 0

        self.current_candidate: str | None = None
        self.consecutive_count = 0
        self.confidence_history: deque[float] = deque(
            maxlen=CONFIRM_FRAMES
        )

        self.session_locked = False
        self.confirmed_result: dict[str, Any] | None = None

        self.last_display: dict[str, Any] | None = None
        self.last_raw_class = "unknown"
        self.last_confidence = 0.0

    # ============================================================
    # Session control
    # ============================================================

    def reset_session(self) -> None:
        """
        Arm the detector for exactly one new student request.

        IMPORTANT:
        Call this only once when main.py enters WAIT_CARD / WAIT_REQUEST
        for a new student. Do not call it every frame.
        """
        self.frame_index = 0
        self.current_candidate = None
        self.consecutive_count = 0
        self.confidence_history.clear()

        self.session_locked = False
        self.confirmed_result = None
        self.last_display = None

        print("[CardDetector] session reset")

    # ============================================================
    # Internal helpers
    # ============================================================

    @staticmethod
    def _normalise_class_name(name: str) -> str:
        return name.strip().lower()

    @staticmethod
    def _roi(frame):
        height, width = frame.shape[:2]

        x1 = int(width * CLASSIFIER_ROI_LEFT)
        x2 = int(width * CLASSIFIER_ROI_RIGHT)
        y1 = int(height * CLASSIFIER_ROI_TOP)
        y2 = int(height * CLASSIFIER_ROI_BOTTOM)

        x1 = max(0, min(x1, width - 1))
        x2 = max(x1 + 1, min(x2, width))
        y1 = max(0, min(y1, height - 1))
        y2 = max(y1 + 1, min(y2, height))

        return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

    def _preprocess(self, roi_bgr: np.ndarray) -> torch.Tensor:
        """将 OpenCV BGR ROI 预处理为模型输入 tensor (1, 3, H, W)。"""
        # BGR → RGB
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        # numpy → PIL → transform
        image = Image.fromarray(roi_rgb)
        tensor = self._transform(image).unsqueeze(0)  # (1, 3, H, W)
        return tensor.to(self.device)

    def _predict(self, roi) -> tuple[str | None, str, float]:
        # 预处理
        input_tensor = self._preprocess(roi)

        # 推理
        with torch.no_grad():
            logits = self.model(input_tensor)

        # softmax → 概率
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])

        if 0 <= class_id < len(self.class_names):
            class_name = self._normalise_class_name(
                str(self.class_names[class_id])
            )
        else:
            return None, "unknown", confidence

        request = CLASS_NAME_TO_REQUEST.get(class_name)

        if request is None:
            return None, class_name, confidence

        if confidence < OBJECT_CONFIDENCE:
            return None, class_name, confidence

        return request, class_name, confidence

    def _reset_candidate(self) -> None:
        self.current_candidate = None
        self.consecutive_count = 0
        self.confidence_history.clear()

    # ============================================================
    # Detect
    # ============================================================

    def detect(self, frame):
        """
        Return one confirmed result once per session.

        Normal frame:
            []

        First stable confirmation:
            [{... confirmed=True ...}]

        Every later frame in the same session:
            []
        """
        if frame is None:
            return []

        roi, bbox = self._roi(frame)
        x1, y1, x2, y2 = bbox

        center = (
            int((x1 + x2) / 2),
            int((y1 + y2) / 2),
        )

        corners = [
            (x1, y1),
            (x2, y1),
            (x2, y2),
            (x1, y2),
        ]

        # Hard lock: never return another request in this session.
        if self.session_locked:
            if self.confirmed_result is not None:
                self.last_display = {
                    **self.confirmed_result,
                    "bbox": bbox,
                    "center": center,
                    "corners": corners,
                }
            return []

        self.frame_index += 1

        # Infer only at configured stride.
        if self.frame_index % max(1, CLASSIFIER_FRAME_STRIDE) != 0:
            return []

        request, class_name, confidence = self._predict(roi)

        self.last_raw_class = class_name
        self.last_confidence = confidence

        # Any invalid/low-confidence frame breaks the consecutive streak.
        if request is None:
            self._reset_candidate()

            self.last_display = {
                "id": -1,
                "request": None,
                "class_name": class_name,
                "confidence": confidence,
                "average_confidence": 0.0,
                "center": center,
                "corners": corners,
                "bbox": bbox,
                "count": 0,
                "confirmed": False,
            }

            return []

        # A class switch also breaks the streak completely.
        if request != self.current_candidate:
            self.current_candidate = request
            self.consecutive_count = 1
            self.confidence_history.clear()
            self.confidence_history.append(confidence)
        else:
            self.consecutive_count += 1
            self.confidence_history.append(confidence)

        average_confidence = (
            sum(self.confidence_history)
            / len(self.confidence_history)
        )

        result = {
            "id": REQUEST_TO_ID[request],
            "request": request,
            "class_name": class_name,
            "confidence": confidence,
            "average_confidence": average_confidence,
            "center": center,
            "corners": corners,
            "bbox": bbox,
            "count": self.consecutive_count,
            "confirmed": False,
        }

        # Strict confirmation:
        # - same class in CONFIRM_FRAMES consecutive inference frames
        # - confidence threshold passed on every one of those frames
        # - average confidence threshold passed
        if (
            self.consecutive_count >= CONFIRM_FRAMES
            and len(self.confidence_history) == CONFIRM_FRAMES
            and average_confidence >= CLASSIFIER_MIN_AVG_CONFIDENCE
        ):
            result["confirmed"] = True

            self.session_locked = True
            self.confirmed_result = result.copy()
            self.last_display = result.copy()

            print(
                "[CardDetector] confirmed exactly once:",
                request,
                f"count={self.consecutive_count}",
                f"avg={average_confidence:.3f}",
            )

            return [result.copy()]

        self.last_display = result
        return []

    # ============================================================
    # Draw
    # ============================================================

    def draw(self, frame):
        """
        Draw only the latest stored result.

        This method never runs classification again.
        Always call detect(frame) before draw(frame).
        """
        if frame is None or self.last_display is None:
            return frame

        result = self.last_display

        x1, y1, x2, y2 = result["bbox"]
        center_x, center_y = result["center"]

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            BOX_COLOR,
            2,
        )

        cv2.circle(
            frame,
            (center_x, center_y),
            5,
            CENTER_COLOR,
            -1,
        )

        request = result["request"]
        confidence = result["confidence"]
        count = result["count"]
        confirmed = result["confirmed"]

        if request is None:
            text = (
                f"{result['class_name']} "
                f"{confidence:.2f} INVALID"
            )
        elif confirmed:
            text = (
                f"{request} {confidence:.2f} "
                "LOCKED"
            )
        else:
            text = (
                f"{request} {confidence:.2f} "
                f"({count}/{CONFIRM_FRAMES})"
            )

        cv2.putText(
            frame,
            text,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            TEXT_COLOR,
            2,
            cv2.LINE_AA,
        )

        return frame
