"""
card_detector_classifier.py

Priority-based classification detector for the classroom robot.

Design goals:
1. Keep the original CardDetector API:
       detector = CardDetector()
       results = detector.detect(frame)
       frame = detector.draw(frame)
2. Return only ONE confirmed request in each request session.
3. Priority logic: blocks > eraser > pencil.
   Within CONFIRM_FRAMES, any block/eraser detection wins over pencil.
4. Never auto-unlock when the object leaves the ROI.
5. Only main.py may start a new session by calling reset_session().
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

from config import (
    BOX_COLOR,
    CENTER_COLOR,
    TEXT_COLOR,
    OBJECT_MODEL_PATH,
    OBJECT_CONFIDENCE,
    OBJECT_IMGSZ,
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
    Classifier-based request detector with priority-based window decision.

    Within a detection window of CONFIRM_FRAMES inference frames:
    - If any block detection occurs → result is blocks.
    - Else if any eraser detection occurs → result is eraser.
    - Else if any pencil detection occurs → result is pencil.
    - If nothing valid is seen → reset the window and keep waiting.

    After confirmation, detect() returns [] forever until reset_session()
    is explicitly called.
    """

    def __init__(self) -> None:
        model_path = Path(OBJECT_MODEL_PATH)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Classification model not found: {model_path.resolve()}"
            )

        self.model = YOLO(str(model_path))

        print("[CardDetector] model:", model_path)
        print("[CardDetector] classes:", self.model.names)

        self.frame_index = 0

        self.seen_eraser = False
        self.seen_block = False
        self.seen_pencil = False
        self.window_count = 0

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
        self.seen_eraser = False
        self.seen_block = False
        self.seen_pencil = False
        self.window_count = 0

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

    def _predict(self, roi) -> tuple[str | None, str, float]:
        result = self.model.predict(
            source=roi,
            imgsz=OBJECT_IMGSZ,
            device=CLASSIFIER_DEVICE,
            verbose=False,
        )[0]

        # ---- Classification mode (best.pt) ----
        if result.probs is not None:
            class_id = int(result.probs.top1)
            confidence = float(result.probs.top1conf)
            class_name = self._normalise_class_name(
                str(self.model.names[class_id])
            )

            request = CLASS_NAME_TO_REQUEST.get(class_name)

            if request is None:
                return None, class_name, confidence

            if confidence < OBJECT_CONFIDENCE:
                return None, class_name, confidence

            return request, class_name, confidence

        # ---- Detection mode fallback (best-1.pt / best-2.pt) ----
        if result.boxes is not None and len(result.boxes) > 0:
            # Choose the detection with highest confidence
            best_idx = int(result.boxes.conf.argmax())
            class_id = int(result.boxes.cls[best_idx])
            confidence = float(result.boxes.conf[best_idx])
            class_name = self._normalise_class_name(
                str(self.model.names[class_id])
            )

            request = CLASS_NAME_TO_REQUEST.get(class_name)

            if request is None:
                return None, class_name, confidence

            if confidence < OBJECT_CONFIDENCE:
                return None, class_name, confidence

            return request, class_name, confidence

        # Neither classification nor detection produced results
        return None, "unknown", 0.0

    def _reset_window(self) -> None:
        self.seen_eraser = False
        self.seen_block = False
        self.seen_pencil = False
        self.window_count = 0

    # ============================================================
    # Detect
    # ============================================================

    def detect(self, frame):
        """
        Return one confirmed result once per session.

        Within a detection window of CONFIRM_FRAMES inference frames,
        if any eraser or block detection occurs, the result is determined
        as eraser or block respectively. Otherwise the result is pencil.

        Normal frame:
            []

        First confirmation:
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

        # Track which classes have been seen in this window.
        if request is not None:
            if request == "eraser":
                self.seen_eraser = True
            elif request == "blocks":
                self.seen_block = True
            elif request == "pencil":
                self.seen_pencil = True

        self.window_count += 1

        display = {
            "id": REQUEST_TO_ID.get(request, -1),
            "request": request,
            "class_name": class_name,
            "confidence": confidence,
            "average_confidence": 0.0,
            "center": center,
            "corners": corners,
            "bbox": bbox,
            "count": self.window_count,
            "confirmed": False,
        }

        # After the window is full, determine the result.
        if self.window_count >= CONFIRM_FRAMES:
            if self.seen_block:
                final_request = "blocks"
            elif self.seen_eraser:
                final_request = "eraser"
            elif self.seen_pencil:
                final_request = "pencil"
            else:
                # Nothing valid was seen in this window;
                # reset and keep waiting.
                self._reset_window()
                self.last_display = display
                return []

            result = {
                "id": REQUEST_TO_ID[final_request],
                "request": final_request,
                "class_name": class_name,
                "confidence": confidence,
                "average_confidence": 0.0,
                "center": center,
                "corners": corners,
                "bbox": bbox,
                "count": self.window_count,
                "confirmed": True,
            }

            self.session_locked = True
            self.confirmed_result = result.copy()
            self.last_display = result.copy()

            print(
                "[CardDetector] confirmed:", final_request,
                f"window={self.window_count}",
                f"eraser={self.seen_eraser} block={self.seen_block}"
                f" pencil={self.seen_pencil}",
            )

            return [result.copy()]

        self.last_display = display
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
