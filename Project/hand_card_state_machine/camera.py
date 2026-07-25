"""
camera.py

Camera wrapper.
"""

import time
import cv2


class Camera:

    def __init__(self, source=0):
        """
        Args:
            source:
                0 -> Local USB camera
                URL -> Network video stream
        """
        self.source = source
        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera source: {source}")

    def read(self):

        ret, frame = self.cap.read()

        if not ret:
            return None

        return frame

    def reconnect(self):
        """Release the current capture and attempt to reconnect to the same source."""
        print("[CAMERA] Attempting to reconnect to video stream...")
        self.cap.release()
        time.sleep(2.0)  # 等待网络恢复
        self.cap = cv2.VideoCapture(self.source)
        if self.cap.isOpened():
            print("[CAMERA] Reconnected successfully.")
            return True
        else:
            print("[CAMERA] Reconnection failed.")
            return False

    def release(self):

        self.cap.release()