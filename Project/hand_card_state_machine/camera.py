"""
camera.py

Camera wrapper.
"""

import cv2


class Camera:

    def __init__(self, source=0):
        """
        Args:
            source:
                0 -> Local USB camera
                URL -> Network video stream
        """

        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera source: {source}")

    def read(self):

        ret, frame = self.cap.read()

        if not ret:
            return None

        return frame

    def release(self):

        self.cap.release()