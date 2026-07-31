"""
config.py

Project configuration for YOLO object-request detection.
"""

import cv2

# --------------------------------------------------
# Camera
# --------------------------------------------------
CAMERA_ID = 0
FRAME_WIDTH = 1000

# --------------------------------------------------
# Legacy ArUco configuration
# --------------------------------------------------
ARUCO_DICT = cv2.aruco.DICT_4X4_50

# --------------------------------------------------
# YOLO object request detection
# --------------------------------------------------
OBJECT_MODEL_PATH = "Project/hand_card_state_machine/best-2.pt"
OBJECT_CONFIDENCE = 0.6 

OBJECT_IMGSZ = 800

CLASSIFIER_WINDOW_SIZE = 10
CLASSIFIER_MIN_AVG_CONFIDENCE = 0.80

# 每隔几帧运行一次分类
CLASSIFIER_FRAME_STRIDE = 1

# 连续多少帧没有可靠目标后，允许同一物品再次触发
CLASSIFIER_RELEASE_FRAMES = 5

# 中央 ROI（全帧，不做裁剪）
CLASSIFIER_ROI_LEFT = 0.0
CLASSIFIER_ROI_RIGHT = 1.0
CLASSIFIER_ROI_TOP = 0.0
CLASSIFIER_ROI_BOTTOM = 1.0

CLASSIFIER_DEVICE = 0

SHOW_CLASSIFIER_DEBUG = True

# --------------------------------------------------
# Multi-frame confirmation
# --------------------------------------------------
CONFIRM_FRAMES = 15

# --------------------------------------------------
# Drawing
# --------------------------------------------------
BOX_COLOR = (0, 255, 0)
CENTER_COLOR = (0, 0, 255)
TEXT_COLOR = (255, 0, 0)
