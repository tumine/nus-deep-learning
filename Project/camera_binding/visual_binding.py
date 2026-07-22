import cv2
import numpy as np
import glob
import os

# ================= 1. 配置参数 =================
# 根据图2参数设置：10列，5行
SQUARES_X = 10
SQUARES_Y = 5

# 【必须修改】根据你在屏幕上的实际测量值填写（单位：米）
SQUARE_LENGTH = 0.025  # 示例: 25mm 
MARKER_LENGTH = 0.018  # 示例: 18mm 

# 存放你用树莓派摄像头拍摄的屏幕照片的文件夹路径
IMAGE_DIR = './calibration_images/*.jpg'

# ================= 2. 初始化标定板 =================
# Deepen AI 的 "original" 对应 OpenCV 的 DICT_ARUCO_ORIGINAL
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)

# 创建 ChArUco 板对象 (OpenCV 4.7+ 语法兼容)
board = cv2.aruco.CharucoBoard(
    (SQUARES_X, SQUARES_Y), 
    SQUARE_LENGTH, 
    MARKER_LENGTH, 
    aruco_dict
)

# 初始化检测器
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# ================= 3. 提取角点 =================
all_charuco_corners = []
all_charuco_ids = []
image_shape = None

images = glob.glob(IMAGE_DIR)
if not images:
    print("未找到图片，请检查路径。")
    exit()

print(f"找到 {len(images)} 张图片，开始处理...")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if image_shape is None:
        image_shape = gray.shape[::-1] # (width, height)

    # 检测 ArUco 标记
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None and len(ids) > 0:
        # 插值计算 ChArUco 棋盘格的内角点
        charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, board
        )

        # 至少需要 4 个角点来进行有效计算
        if charuco_retval > 3:
            all_charuco_corners.append(charuco_corners)
            all_charuco_ids.append(charuco_ids)
        else:
            print(f"图片 {fname} 提取的有效角点不足。")
    else:
        print(f"图片 {fname} 未检测到 ArUco 码。")

# ================= 4. 鱼眼相机标定 (Fisheye Calibration) =================
if len(all_charuco_corners) < 10:
    print("有效图片太少，建议至少需要 10 张以上的高质量图片！")
    exit()

print("角点提取完成，正在执行鱼眼标定算法...")

# 准备鱼眼模型所需的数据结构
obj_points = []
img_points = []

for i in range(len(all_charuco_corners)):
    # 获取当前图片中检测到的角点对应的 3D 物理坐标
    obj_p = board.getChessboardCorners()[all_charuco_ids[i]]
    obj_points.append(obj_p.reshape(-1, 1, 3))
    img_points.append(all_charuco_corners[i].reshape(-1, 1, 2))

# 初始化内参矩阵和畸变系数
K = np.zeros((3, 3))
D = np.zeros((4, 1))

# 设置鱼眼标定标志（校准主点，并使用 k1,k2,k3,k4）
flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW

# 执行标定
rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
    obj_points,
    img_points,
    image_shape,
    K,
    D,
    None,
    None,
    flags,
    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
)

print("\n--- 标定结果 ---")
print(f"重投影误差 (RMS): {rms:.4f} pixels (越小越好，通常应小于 1.0)")
print("\n内参矩阵 (K):")
print(K)
print("\n鱼眼畸变系数 (D) [k1, k2, k3, k4]:")
print(D.T)
