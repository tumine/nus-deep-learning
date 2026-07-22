import cv2
import numpy as np
import glob

# ================= 1. 配置参数 =================
SQUARES_X = 16
SQUARES_Y = 10

SQUARE_LENGTH = 0.0216
MARKER_LENGTH = 0.017

IMAGE_DIR = './calib_images/*.jpg'

# ================= 2. 初始化标定板 =================
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)

board = cv2.aruco.CharucoBoard(
    (SQUARES_X, SQUARES_Y), 
    SQUARE_LENGTH, 
    MARKER_LENGTH, 
    aruco_dict
)

charuco_params = cv2.aruco.CharucoParameters()
detector_params = cv2.aruco.DetectorParameters()
refine_params = cv2.aruco.RefineParameters()

# 针对屏幕拍摄优化的检测参数
detector_params.adaptiveThreshWinSizeMin = 3
detector_params.adaptiveThreshWinSizeMax = 23
detector_params.adaptiveThreshConstant = 7

charuco_detector = cv2.aruco.CharucoDetector(
    board, charuco_params, detector_params, refine_params
)

# ================= 3. 提取角点 =================
all_charuco_corners = []
all_charuco_ids = []
image_shape = None

images = glob.glob(IMAGE_DIR)
if not images:
    print("未找到图片，请检查路径。")
    exit()

print(f"找到 {len(images)} 张图片，开始处理...")

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

for fname in images:
    img = cv2.imread(fname)
    
    if image_shape is None:
        image_shape = (img.shape[1], img.shape[0]) 

    # 提取绿通道并应用 CLAHE
    g_channel = img[:, :, 1]
    gray_enhanced = clahe.apply(g_channel)

    charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray_enhanced)

    if charuco_ids is not None and len(charuco_ids) > 3:
        all_charuco_corners.append(charuco_corners)
        all_charuco_ids.append(charuco_ids)
    elif marker_ids is not None and len(marker_ids) > 0:
        print(f"图片 {fname} 检测到 ArUco 码但 ChArUco 角点不足。")
    else:
        print(f"图片 {fname} 未检测到 ArUco 码。")

# ================= 3.5. 构建对象点 / 图像点 =================

if len(all_charuco_corners) < 10:
    print("有效图片太少，建议至少需要 10 张以上的高质量图片！")
    exit()

print("角点提取完成，正在构建标定数据...")

obj_points = []
img_points = []

NUM_COLS = SQUARES_X - 1

for i in range(len(all_charuco_corners)):
    ids = all_charuco_ids[i].flatten().astype(np.int32)
    corners = all_charuco_corners[i]

    n_pts = len(ids)
    obj_p = np.zeros((n_pts, 3), dtype=np.float32)
    for j, cid in enumerate(ids):
        cy = int(cid) // NUM_COLS
        cx = int(cid) % NUM_COLS
        obj_p[j] = [cx * SQUARE_LENGTH, cy * SQUARE_LENGTH, 0.0]

    # 【修复重点 1】: 强制指定形状为 (1, N, C)，并转换数据类型为 float64
    obj_points.append(obj_p.reshape(1, -1, 3).astype(np.float64))
    img_points.append(corners.reshape(1, -1, 2).astype(np.float64))

print(f"使用 {len(obj_points)} 张图片进行标定")
print(f"每张图片角点数: {[pts.shape[1] for pts in img_points]}")

# ================= 4. 相机标定 =================
USE_FISHEYE = True  

# 【修复重点 2】: 确保 K 和 D 初始化为 float64
K = np.zeros((3, 3), dtype=np.float64)
D = np.zeros((4, 1), dtype=np.float64)

if USE_FISHEYE:
    print("使用鱼眼标定模型...")
    try:
        fisheye_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
    except AttributeError:
        fisheye_flags = 2 | 8
        
    try:
        rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            obj_points, img_points, image_shape,
            K, D, None, None,
            fisheye_flags,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )
    except cv2.error as e:
        print(f"鱼眼标定抛出变长角点异常，正在对齐张量维度并重试...")
        # 【修复重点 3】: 适配 (1, N, C) 的形状获取和切片逻辑
        min_pts = min(pts.shape[1] for pts in img_points)
        obj_filtered = [p for p in obj_points if p.shape[1] >= min_pts]
        img_filtered = [p for p in img_points if p.shape[1] >= min_pts]
        
        # 使用多维切片 [:, :min_pts, :] 保留前缀维度
        obj_filtered = [p[:, :min_pts, :] for p in obj_filtered]
        img_filtered = [p[:, :min_pts, :] for p in img_filtered]
        
        print(f"统一维度为 {min_pts} 个角点，共 {len(obj_filtered)} 张图片参与计算")
        
        rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            obj_filtered, img_filtered, image_shape,
            K, D, None, None,
            fisheye_flags,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )
    print("\n鱼眼畸变系数 (D) [k1, k2, k3, k4]:")
    print(D.T)
else:
    print("使用标准相机标定模型 (pinhole)...")
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_shape,
        K, D,
        flags=0,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-9)
    )
    print("\n畸变系数 (D) [k1, k2, p1, p2, k3]:")
    print(D.T)

print("\n--- 标定结果 ---")
print(f"重投影误差 (RMS): {rms:.4f} pixels")
print("\n内参矩阵 (K):")
print(K)
