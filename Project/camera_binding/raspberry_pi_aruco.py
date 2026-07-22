"""
raspberry_pi_aruco.py
====================
基于 ArUco 标签的视觉定位与小车控制程序（树莓派端）

功能：
  1. 使用 USB 摄像头实时检测 ArUco 标签（DICT_4X4_50）
  2. 通过 PnP 算法估算标签相对于摄像头的 3D 位姿
  3. 将 3D 位姿简化为 2D 平面偏差（距离 + 角度）
  4. 通过串口将偏差数据以 `<距离,角度>` 格式发送给 Arduino

硬件要求：
  - 树莓派 + USB 摄像头（建议 160° 鱼眼已标定）
  - Arduino 通过 USB 串口连接（波特率 115200）
  - 在环境中粘贴 ArUco DICT_4X4_50 标签，边长 10cm

数学约定（OpenCV 右手坐标系）：
  以摄像头光心为原点：
    X 轴 → 指向右方
    Y 轴 → 指向下方
    Z 轴 → 指向前方（摄像头拍摄方向）

  tvec = [X, Y, Z]  是标签坐标系原点在相机坐标系下的坐标
  2D 简化：
    水平距离  dist = sqrt(X² + Z²)
    水平夹角  angle = atan2(X, Z) * 180/π    （正值=标签在右侧，负值=标签在左侧）

  对小车而言，"angle > 0" 意味着目标在右侧，需要右转修正；
  "angle < 0" 意味着目标在左侧，需要左转修正。
"""

import cv2
import numpy as np
import serial
import time
import signal
import sys
import os

# ===========================================================================
# ★★★ 第 1 步：填入你的相机标定参数 ★★★
# ===========================================================================
# 以下占位数据是无效的示例，你必须用自己标定得到的结果替换！
#
# 如何获取这些参数：
#   1. 打印一张 ChArUco 或棋盘格标定板
#   2. 从不同角度、不同距离拍摄 15-30 张标定板照片（覆盖画面四角）
#   3. 使用本项目的 visual_binding.py 脚本进行鱼眼标定（160° 超广角镜头）
#      或使用 OpenCV 官方的 camera_calibration 工具进行标准针孔标定
#   4. 将标定得到的矩阵数值填入下方变量
#
# 内参矩阵 K (3x3) 的含义：
#   K = [[fx,  0, cx],
#        [ 0, fy, cy],
#        [ 0,  0,  1]]
#   fx, fy — 焦距（以像素为单位）
#   cx, cy — 主点坐标（光心在图像上的投影点，通常接近 宽/2, 高/2）
#
# 畸变系数 D 对于针孔模型（5 个参数）：
#   D = [k1, k2, p1, p2, k3]
#   k1,k2,k3 — 径向畸变系数
#   p1,p2   — 切向畸变系数
# ===========================================================================

# --- 方案 A：标准针孔相机模型（普通 USB 摄像头，视角 < 100°） ---
# 请将你的标定结果填入下方，覆盖这些示例值

CAMERA_MATRIX = np.array([
    [640.0,   0.0, 320.0],   # ← 替换：fx,  0,  cx
    [  0.0, 640.0, 240.0],   # ← 替换：0,  fy, cy
    [  0.0,   0.0,   1.0]    # 保持 [0, 0, 1] 不变
], dtype=np.float64)

# 针孔模型的畸变系数 [k1, k2, p1, p2, k3]
DIST_COEFFS = np.array([
    [0.1],                    # ← 替换：k1
    [-0.05],                  # ← 替换：k2
    [0.0],                    # ← 替换：p1
    [0.0],                    # ← 替换：p2
    [0.0]                     # ← 替换：k3
], dtype=np.float64)

# --- 方案 B：鱼眼相机模型（160° 超广角镜头，如 C3753N） ---
# 如果你的摄像头是鱼眼镜头，请：
#   1. 设置 USE_FISHEYE = True
#   2. 填入 FISHEYE_K 和 FISHEYE_D（通过 visual_binding.py 标定得到）
# 程序会在 ArUco 检测前先用 cv2.fisheye.undistortImage() 校正图像，
# 然后用内参矩阵 CAMERA_MATRIX（校正后的等效矩阵）进行位姿估计。

USE_FISHEYE = False  # ★ 如果使用鱼眼镜头，改为 True

# 鱼眼内参矩阵（原始未校正，通过 visual_binding.py 标定得到）
FISHEYE_K = np.array([
    [300.0,   0.0, 320.0],
    [  0.0, 300.0, 240.0],
    [  0.0,   0.0,   1.0]
], dtype=np.float64)

# 鱼眼畸变系数 [k1, k2, k3, k4]（4 个参数）
FISHEYE_D = np.array([
    [0.0],   # ← 替换：k1
    [0.0],   # ← 替换：k2
    [0.0],   # ← 替换：k3
    [0.0]    # ← 替换：k4
], dtype=np.float64)

# 鱼眼校正后的等效针孔内参矩阵（如果 USE_FISHEYE=True，在 main() 中自动计算）
# 你也可以手动填入一个平衡矩阵，例如：
#   fx = 图像对角线像素数 / (2 * tan(FOV/2))
RECTIFIED_CAMERA_MATRIX = None  # None 表示自动从 FISHEYE_K 计算

# ===========================================================================
# ★★★ 第 2 步：ArUco 标签参数 ★★★
# ===========================================================================

# 使用的 ArUco 字典类型，DICT_4X4_50 包含 50 个 4x4 标记
ARUCO_DICT = cv2.aruco.DICT_4X4_50

# ★ 标签物理边长，单位：米（米的十进制，例如贴纸 4.5cm → 0.045）
MARKER_SIZE = 0.1  # 10 cm

# ===========================================================================
# ★★★ 第 3 步：串口与通信参数 ★★★
# ===========================================================================

# ArUco 偏差数据→Arduino 的串口
SERIAL_PORT = '/dev/ttyUSB0'    # 也可能是 /dev/ttyACM0、/dev/ttyAMA0
SERIAL_BAUD = 115200            # 必须与 Arduino 端一致
SERIAL_TIMEOUT = 0.5            # 串口读写超时（秒）

# 数据发送频率控制（秒/帧），避免串口拥塞
SEND_INTERVAL = 0.1             # 10 Hz

# ===========================================================================
# ★★★ 第 4 步：摄像头与控制参数 ★★★
# ===========================================================================

CAMERA_INDEX = 0                # 摄像头设备索引（通常 0 是内置/第一个 USB 摄像头）
FRAME_WIDTH = 640               # 处理分辨率宽度
FRAME_HEIGHT = 480              # 处理分辨率高度
DISPLAY_WINDOW = True           # 是否显示实时画面（无显示器时可关闭）

# -------------------------------------------------------
# 2D 偏差计算与过滤
# -------------------------------------------------------

def compute_2d_error(tvec):
    """
    从 ArUco 标签的 3D 平移向量 tvec 计算小车导航所需的 2D 平面偏差。

    数学推导（OpenCV 右手坐标系）：
      tvec = [X, Y, Z] 是一个 3×1 的向量，表示：
        X — 标签中心在相机水平方向的偏移（m）。  X > 0 表示标签在相机右侧
        Y — 标签中心在相机垂直方向的偏移（m）。  Y > 0 表示标签在相机下方
        Z — 标签中心到相机光心的深度距离（m）。Z > 0 表示标签在相机前方

      小车运动在 XZ 平面上进行（忽略高度变化 Y轴）。

      水平距离 dist：
        dist = sqrt(X² + Z²)
        这是在 XZ 平面上相机到标签的直线距离（米）。

      水平偏移角 angle：
        angle = atan2(X, Z) * 180 / π
        当 X > 0（标签在右侧）→ angle > 0 → 车需右转
        当 X < 0（标签在左侧）→ angle < 0 → 车需左转

    参数：
      tvec: shape (1, 1, 3) 或 (3, 1)，从 estimatePoseSingleMarkers 返回

    返回：
      (distance_meters, angle_degrees) 元组
    """
    # 兼容不同的 tvec 返回格式：
    #   OpenCV 4.5+:  (1, 1, 3)
    #   直接 np.array: (3, 1) 或 (3,)
    tvec_flat = np.array(tvec).flatten()
    X, Y, Z = tvec_flat[0], tvec_flat[1], tvec_flat[2]

    # 水平距离 = sqrt(X² + Z²)，忽略 Y 轴高度差
    distance = np.sqrt(X * X + Z * Z)

    # 水平夹角（度），正值表示标签在右侧，负值表示在左侧
    angle_rad = np.arctan2(X, Z)  # atan2(水平偏移, 深度)
    angle_deg = np.degrees(angle_rad)

    return float(distance), float(angle_deg)


def is_valid_measurement(distance, max_dist=5.0):
    """
    对测量值做合理性校验，过滤离群值。

    返回 True 表示该测量值可信。
    """
    if distance <= 0.01:       # 太近，可能误检
        return False
    if distance > max_dist:    # 超过有效范围，精度不可靠
        return False
    return True


# ===========================================================================
# 信号处理（优雅退出）
# ===========================================================================

_shutdown_flag = False

def _signal_handler(sig, frame):
    """Ctrl+C 或 SIGTERM 时的清理回调。"""
    global _shutdown_flag
    _shutdown_flag = True
    print("\n[信息] 收到终止信号，正在安全退出...")


# ===========================================================================
# 主程序
# ===========================================================================

def main():
    global _shutdown_flag

    # 注册信号处理
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # ────────────────────────────────────────────────────────────────
    # 1. 初始化摄像头
    # ────────────────────────────────────────────────────────────────
    print(f"[信息] 正在打开摄像头 (index={CAMERA_INDEX})...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[错误] 无法打开摄像头 {CAMERA_INDEX}！")
        print("  请检查：① 摄像头是否连接 ② 设备索引是否正确 ③ 权限 (sudo)")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"[信息] 摄像头已就绪，分辨率: {int(actual_w)}x{int(actual_h)}")

    # ────────────────────────────────────────────────────────────────
    # 2. 初始化 ArUco 检测器
    # ────────────────────────────────────────────────────────────────
    print(f"[信息] 初始化 ArUco 检测器 (DICT_4X4_50, 标签边长={MARKER_SIZE}m)...")
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    aruco_params = cv2.aruco.DetectorParameters()

    # 提高检测鲁棒性的参数调整（可选的精细调参）
    # aruco_params.adaptiveThreshWinSizeMin = 3
    # aruco_params.adaptiveThreshWinSizeMax = 23
    # aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    # ────────────────────────────────────────────────────────────────
    # 3. 处理鱼眼校正（如果启用）
    # ────────────────────────────────────────────────────────────────
    cam_matrix = CAMERA_MATRIX       # 用于 PnP 的内参矩阵
    dist_coeffs = DIST_COEFFS        # 用于 PnP 的畸变系数

    if USE_FISHEYE:
        print("[信息] 鱼眼模式已启用，将先校正图像再检测 ArUco...")
        # 计算校正映射（只计算一次，提高实时性）
        DIM = (int(actual_w), int(actual_h))
        if RECTIFIED_CAMERA_MATRIX is None:
            # 自动计算平衡后的校正内参矩阵
            # balance=1.0: 保留所有有效像素；balance=0.0: 裁剪掉所有黑边
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                FISHEYE_K, FISHEYE_D, DIM, np.eye(3), balance=0.8
            )
            cam_matrix = new_K
            dist_coeffs = np.zeros((4, 1), dtype=np.float64)  # 校正后畸变为零
        else:
            cam_matrix = RECTIFIED_CAMERA_MATRIX
            dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # 预计算 undistort 映射表（initUndistortRectifyMap）
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            FISHEYE_K, FISHEYE_D, np.eye(3), cam_matrix, DIM, cv2.CV_16SC2
        )
        print(f"[信息] 鱼眼校正映射已计算，等效内参矩阵:\n{cam_matrix}")
    else:
        print(f"[信息] 使用标准针孔模型，内参矩阵:\n{cam_matrix}")

    # ────────────────────────────────────────────────────────────────
    # 4. 初始化串口（连接 Arduino）
    # ────────────────────────────────────────────────────────────────
    print(f"[信息] 正在打开串口 {SERIAL_PORT} @ {SERIAL_BAUD} baud...")
    ser = None
    serial_ok = False
    try:
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=SERIAL_TIMEOUT)
        time.sleep(2.0)  # 等待 Arduino 复位完成
        # 清空缓冲区
        ser.reset_input_buffer()
        serial_ok = True
        print(f"[信息] 串口 {SERIAL_PORT} 已打开，速率 {SERIAL_BAUD} baud")
    except serial.SerialException as e:
        print(f"[警告] 无法打开串口 {SERIAL_PORT}: {e}")
        print("        将仅运行视觉检测模式，不会发送数据到 Arduino。")
        print("        请确认：① 端口名是否正确 ② 是否有权限 (/dev/tty*) ③ 设备是否被占用")

    # ────────────────────────────────────────────────────────────────
    # 5. 主循环：读帧 → ArUco检测 → 位姿估计 → 发送串口
    # ────────────────────────────────────────────────────────────────
    print("\n[信息] 开始视觉定位。按 'q' 键退出，按 's' 切换显示模式。\n")
    last_send_time = 0.0
    frame_count = 0
    marker_found_count = 0

    while not _shutdown_flag:
        ret, frame = cap.read()
        if not ret:
            print("[警告] 读取帧失败，重试...")
            time.sleep(0.01)
            continue

        frame_count += 1

        # ── 鱼眼校正 ──
        if USE_FISHEYE:
            frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

        # ── 灰度转换 + ArUco 检测 ──
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)

        distance = None
        angle = None

        if ids is not None and len(ids) > 0:
            marker_found_count += 1

            # ── 位姿估计：PnP 求解 ──
            # cv2.aruco.estimatePoseSingleMarkers 使用 PnP 算法，
            # 根据 4 个角点的图像坐标和世界坐标，求解标签的旋转向量 rvec 和平移向量 tvec
            #
            # 返回：
            #   rvecs — 旋转向量（Rodrigues 形式），shape (N, 1, 3)
            #   tvecs — 平移向量，shape (N, 1, 3)，单位：米（与 MARKER_SIZE 一致）
            #   _objPoints — 标签角点的 3D 世界坐标
            try:
                rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(
                    corners,
                    MARKER_SIZE,
                    cam_matrix,
                    dist_coeffs
                )
            except Exception as e:
                print(f"[警告] PnP 位姿估计失败: {e}")
                # 仍然绘制检测到的标记框（无位姿信息）
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                continue

            # ── 2D 偏差计算 ──
            # 只处理第一个检测到的标签（单标签定位场景）
            tvec = tvecs[0]
            rvec = rvecs[0]
            distance, angle = compute_2d_error(tvec)

            # ── 合理性校验 ──
            if not is_valid_measurement(distance):
                # 测量结果异常，跳过本次发送但不影响可视化
                pass
            else:
                # ── 绘制检测结果 ──
                # 在画面中画出标签的坐标轴（红=X, 绿=Y, 蓝=Z）
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                cv2.drawFrameAxes(frame, cam_matrix, dist_coeffs,
                                  rvec, tvec, MARKER_SIZE * 0.5)

                # ── 在图像上叠加文字信息 ──
                info_lines = [
                    f"ID: {ids[0][0]}",
                    f"Dist: {distance:.2f}m",
                    f"Angle: {angle:+.1f} deg",
                ]
                y0 = 30
                for i, txt in enumerate(info_lines):
                    cv2.putText(frame, txt, (10, y0 + i * 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 0), 2)

                # ── 通过串口发送偏差数据给 Arduino ──
                if serial_ok and ser is not None and ser.is_open:
                    now = time.time()
                    if now - last_send_time >= SEND_INTERVAL:
                        # 数据帧格式：<距离,角度>
                        # 距离：米，保留 2 位小数
                        # 角度：度，保留 1 位小数，带正负号
                        frame_data = f"<{distance:.2f},{angle:.1f}>\n"
                        try:
                            ser.write(frame_data.encode('utf-8'))
                            ser.flush()  # 确保立即发送
                            last_send_time = now

                            # 控制台打印
                            direction = "右" if angle > 0 else ("左" if angle < 0 else "正前")
                            print(f"\r[发送] <{distance:.2f},{angle:+.1f}>  "
                                  f"| 标签{ids[0][0]} 偏{direction}  "
                                  f"| 帧#{frame_count}",
                                  end='', flush=True)
                        except serial.SerialException as e:
                            print(f"\n[错误] 串口写入失败: {e}")
                            serial_ok = False
        else:
            # 未检测到标签
            cv2.putText(frame, "No marker found", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # ── 显示画面 ──
        if DISPLAY_WINDOW:
            cv2.imshow('ArUco Visual Binding', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[信息] 收到 'q' 指令，退出程序。")
                break
            elif key == ord('s'):
                DISPLAY_WINDOW = not DISPLAY_WINDOW
                if DISPLAY_WINDOW:
                    cv2.namedWindow('ArUco Visual Binding')
                else:
                    cv2.destroyAllWindows()
                    print("[信息] 显示窗口已关闭（按 's' 重新打开）")

    # ────────────────────────────────────────────────────────────────
    # 6. 清理资源
    # ────────────────────────────────────────────────────────────────
    print(f"\n[统计] 共处理 {frame_count} 帧，检测到标签 {marker_found_count} 次")
    print("[信息] 正在释放资源...")
    cap.release()
    if ser is not None and ser.is_open:
        ser.close()
    cv2.destroyAllWindows()
    print("[信息] 程序已安全退出。")


# ===========================================================================
# 程序入口
# ===========================================================================

if __name__ == "__main__":
    main()
