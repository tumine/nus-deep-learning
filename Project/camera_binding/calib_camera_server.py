"""
calib_camera_server.py
======================
树莓派相机标定图片采集服务器（程序 A）

功能：
  1. 通过 Flask 提供网页界面
  2. 网页上实时显示摄像头画面（MJPEG 流）
  3. 点击"拍照"按钮即可拍摄并保存标定图片到本地

用法：
    python calib_camera_server.py
    python calib_camera_server.py --port 8080 --camera 0 --resolution 1280x720 --save-dir ./calib_captures
"""

from __future__ import annotations

import argparse
import time
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, render_template_string, jsonify

# =============================================================================
# 全局状态
# =============================================================================

# 摄像头帧（线程共享）
_frame_lock = threading.Lock()
_latest_frame = None  # raw RGB numpy array

# 拍照计数
_capture_count = 0
_capture_count_lock = threading.Lock()

# 保存目录
_save_dir: Path = Path("./calib_captures")

# =============================================================================
# HTML 页面模板
# =============================================================================

PAGE_HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>相机标定图片采集</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #1a1a2e;
    color: #eee;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 20px;
  }

  h1 {
    font-size: 1.5rem;
    font-weight: 600;
    margin-bottom: 16px;
    color: #e0e0e0;
    letter-spacing: 0.5px;
  }

  .stream-container {
    position: relative;
    border: 2px solid #333;
    border-radius: 12px;
    overflow: hidden;
    background: #000;
    max-width: 960px;
    width: 100%;
    box-shadow: 0 4px 24px rgba(0,0,0,0.5);
  }

  .stream-container img {
    display: block;
    width: 100%;
    height: auto;
  }

  .controls {
    display: flex;
    align-items: center;
    gap: 20px;
    margin-top: 20px;
    flex-wrap: wrap;
    justify-content: center;
  }

  .btn-capture {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 14px 36px;
    font-size: 1.1rem;
    font-weight: 600;
    color: #fff;
    background: linear-gradient(135deg, #e94560, #c23152);
    border: none;
    border-radius: 10px;
    cursor: pointer;
    transition: transform 0.15s, box-shadow 0.15s, opacity 0.15s;
    box-shadow: 0 4px 14px rgba(233,69,96,0.35);
    user-select: none;
    -webkit-tap-highlight-color: transparent;
  }

  .btn-capture:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(233,69,96,0.5);
  }

  .btn-capture:active {
    transform: scale(0.96);
  }

  .btn-capture:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    transform: none;
  }

  .info-card {
    background: #16213e;
    border: 1px solid #333;
    border-radius: 10px;
    padding: 12px 24px;
    text-align: center;
    min-width: 120px;
  }

  .info-card .label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #888;
  }

  .info-card .value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #53d8fb;
  }

  .toast {
    position: fixed;
    bottom: 32px;
    left: 50%;
    transform: translateX(-50%) translateY(100px);
    background: #0f3460;
    color: #53d8fb;
    padding: 12px 28px;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.95rem;
    opacity: 0;
    transition: transform 0.3s ease, opacity 0.3s ease;
    z-index: 999;
    box-shadow: 0 4px 16px rgba(0,0,0,0.4);
    pointer-events: none;
  }

  .toast.show {
    transform: translateX(-50%) translateY(0);
    opacity: 1;
  }

  .status-dot {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #4ade80;
    margin-right: 6px;
    animation: pulse 1.5s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }

  .footer {
    margin-top: 24px;
    font-size: 0.8rem;
    color: #555;
  }
</style>
</head>
<body>

  <h1><span class="status-dot"></span>相机标定图片采集</h1>

  <div class="stream-container">
    <img id="stream" src="/video_feed" alt="摄像头实时画面"
         onerror="this.style.display='none'; document.getElementById('stream-error').style.display='block';">
    <div id="stream-error" style="display:none; padding:40px; text-align:center; color:#e94560;">
      ⚠ 无法连接摄像头，请检查摄像头是否已连接
    </div>
  </div>

  <div class="controls">
    <button class="btn-capture" id="btn-capture" onclick="capture()">
      📷 拍照
    </button>
    <div class="info-card">
      <div class="label">已拍摄</div>
      <div class="value" id="count">0</div>
    </div>
  </div>

  <div class="toast" id="toast"></div>

  <div class="footer">
    图片保存目录：{{ save_dir }}
  </div>

  <script>
    let capturing = false;

    async function capture() {
      if (capturing) return;
      capturing = true;
      const btn = document.getElementById('btn-capture');
      btn.disabled = true;
      btn.textContent = '⏳ 拍摄中...';

      try {
        const resp = await fetch('/capture');
        const data = await resp.json();
        if (data.status === 'ok') {
          document.getElementById('count').textContent = data.count;
          showToast('✅ ' + data.filename + ' 保存成功');
        } else {
          showToast('❌ ' + data.msg);
        }
      } catch (e) {
        showToast('❌ 请求失败：' + e.message);
      }

      btn.disabled = false;
      btn.textContent = '📷 拍照';
      capturing = false;
    }

    function showToast(msg) {
      const toast = document.getElementById('toast');
      toast.textContent = msg;
      toast.classList.add('show');
      clearTimeout(toast._timeout);
      toast._timeout = setTimeout(() => toast.classList.remove('show'), 2500);
    }

    // 页面加载时查询当前拍摄数量
    fetch('/count').then(r => r.json()).then(d => {
      document.getElementById('count').textContent = d.count;
    });
  </script>
</body>
</html>"""

# =============================================================================
# Flask 应用
# =============================================================================

app = Flask(__name__)


@app.route("/")
def index():
    """返回主页面"""
    return render_template_string(PAGE_HTML, save_dir=str(_save_dir.resolve()))


@app.route("/video_feed")
def video_feed():
    """MJPEG 视频流 — 内部帧为 RGB，实时转换为 BGR 后编码为 JPEG"""

    def generate():
        while True:
            with _frame_lock:
                frame = _latest_frame
            if frame is None:
                # 还没有帧，发一个小的占位图像
                placeholder = _make_placeholder()
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" +
                       placeholder + b"\r\n")
            else:
                # 参考 videostream.py：RGB → BGR → JPEG 编码
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                _, jpeg = cv2.imencode(".jpg", frame_bgr,
                                       [cv2.IMWRITE_JPEG_QUALITY, 85])
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" +
                       jpeg.tobytes() + b"\r\n")
            time.sleep(0.04)  # ~25 fps

    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/capture", methods=["POST", "GET"])
def capture():
    """拍摄并保存一张照片"""
    global _capture_count

    with _frame_lock:
        frame = _latest_frame

    if frame is None:
        return jsonify({"status": "error", "msg": "摄像头未就绪"}), 503

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"calib_{timestamp}.jpg"
    save_path = _save_dir / filename

    try:
        # 参考 videostream.py：RGB → BGR → JPEG 编码后保存
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        _, jpeg = cv2.imencode(".jpg", frame_bgr,
                               [cv2.IMWRITE_JPEG_QUALITY, 85])
        with open(save_path, "wb") as f:
            f.write(jpeg.tobytes())
    except OSError as e:
        return jsonify({"status": "error", "msg": f"写入失败: {e}"}), 500

    with _capture_count_lock:
        _capture_count += 1
        count = _capture_count

    print(f"[拍照] {filename} 已保存 (总计: {count})")

    return jsonify({
        "status": "ok",
        "filename": filename,
        "count": count,
        "path": str(save_path.resolve()),
    })


@app.route("/count")
def get_count():
    """返回当前已拍摄张数"""
    with _capture_count_lock:
        count = _capture_count
    return jsonify({"count": count})


# =============================================================================
# 辅助函数
# =============================================================================

def _make_placeholder() -> bytes:
    """生成占位图像（摄像头未就绪时显示）"""
    img = 128 * np.ones((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, "Waiting for camera...", (100, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes()


def _camera_thread(camera_index: int, width: int, height: int):
    """后台线程：持续从摄像头读取帧并编码为 JPEG

    策略：
      1. 树莓派 CSI 摄像头 → 使用 picamera2（兼容现代 Raspberry Pi OS Bookworm）
      2. USB 摄像头 / 桌面环境 → 回退到 OpenCV cv2.VideoCapture
    """
    global _latest_frame

    cap = None
    picam2 = None

    # ── 方案 A：尝试 picamera2（树莓派 CSI 摄像头）──────────────────
    try:
        from picamera2 import Picamera2
        print("[摄像头] 检测到 picamera2，尝试使用 CSI 摄像头...")
        picam2 = Picamera2()
        video_config = picam2.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},
            controls={"FrameRate": 30},
        )
        picam2.configure(video_config)
        picam2.start()
        actual_w = video_config["main"]["size"][0]
        actual_h = video_config["main"]["size"][1]
        print(f"[摄像头] picamera2 已就绪, 分辨率: {actual_w}x{actual_h}")
    except Exception as e:
        print(f"[摄像头] picamera2 不可用 ({e})，回退到 OpenCV VideoCapture...")
        if picam2 is not None:
            try:
                picam2.stop()
                picam2.close()
            except Exception:
                pass
        picam2 = None

    # ── 方案 B：OpenCV VideoCapture（USB 摄像头 / 桌面）─────────────
    if picam2 is None:
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not cap.isOpened():
            print(f"[摄像头] ✗ 无法打开摄像头 (索引 {camera_index})")
            print(f"  排查建议：")
            print(f"    - ls /dev/video*  检查是否有摄像头设备")
            print(f"    - 树莓派请确认是否安装了 picamera2 (pip install picamera2)")
            print(f"    - 或尝试 --camera 1 等不同索引")
            return

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[摄像头] OpenCV VideoCapture 已就绪, 索引 {camera_index}, 分辨率: {actual_w}x{actual_h}")

    # ── 主循环：读取帧，统一存储为 RGB 格式 ──────────────────────────
    while True:
        try:
            if picam2 is not None:
                # picamera2 配置为 RGB888，capture_array() 直接返回 RGB
                frame = picam2.capture_array()
                if frame is None or frame.size == 0:
                    time.sleep(0.01)
                    continue
                # frame 已经是 RGB，无需转换
            elif cap is not None:
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.1)
                    continue
                # OpenCV VideoCapture 返回 BGR，转为 RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                return  # 两边都不可用，退出线程
        except Exception as ex:
            print(f"[摄像头] 读取帧异常: {ex}")
            time.sleep(0.1)
            continue

        with _frame_lock:
            _latest_frame = frame

    # 清理
    if cap is not None:
        cap.release()
    if picam2 is not None:
        try:
            picam2.stop()
            picam2.close()
        except Exception:
            pass


# =============================================================================
# CLI 入口
# =============================================================================

def main():
    global _save_dir, _capture_count

    parser = argparse.ArgumentParser(
        description="树莓派相机标定图片采集服务器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python calib_camera_server.py
  python calib_camera_server.py --port 8080 --camera 0
  python calib_camera_server.py --resolution 1280x720 --save-dir ./my_calib
        """,
    )
    parser.add_argument("--port", type=int, default=8080,
                        help="HTTP 服务端口（默认 8080）")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="监听地址（默认 0.0.0.0）")
    parser.add_argument("--camera", type=int, default=0,
                        help="摄像头设备索引（默认 0）")
    parser.add_argument("--resolution", type=str, default="1280x720",
                        help="摄像头分辨率 WxH（默认 1280x720）")
    parser.add_argument("--save-dir", type=str, default="./calib_captures",
                        help="图片保存目录（默认 ./calib_captures）")
    args = parser.parse_args()

    # 解析分辨率
    try:
        w_str, h_str = args.resolution.split("x")
        width, height = int(w_str), int(h_str)
    except ValueError:
        print(f"[错误] 分辨率格式错误: {args.resolution}，应为 WxH 如 1280x720")
        return

    # 创建保存目录
    _save_dir = Path(args.save_dir).resolve()
    _save_dir.mkdir(parents=True, exist_ok=True)

    # 统计已有图片数
    _capture_count = len(list(_save_dir.glob("calib_*.jpg")))

    print("=" * 55)
    print("📷 树莓派相机标定图片采集服务器")
    print("=" * 55)
    print(f"  监听地址:     http://{args.host}:{args.port}")
    print(f"  摄像头索引:   {args.camera}")
    print(f"  分辨率:       {width}x{height}")
    print(f"  保存目录:     {_save_dir}")
    print(f"  已有图片:     {_capture_count} 张")
    print("=" * 55)

    # 在局域网中，还可通过 Tailscale IP 访问
    print(f"\n💡 提示: 可通过以下地址访问网页：")
    print(f"   本机:     http://localhost:{args.port}")
    print(f"   局域网:   http://<树莓派局域网IP>:{args.port}")
    print(f"   Tailscale: http://<树莓派Tailscale IP>:{args.port}")
    print()

    # 启动摄像头采集线程
    cam_thread = threading.Thread(
        target=_camera_thread,
        args=(args.camera, width, height),
        daemon=True,
    )
    cam_thread.start()

    # 等摄像头准备好
    time.sleep(1.5)

    # 启动 Flask
    try:
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n[信息] 收到中断信号，正在退出...")


if __name__ == "__main__":
    main()
