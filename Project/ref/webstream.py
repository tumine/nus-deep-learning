from flask import Flask, Response, jsonify   # 添加了 jsonify
from picamera2 import Picamera2
import cv2
import time
import io
import requests   # 添加 requests

app = Flask(__name__)

# ========== 添加 CORS 支持 ==========
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST')
    return response
# ====================================

# ---------- 目标电脑接收服务的地址（记得加 /upload） ----------
TARGET_URL = "http://100.78.156.12:5001/upload"

# 初始化摄像头
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (640,360)})
picam2.configure(config)
picam2.start()
time.sleep(0.5)

def generate_frames():
    while True:
        frame_rgb = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        ret, buffer = cv2.imencode('.jpg', frame_bgr)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/download_screenshot', methods=['GET'])
def download_screenshot():
    """捕获图片，通过 HTTP POST 发送到目标电脑"""
    try:
        # 1. 捕获一帧
        frame_rgb = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        ret, buffer = cv2.imencode('.jpg', frame_bgr)
        if not ret:
            return jsonify({"status": "error", "msg": "编码失败"}), 500

        # 2. 将图片数据转为字节
        img_bytes = buffer.tobytes()

        # 3. 构造 multipart/form-data 文件上传
        files = {
            'image': ('screenshot.jpg', img_bytes, 'image/jpeg')
        }

        # 4. 发送 POST 请求到目标电脑
        response = requests.post(TARGET_URL, files=files, timeout=5)

        # 5. 检查目标电脑返回的结果
        if response.status_code == 200:
            result = response.json()
            return jsonify({"status": "success", "msg": result.get("msg", "已保存")})
        else:
            return jsonify({"status": "error", "msg": f"接收端返回 {response.status_code}"}), 500

    except requests.exceptions.ConnectionError:
        return jsonify({"status": "error", "msg": "无法连接到目标电脑，请检查IP和端口"}), 500
    except requests.exceptions.Timeout:
        return jsonify({"status": "error", "msg": "连接超时"}), 500
    except Exception as e:
        print(f"❌ 截图错误: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "msg": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)