from flask import Flask, Response, jsonify, request
from picamera2 import Picamera2
import cv2
import time
import requests
import os

app = Flask(__name__)

# ========== 添加 CORS 支持 ==========
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST')
    return response
# ====================================

# ---------- 目标电脑接收服务的地址 ----------
# TARGET_URL_UPLOAD = "http://100.78.156.12:5001/upload"       # 用于接收视频文件
TARGET_URL_STREAM = "http://100.70.117.52:5001/process_stream" # 用于接收视频流通知 (如果是方案B)

# 初始化摄像头
print("正在初始化 Picamera2...")
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (640, 360)})
picam2.configure(config)
picam2.start()
time.sleep(1.0)
print("摄像头就绪！")

# ==============================================================================
# 基础功能：提供实时的 MJPEG 视频流（模型端也可以直接填这个地址来获取实时视频）
# ==============================================================================
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
    """
    流媒体路由。如果在浏览器或者 OpenCV 中打开 http://树莓派IP:5000/video_feed，
    就能看到实时视频。
    """
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ==============================================================================
# 方案 B：告知目标电脑读取实时流 (对应拉流模型)
# ==============================================================================
@app.route('/notify_stream', methods=['GET'])
def notify_stream():
    """
    小车转弯后调用此接口。它只发一个 JSON 告诉模型端：“我转弯了，请抓取视频流并告诉我结果”。
    """
    try:
        # 告诉模型端，视频流的地址在哪里
        payload = {
            "event": "turn_completed",
            # 注意把这里的 IP 换成树莓派当前的局域网/虚拟网 IP 
            "stream_url": "http://100.84.2.68:5000/video_feed" 
        }
        
        print("📡 正在通知模型端拉取视频流...")
        response = requests.post(TARGET_URL_STREAM, json=payload, timeout=20)
        
        if response.status_code == 200:
            result = response.json()
            return jsonify({"status": "success", "msg": "模型已分析流", "data": result})
        else:
            return jsonify({"status": "error", "msg": f"目标端错误: {response.status_code}"}), 500
            
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)}), 500


if __name__ == '__main__':
    # threaded=True 确保在推流的同时，还能响应上传请求
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)