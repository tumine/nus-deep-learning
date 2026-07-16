"""
中继服务器 — 接收图片并转发到推理服务器
==========================================

功能：
  1. 通过 Flask HTTP 接收树莓派发来的图片（继承 receiver.py 的逻辑）
  2. 将收到的图片通过 TCP 协议转发到推理服务器（使用 tcp_client.py 的 TcpBreedClient）
  3. 接收推理服务器的分类结果，打印到控制台

用法：
    python relay_server.py

    # 指定 TCP 服务器地址
    python relay_server.py --host 192.168.1.100 --port 9000

    # 指定 HTTP 监听端口
    python relay_server.py --http-port 5001

    # 指定图片保存目录
    python relay_server.py --save-dir ./img_recv
"""

import argparse
import os
import socket
import sys
import threading
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify

# 导入 TCP 客户端
sys.path.insert(0, str(Path(__file__).parent / "deploy"))
from tcp_client import TcpBreedClient, print_result

app = Flask(__name__)

# ============================================================
# 全局配置（由 main() 设置）
# ============================================================

SAVE_DIR: str = r"C:\Documents\nus-deep-learning\Baseline\img_recv"
TCP_HOST: str = "127.0.0.1"
TCP_PORT: int = 9000
TCP_CLIENT: TcpBreedClient = None
TCP_LOCK = threading.Lock()     # 保护 TCP_CLIENT 的线程安全
MAX_RETRIES: int = 1            # 连接失败后的最大重试次数


# ============================================================
# TCP 转发辅助
# ============================================================

def _is_connection_error(exc: Exception) -> bool:
    """判断异常是否是 TCP 连接断开导致的。"""
    if isinstance(exc, (ConnectionError, ConnectionResetError,
                         ConnectionAbortedError, ConnectionRefusedError,
                         BrokenPipeError, socket.timeout)):
        return True
    # Windows: WinError 10053 / 10054 等被封装为 OSError
    if isinstance(exc, OSError):
        return True
    return False


def _reconnect_tcp_client():
    """关闭当前 TCP 连接并重新连接。"""
    global TCP_CLIENT
    print(f"[TCP] 尝试重新连接到 {TCP_HOST}:{TCP_PORT} ...")
    try:
        TCP_CLIENT.close()
    except Exception:
        pass
    TCP_CLIENT = TcpBreedClient(TCP_HOST, TCP_PORT)
    TCP_CLIENT.connect()
    print(f"[TCP] 重新连接成功")


def forward_to_tcp(image_bytes: bytes, filename: str) -> dict:
    """将图片转发到 TCP 推理服务器，带自动重连+重试。

    如果连接断开，会自动重连并重试一次。
    返回的 dict 包含结果或错误信息，可直接用于 HTTP 响应。
    """
    global TCP_CLIENT

    print(f"[TCP] 转发图片到 {TCP_HOST}:{TCP_PORT} ...")

    for attempt in range(MAX_RETRIES + 1):
        try:
            result = TCP_CLIENT.predict(image_bytes)

            if "error" in result:
                print(f"[TCP] 推理失败: {result['error']}")
                return {
                    "status": "partial",
                    "msg": f"Saved as {filename}, but inference failed: {result['error']}",
                    "filename": filename,
                }, 200
            else:
                print_result(result, filename)
                return {
                    "status": "success",
                    "msg": f"Saved as {filename}",
                    "filename": filename,
                    "result": result,
                }, 200

        except Exception as e:
            if attempt < MAX_RETRIES and _is_connection_error(e):
                print(f"[TCP] 连接已断开 ({e})，正在重连...")
                with TCP_LOCK:
                    _reconnect_tcp_client()
                continue
            else:
                print(f"[TCP] 转发失败: {e}")
                return {
                    "status": "partial",
                    "msg": f"Saved as {filename}, but forwarding failed: {e}",
                    "filename": filename,
                }, 200


# ============================================================
# Flask 路由 — 接收图片
# ============================================================

@app.route('/upload', methods=['POST'])
def upload_image():
    """接收树莓派发来的图片，保存后转发到推理服务器。"""
    try:
        # 1. 检查是否有文件
        if 'image' not in request.files:
            return jsonify({"status": "error", "msg": "No image file"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"status": "error", "msg": "Empty filename"}), 400

        # 2. 保存图片到本地
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.jpg"
        save_path = os.path.join(SAVE_DIR, filename)

        file.save(save_path)
        print(f"[HTTP] 收到图片并保存: {save_path}")

        # 3. 读取图片字节流
        with open(save_path, "rb") as f:
            image_bytes = f.read()

        # 4. 转发到 TCP 推理服务器（带自动重连）
        with TCP_LOCK:
            result_tuple = forward_to_tcp(image_bytes, filename)

        return jsonify(result_tuple[0]), result_tuple[1]

    except Exception as e:
        print(f"[HTTP] 保存失败: {e}")
        return jsonify({"status": "error", "msg": str(e)}), 500


# ============================================================
# 入口
# ============================================================

def main():
    global SAVE_DIR, TCP_HOST, TCP_PORT, TCP_CLIENT

    parser = argparse.ArgumentParser(
        description="中继服务器：HTTP 接收图片 → TCP 转发推理 → 打印结果",
    )
    parser.add_argument("--host", default="localhost",
                        help="TCP 推理服务器地址 (默认 localhost)")
    parser.add_argument("--port", type=int, default=9000,
                        help="TCP 推理服务器端口 (默认 9000)")
    parser.add_argument("--http-port", type=int, default=5001,
                        help="HTTP 监听端口 (默认 5001)")
    parser.add_argument("--save-dir", type=str,
                        default=str(Path(__file__).parent / "img_recv"),
                        help="图片保存目录 (默认 ./img_recv)")
    args = parser.parse_args()

    # 设置全局变量
    SAVE_DIR = args.save_dir
    TCP_HOST = args.host
    TCP_PORT = args.port

    # 确保保存目录存在
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 创建 TCP 客户端（长连接模式）
    TCP_CLIENT = TcpBreedClient(TCP_HOST, TCP_PORT)
    TCP_CLIENT.connect()
    print(f"[TCP] 已连接到推理服务器: {TCP_HOST}:{TCP_PORT}")

    print(f"[HTTP] 图片保存目录: {SAVE_DIR}")
    print(f"[HTTP] 中继服务器启动: http://0.0.0.0:{args.http_port}")
    print("=" * 55)

    try:
        app.run(host='0.0.0.0', port=args.http_port, debug=False)
    finally:
        TCP_CLIENT.close()
        print("[TCP] 连接已关闭")


if __name__ == '__main__':
    main()
