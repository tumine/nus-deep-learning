"""
TCP 客户端 — 猫品种分类
========================

通过 TCP 自定义协议与推理服务器通信：
  发送 JPEG 图片 → 接收 JSON 分类结果

协议：
  请求: [4 bytes: 图像长度][N bytes: JPEG 图像数据]
  响应: [4 bytes: JSON 长度][N bytes: JSON 分类结果]

用法：
    # 单张图片
    python deploy/tcp_client.py --image cat.jpg

    # 整文件夹批量发送
    python deploy/tcp_client.py --dir ./test_images/

    # 指定服务器地址
    python deploy/tcp_client.py --image cat.jpg --host 192.168.1.100 --port 9000

    # Webcam 实时模式（OpenCV）
    python deploy/tcp_client.py --webcam

    # 输出结果到 JSON
    python deploy/tcp_client.py --image cat.jpg --output result.json
"""

import argparse
import json
import logging
import socket
import struct
import sys
import time
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("tcp-breed-client")

# ============================================================
# 协议常量（与服务器一致）
# ============================================================

HEADER_SIZE = 4
MAX_RESULT_SIZE = 64 * 1024    # 结果最大 64KB
RECV_TIMEOUT = 30.0


# ============================================================
# 协议编解码
# ============================================================

def pack_image(image_bytes: bytes) -> bytes:
    """打包图片：4 字节长度前缀 + JPEG 数据。"""
    return struct.pack("!I", len(image_bytes)) + image_bytes


def unpack_result(data: bytes) -> dict:
    """解包 JSON 结果。"""
    if len(data) < HEADER_SIZE:
        raise ValueError(f"响应太短: {len(data)} 字节")
    msg_len = struct.unpack("!I", data[:HEADER_SIZE])[0]
    if msg_len > MAX_RESULT_SIZE:
        raise ValueError(f"结果过大: {msg_len} 字节")
    return json.loads(data[HEADER_SIZE:HEADER_SIZE + msg_len])


def recv_exactly(sock: socket.socket, n: int) -> bytes:
    """精确接收 n 字节。"""
    sock.settimeout(RECV_TIMEOUT)
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError(f"连接中断：期望 {n} 字节，仅收到 {len(data)} 字节")
        data += chunk
    return data


# ============================================================
# TCP 客户端
# ============================================================

class TcpBreedClient:
    """猫品种分类 TCP 客户端。

    支持两种模式：
      - 短连接: 每次 predict() 新建连接、用完即断
      - 长连接: 使用 with 语句保持连接复用

    Usage::

        # 短连接模式
        client = TcpBreedClient("192.168.1.100", 9000)
        result = client.predict_file("cat.jpg")
        print(result["class_name_cn"], result["confidence"])

        # 长连接模式（批量图片时效率更高）
        with TcpBreedClient("192.168.1.100", 9000) as client:
            for img in ["cat1.jpg", "cat2.jpg", "cat3.jpg"]:
                result = client.predict_file(img)
                print(result["class_name_cn"])
    """

    def __init__(self, host: str = "localhost", port: int = 9000):
        self.host = host
        self.port = port
        self._sock: Optional[socket.socket] = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()

    # ----------------------------------------------------------
    # 连接管理
    # ----------------------------------------------------------

    def connect(self) -> None:
        """建立 TCP 连接。"""
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(RECV_TIMEOUT)
        self._sock.connect((self.host, self.port))
        logger.info(f"已连接: {self.host}:{self.port}")

    def close(self) -> None:
        """关闭连接。"""
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
            logger.debug("连接已关闭")

    @property
    def is_connected(self) -> bool:
        return self._sock is not None

    # ----------------------------------------------------------
    # 推理 API
    # ----------------------------------------------------------

    def predict(self, image_bytes: bytes) -> dict:
        """发送图片字节流，返回分类结果。

        Args:
            image_bytes: JPEG 编码的图片数据

        Returns:
            {
                "class_id": 0,
                "class_name": "ragdoll",
                "class_name_cn": "布偶猫",
                "confidence": 0.9521,
                "top5": [...],
                "latency_ms": 3.2,
                "server_time_ms": 1690000000000
            }
        """
        was_connected = self.is_connected
        if not was_connected:
            self.connect()

        try:
            # 1. 发送图片
            self._sock.sendall(pack_image(image_bytes))

            # 2. 接收响应
            #    先读 4 字节头部 → 得到 JSON 长度 → 读取完整 JSON
            header = recv_exactly(self._sock, HEADER_SIZE)
            result_len = struct.unpack("!I", header)[0]

            if result_len > MAX_RESULT_SIZE:
                raise ValueError(f"结果过大: {result_len} 字节")

            json_bytes = recv_exactly(self._sock, result_len)
            result = json.loads(json_bytes)

            # 检查是否有错误
            if "error" in result:
                logger.error(f"服务器返回错误: {result['error']}")
                return result

            return result

        finally:
            if not was_connected:
                self.close()

    def predict_file(self, image_path: str) -> dict:
        """从文件路径读取图片并推理。"""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {image_path}")

        with open(path, "rb") as f:
            image_bytes = f.read()

        logger.info(f"发送: {path.name} ({len(image_bytes)} 字节)")
        return self.predict(image_bytes)

    def predict_frame(self, frame) -> dict:
        """从 OpenCV frame 推理（numpy array BGR→RGB→JPEG）。"""
        import cv2
        import numpy as np

        if isinstance(frame, np.ndarray):
            # BGR → RGB → JPEG
            if frame.shape[-1] == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
            _, jpeg = cv2.imencode(".jpg", frame_rgb)
            image_bytes = jpeg.tobytes()
        else:
            raise TypeError(f"不支持的 frame 类型: {type(frame)}")

        return self.predict(image_bytes)


# ============================================================
# 结果格式化
# ============================================================

def print_result(result: dict, image_name: str = ""):
    """格式化打印分类结果。"""
    if "error" in result:
        print(f"\n❌ [{image_name}] 错误: {result['error']}")
        return

    print(f"\n{'='*55}")
    if image_name:
        print(f"📷 {image_name}")
    print(f"  🐱 识别结果: {result['class_name_cn']} ({result['class_name']})")
    print(f"  📊 置信度:   {result['confidence']*100:.2f}%")
    print(f"  ⏱️  推理耗时: {result['latency_ms']:.2f} ms")
    print(f"  🕐 服务时间: {result.get('server_time_ms', 'N/A')}")
    print(f"\n  Top-5 概率分布:")
    for item in result.get("top5", []):
        bar = "█" * max(1, int(item["probability"] * 30))
        marker = " ←" if item["rank"] == 1 else ""
        print(f"    {item['rank']}. {item['class_name_cn']:<8} "
              f"{bar} {item['probability']*100:.1f}%{marker}")
    print(f"{'='*55}")


# ============================================================
# CLI 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="猫品种分类 TCP 客户端",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 单张图片
  python deploy/tcp_client.py --image cat.jpg

  # 整文件夹
  python deploy/tcp_client.py --dir ./test_images/

  # 远程服务器
  python deploy/tcp_client.py --image cat.jpg --host 192.168.1.100 --port 9000

  # Webcam 实时
  python deploy/tcp_client.py --webcam --host 192.168.1.100
        """,
    )
    parser.add_argument("--host", default="localhost", help="服务器地址 (默认 localhost)")
    parser.add_argument("--port", type=int, default=9000, help="服务器端口 (默认 9000)")
    parser.add_argument("--image", type=str, help="单张图片路径")
    parser.add_argument("--dir", type=str, help="图片文件夹（批量发送）")
    parser.add_argument("--webcam", action="store_true", help="启用摄像头实时模式")
    parser.add_argument("--output", type=str, help="结果保存为 JSON 文件")
    parser.add_argument("--interval", type=float, default=0.0,
                        help="批量模式每张间隔（秒），避免服务器过载")
    args = parser.parse_args()

    # ---- 摄像头实时模式 ----
    if args.webcam:
        run_webcam(args)
        return

    # ---- 单张图片 ----
    if args.image:
        client = TcpBreedClient(args.host, args.port)
        try:
            result = client.predict_file(args.image)
            print_result(result, Path(args.image).name)
            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print(f"结果已保存: {args.output}")
        except Exception as e:
            logger.error(f"失败: {e}")
        return

    # ---- 文件夹批量 ----
    if args.dir:
        dir_path = Path(args.dir)
        if not dir_path.is_dir():
            print(f"文件夹不存在: {dir_path}")
            sys.exit(1)

        image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        image_paths = sorted(
            p for p in dir_path.iterdir()
            if p.suffix.lower() in image_exts
        )

        if not image_paths:
            print(f"文件夹中没有图片: {dir_path}")
            return

        print(f"找到 {len(image_paths)} 张图片")
        print(f"服务器: {args.host}:{args.port}")

        all_results = []
        success = 0
        fail = 0
        t_start = time.time()

        # 使用长连接批量发送
        with TcpBreedClient(args.host, args.port) as client:
            for i, img_path in enumerate(image_paths):
                try:
                    result = client.predict_file(str(img_path))
                    if "error" in result:
                        fail += 1
                        logger.warning(f"[{i+1}/{len(image_paths)}] {img_path.name}: {result['error']}")
                    else:
                        success += 1
                        logger.info(
                            f"[{i+1}/{len(image_paths)}] {img_path.name} → "
                            f"{result['class_name_cn']} ({result['confidence']*100:.1f}%)"
                        )
                    all_results.append({
                        "file": str(img_path),
                        "result": result,
                    })
                except Exception as e:
                    fail += 1
                    logger.error(f"[{i+1}/{len(image_paths)}] {img_path.name}: {e}")
                    all_results.append({
                        "file": str(img_path),
                        "error": str(e),
                    })

                if args.interval > 0:
                    time.sleep(args.interval)

        elapsed = time.time() - t_start

        print(f"\n{'='*55}")
        print(f"批量完成: {len(image_paths)} 张, 成功 {success}, 失败 {fail}")
        print(f"总耗时: {elapsed:.1f}s, 平均 {elapsed/len(image_paths)*1000:.0f}ms/张")
        print(f"{'='*55}")

        if args.output:
            output = {
                "total": len(image_paths),
                "success": success,
                "fail": fail,
                "elapsed_seconds": round(elapsed, 2),
                "results": all_results,
            }
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            print(f"结果已保存: {args.output}")
        return

    parser.print_help()


def run_webcam(args):
    """摄像头实时推理模式（需要 OpenCV）。"""
    try:
        import cv2
    except ImportError:
        print("Webcam 模式需要 opencv-python: pip install opencv-python")
        sys.exit(1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        sys.exit(1)

    print(f"📹 摄像头已启动 → 服务器: {args.host}:{args.port}")
    print("   按 SPACE 拍照推理, 按 Q 退出")

    last_result = None
    font = cv2.FONT_HERSHEY_SIMPLEX

    with TcpBreedClient(args.host, args.port) as client:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            display = frame.copy()

            # 显示上一次的结果
            if last_result and "error" not in last_result:
                text = f"{last_result['class_name_cn']}: {last_result['confidence']*100:.1f}%"
                cv2.putText(display, text, (10, 30), font, 0.8, (0, 255, 0), 2)

            cv2.imshow("Cat Breed Client (SPACE=predict, Q=quit)", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):  # 空格键拍照推理
                print("\n📸 拍照推理...")
                result = client.predict_frame(frame)
                last_result = result
                print_result(result, "webcam")
                print("   按 SPACE 再次拍照, 按 Q 退出")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
