"""
TCP 推理服务 — 猫品种分类
===========================

基于 TCP Socket 自定义协议的推理服务。作为分布式架构中的
"云端 GPU 推理节点"，运行在笔记本电脑（RTX A2000）上。

功能：
  1. TCP 接收单张 JPEG 图片字节流
  2. ResNet-50 猫品种分类推理（毫秒级）
  3. TCP 返回 JSON 分类结果

协议设计：
  ┌──────────────────────────────────────┐
  │           请求（客户端 → 服务器）        │
  ├──────────┬───────────────────────────┤
  │ 4 bytes  │  N bytes                  │
  │ 图像长度  │  JPEG 图像数据             │
  │ (uint32) │                           │
  └──────────┴───────────────────────────┘

  ┌──────────────────────────────────────┐
  │           响应（服务器 → 客户端）        │
  ├──────────┬───────────────────────────┤
  │ 4 bytes  │  N bytes                  │
  │ JSON长度  │  JSON 分类结果             │
  │ (uint32) │                           │
  └──────────┴───────────────────────────┘

JSON 结果格式：
  {
    "class_id": 0,
    "class_name": "ragdoll",
    "class_name_cn": "布偶猫",
    "confidence": 0.9521,
    "top5": [
      {"rank": 1, "class_name": "ragdoll", "class_name_cn": "布偶猫", "probability": 0.9521},
      ...
    ],
    "latency_ms": 3.2,
    "server_time_ms": 1690000000000
  }

用法（服务器端）：
    python -m deploy.tcp_server --model best_model.pth --port 9000

    # ONNX 后端
    python -m deploy.tcp_server --model resnet50_cat.onnx --backend onnx --port 9000

    # 同时启动 HTTP 和 TCP 服务
    python -m deploy.tcp_server --model best_model.pth --tcp-port 9000 --http-port 8000
"""

import argparse
import json
import logging
import socket
import struct
import sys
import threading
import time
from pathlib import Path

# 将父目录加入 sys.path
_script_dir = Path(__file__).resolve().parent
if str(_script_dir.parent) not in sys.path:
    sys.path.insert(0, str(_script_dir.parent))

from deploy.inference import CatBreedClassifier, get_classifier

# ============================================================
# 日志
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("tcp-breed-server")


# ============================================================
# 自定义 TCP 协议
# ============================================================

# 协议常量
HEADER_SIZE = 4          # uint32 长度前缀
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 最大图片 10MB
RECV_TIMEOUT = 30.0      # 接收超时（秒）


def pack_message(payload: bytes) -> bytes:
    """打包消息：4 字节长度前缀 + 数据。"""
    return struct.pack("!I", len(payload)) + payload


def unpack_header(data: bytes) -> int:
    """解包 4 字节长度前缀，返回数据长度。"""
    if len(data) < HEADER_SIZE:
        raise ValueError(f"数据太短: {len(data)} 字节")
    return struct.unpack("!I", data[:HEADER_SIZE])[0]


def recv_exactly(sock: socket.socket, n: int, timeout: float = RECV_TIMEOUT) -> bytes:
    """从 socket 精确接收 n 个字节。

    Args:
        sock: TCP socket
        n: 期望接收的字节数
        timeout: 超时时间（秒）

    Returns:
        接收到的字节数据

    Raises:
        ConnectionError: 连接断开或超时
    """
    sock.settimeout(timeout)
    data = b""
    while len(data) < n:
        try:
            chunk = sock.recv(n - len(data))
        except socket.timeout:
            raise ConnectionError(f"接收超时（已等待 {timeout}s）")

        if not chunk:
            if not data:
                raise ConnectionError("客户端断开连接")
            raise ConnectionError(
                f"连接中断：期望 {n} 字节，仅收到 {len(data)} 字节"
            )
        data += chunk
    return data


def recv_image(sock: socket.socket) -> bytes:
    """从 socket 接收一张图片。

    协议：[4 bytes: 图像数据长度][N bytes: JPEG 图像数据]
    """
    # 1. 读取长度前缀
    header = recv_exactly(sock, HEADER_SIZE)
    img_len = unpack_header(header)

    # 2. 校验
    if img_len == 0:
        raise ValueError("图片长度为 0")
    if img_len > MAX_IMAGE_SIZE:
        raise ValueError(f"图片过大: {img_len} 字节（上限 {MAX_IMAGE_SIZE}）")

    # 3. 读取图片数据
    logger.debug(f"接收图片: {img_len} 字节")
    img_bytes = recv_exactly(sock, img_len)
    return img_bytes


def send_result(sock: socket.socket, result: dict) -> None:
    """发送 JSON 分类结果。

    协议：[4 bytes: JSON 长度][N bytes: JSON 数据]
    """
    json_bytes = json.dumps(result, ensure_ascii=False).encode("utf-8")
    sock.sendall(pack_message(json_bytes))


# ============================================================
# TCP 推理服务器
# ============================================================

class TcpBreedServer:
    """猫品种分类 TCP 推理服务器。

    特性：
      - 多线程并发处理：每个客户端分配独立线程
      - 连接池管理：自动清理断开的连接
      - 统计报告：定期输出请求数、延迟等指标
      - 优雅关闭：支持 SIGINT / SIGTERM
    """

    def __init__(
        self,
        model_path: str,
        backend: str = "pytorch",
        port: int = 9000,
        host: str = "0.0.0.0",
        max_clients: int = 10,
        confidence_threshold: float = 0.50,
    ):
        self.host = host
        self.port = port
        self.max_clients = max_clients
        self.backend = backend
        self.confidence_threshold = confidence_threshold

        # 加载分类器（使用缓存的单例）
        logger.info(f"加载模型: {model_path} [backend={backend}]")
        # 注意: get_classifier 使用缓存，如果已加载则不会更新 threshold
        # 如需更改 threshold，直接使用 CatBreedClassifier 构造
        from deploy.inference import CatBreedClassifier
        self.classifier = CatBreedClassifier(
            model_path=model_path,
            backend=backend,
            confidence_threshold=confidence_threshold,
        )
        self.classifier.load_model()
        logger.info(f"模型加载完成 (device={self.classifier.device}, "
                     f"拒识: {'✅ 支持' if self.classifier.has_other_class else '⚠️ 仅阈值兜底'})")

        # 运行状态
        self._running = False
        self._server_sock: socket.socket | None = None
        self._active_threads: list[threading.Thread] = []
        self._threads_lock = threading.Lock()

        # 统计
        self._stats = {
            "requests_total": 0,
            "requests_success": 0,
            "requests_error": 0,
            "total_latency_ms": 0.0,
            "start_time": 0.0,
        }
        self._stats_lock = threading.Lock()

    # ----------------------------------------------------------
    # 启动与停止
    # ----------------------------------------------------------

    def start(self) -> None:
        """启动 TCP 服务器。"""
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind((self.host, self.port))
        self._server_sock.listen(self.max_clients)
        self._server_sock.settimeout(1.0)  # 主循环可中断

        self._running = True
        self._stats["start_time"] = time.time()

        # 启动统计报告线程
        stats_thread = threading.Thread(target=self._stats_reporter, daemon=True)
        stats_thread.start()

        logger.info("=" * 60)
        logger.info(f"🐱 猫品种分类 TCP 推理服务")
        logger.info("=" * 60)
        logger.info(f"  地址:       tcp://{self.host}:{self.port}")
        logger.info(f"  模型后端:   {self.backend}")
        logger.info(f"  计算设备:   {self.classifier.device}")
        logger.info(f"  最大连接:   {self.max_clients}")
        logger.info(f"  置信度阈值: {self.confidence_threshold:.2f}")
        logger.info(f"  拒识能力:   {'✅ other类 + 阈值' if self.classifier.has_other_class else '⚠️  仅阈值兜底'}")
        logger.info(f"  协议:       长度前缀 + JPEG 图像 → JSON 结果")
        logger.info("=" * 60)
        logger.info("等待客户端连接... (Ctrl+C 停止)")

        try:
            self._accept_loop()
        except KeyboardInterrupt:
            logger.info("\n收到停止信号")
        finally:
            self.stop()

    def stop(self) -> None:
        """优雅关闭服务器。"""
        self._running = False

        # 关闭服务端 socket
        if self._server_sock:
            try:
                self._server_sock.close()
            except Exception:
                pass

        # 等待活跃线程结束
        with self._threads_lock:
            threads = list(self._active_threads)

        for t in threads:
            t.join(timeout=5.0)

        # 输出最终统计
        self._print_final_stats()
        logger.info("服务器已停止")

    # ----------------------------------------------------------
    # 连接处理
    # ----------------------------------------------------------

    def _accept_loop(self) -> None:
        """主循环：接受客户端连接。"""
        while self._running:
            try:
                client_sock, addr = self._server_sock.accept()
                logger.info(f"[连接] {addr[0]}:{addr[1]}")

                # 启动处理线程
                thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_sock, addr),
                    daemon=True,
                )
                thread.start()

                with self._threads_lock:
                    self._active_threads.append(thread)
                    # 清理已结束的线程
                    self._active_threads = [
                        t for t in self._active_threads if t.is_alive()
                    ]

            except socket.timeout:
                continue
            except OSError:
                if self._running:
                    logger.error("Socket 错误", exc_info=True)
                break

    def _handle_client(self, sock: socket.socket, addr: tuple) -> None:
        """处理单个客户端连接。

        循环接收图片 → 推理 → 返回结果，直到客户端断开。
        """
        client_label = f"{addr[0]}:{addr[1]}"

        try:
            while self._running:
                # 1. 接收图片
                img_bytes = recv_image(sock)

                # 2. 推理
                t0 = time.perf_counter()
                result = self.classifier.predict_from_bytes(img_bytes)
                latency = result.latency_ms

                # 3. 构建响应
                response = result.to_dict()
                response["server_time_ms"] = int(time.time() * 1000)
                response["request_id"] = self._stats["requests_total"]

                # 4. 发送结果
                send_result(sock, response)

                # 5. 更新统计
                with self._stats_lock:
                    self._stats["requests_total"] += 1
                    self._stats["requests_success"] += 1
                    self._stats["total_latency_ms"] += latency

                logger.info(
                    f"[推理] {client_label} → "
                    f"{result.class_name_cn} ({result.confidence*100:.1f}%) "
                    f"延迟={latency:.1f}ms"
                    + (" 🚫 非猫图片" if result.is_not_cat else "")
                )

        except ConnectionError as e:
            logger.info(f"[断开] {client_label}: {e}")
        except ValueError as e:
            logger.warning(f"[协议错误] {client_label}: {e}")
            with self._stats_lock:
                self._stats["requests_error"] += 1
            # 发送错误响应
            try:
                error_resp = {
                    "error": str(e),
                    "server_time_ms": int(time.time() * 1000),
                }
                send_result(sock, error_resp)
            except Exception:
                pass
        except Exception as e:
            logger.error(f"[错误] {client_label}: {e}", exc_info=True)
            with self._stats_lock:
                self._stats["requests_error"] += 1
        finally:
            try:
                sock.close()
            except Exception:
                pass

    # ----------------------------------------------------------
    # 统计
    # ----------------------------------------------------------

    def _stats_reporter(self) -> None:
        """定期输出统计报告。"""
        while self._running:
            time.sleep(15)
            with self._stats_lock:
                total = self._stats["requests_total"]
                success = self._stats["requests_success"]
                errors = self._stats["requests_error"]
                total_lat = self._stats["total_latency_ms"]
                uptime = time.time() - self._stats["start_time"]

                avg_lat = total_lat / total if total > 0 else 0
                qps = total / uptime if uptime > 0 else 0

            logger.info(
                f"[统计] 请求: {total} | 成功: {success} | 错误: {errors} | "
                f"平均延迟: {avg_lat:.1f}ms | QPS: {qps:.2f} | "
                f"运行: {uptime:.0f}s"
            )

    def _print_final_stats(self) -> None:
        """输出最终统计。"""
        with self._stats_lock:
            total = self._stats["requests_total"]
            success = self._stats["requests_success"]
            errors = self._stats["requests_error"]
            total_lat = self._stats["total_latency_ms"]
            uptime = time.time() - self._stats["start_time"]

        if total > 0:
            avg_lat = total_lat / total
            qps = total / uptime if uptime > 0 else 0
        else:
            avg_lat = 0
            qps = 0

        logger.info("\n" + "=" * 60)
        logger.info("运行统计")
        logger.info("=" * 60)
        logger.info(f"  运行时长:     {uptime:.0f}s")
        logger.info(f"  总请求数:     {total}")
        logger.info(f"  成功:         {success}")
        logger.info(f"  错误:         {errors}")
        logger.info(f"  平均延迟:     {avg_lat:.1f}ms")
        logger.info(f"  平均 QPS:     {qps:.2f}")
        logger.info("=" * 60)


# ============================================================
# 便捷函数：起一个推理服务
# ============================================================

def serve(
    host: str = "0.0.0.0",
    port: int = 9000,
    model_path: str = "best_model.pth",
    confidence_threshold: float = 0.50,
) -> None:
    """一键启动 TCP 推理服务（供外部脚本调用）。"""
    server = TcpBreedServer(
        model_path=model_path, port=port, host=host,
        confidence_threshold=confidence_threshold,
    )
    server.start()


# ============================================================
# CLI 入口
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="猫品种分类 — TCP 推理服务器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 基本用法
  python -m deploy.tcp_server --model outputs/.../best_model.pth

  # 自定义端口
  python -m deploy.tcp_server --model best_model.pth --port 9527

  # ONNX 后端 (CPU 推理)
  python -m deploy.tcp_server --model resnet50_cat.onnx --backend onnx --port 9000
        """,
    )
    parser.add_argument("--model", type=str, required=True, help="模型文件路径")
    parser.add_argument("--backend", type=str, default="pytorch",
                        choices=["pytorch", "torchscript", "onnx"],
                        help="推理后端")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=9000, help="监听端口")
    parser.add_argument("--max-clients", type=int, default=10, help="最大并发连接数")
    parser.add_argument("--confidence-threshold", type=float, default=0.50,
                        help="置信度阈值（0-1），低于此值判定为非猫（默认 0.50）")
    args = parser.parse_args()

    server = TcpBreedServer(
        model_path=args.model,
        backend=args.backend,
        host=args.host,
        port=args.port,
        max_clients=args.max_clients,
        confidence_threshold=args.confidence_threshold,
    )
    server.start()


if __name__ == "__main__":
    main()
