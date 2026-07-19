"""TCP 推理服务。

作为分布式架构中的"云端 GPU 推理节点"，运行在笔记本电脑（RTX A2000）上。

功能：
1. UDP 接收树莓派转发的 JPEG 视频帧
2. YOLOv8n 猫识别推理（毫秒级）
3. TCP 返回推理结果给树莓派/Arduino

协议设计（与 plan.md 中的分析一致）：
- 视频帧: UDP (port 9001) — 允许丢帧
- 推理结果: TCP (port 9002) — 必须可靠
- 心跳: TCP — 连接保活
"""

import argparse
import json
import socket
import struct
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

from infer import CatDetector


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================
# 自定义协议
# ============================================================

# UDP 视频帧协议
# [FrameID: 4 bytes][TotalChunks: 2 bytes][ChunkIndex: 2 bytes][DataLength: 4 bytes][JPEG Data]
UDP_HEADER_SIZE = 12
MAX_CHUNK_SIZE = 8192  # 8KB, 避免 IP 分片


def pack_frame_chunk(
    frame_id: int,
    total_chunks: int,
    chunk_index: int,
    data: bytes,
) -> bytes:
    """打包 UDP 视频帧分片。"""
    header = struct.pack(
        "!IHHI",
        frame_id & 0xFFFFFFFF,
        total_chunks & 0xFFFF,
        chunk_index & 0xFFFF,
        len(data),
    )
    return header + data


def unpack_frame_chunk(packet: bytes) -> tuple[int, int, int, bytes]:
    """解包 UDP 视频帧分片。"""
    frame_id = struct.unpack("!I", packet[0:4])[0]
    total_chunks = struct.unpack("!H", packet[4:6])[0]
    chunk_index = struct.unpack("!H", packet[6:8])[0]
    data_length = struct.unpack("!I", packet[8:12])[0]
    data = packet[12:12 + data_length]
    return frame_id, total_chunks, chunk_index, data


# TCP 推理结果协议
# [MessageLength: 4 bytes][JSON Data]
def pack_result_message(data: dict) -> bytes:
    """打包 TCP 推理结果消息。"""
    json_bytes = json.dumps(data, ensure_ascii=False).encode("utf-8")
    header = struct.pack("!I", len(json_bytes))
    return header + json_bytes


def unpack_result_message(data: bytes) -> dict:
    """解包 TCP 推理结果消息。"""
    length = struct.unpack("!I", data[:4])[0]
    return json.loads(data[4:4 + length])


# ============================================================
# 帧重组器
# ============================================================

class FrameReassembler:
    """UDP 分片帧重组器。

    处理 UDP 乱序到达和丢包情况。
    """

    def __init__(self, timeout_ms: int = 5000):
        self.timeout_ms = timeout_ms
        self._buffers: dict[int, dict] = {}  # frame_id → {chunks, total, received, timestamp}

    def add_chunk(
        self,
        frame_id: int,
        total_chunks: int,
        chunk_index: int,
        data: bytes,
    ) -> Optional[bytes]:
        """添加一个分片，如果帧完整则返回完整 JPEG 数据。"""
        now = time.monotonic()

        # 清理过期帧
        expired = [
            fid for fid, buf in self._buffers.items()
            if (now - buf["timestamp"]) * 1000 > self.timeout_ms
        ]
        for fid in expired:
            del self._buffers[fid]

        # 初始化或更新缓冲区
        if frame_id not in self._buffers:
            self._buffers[frame_id] = {
                "chunks": [None] * total_chunks,
                "total": total_chunks,
                "received": 0,
                "timestamp": now,
            }

        buf = self._buffers[frame_id]

        # 兼容 total_chunks 变化（JPEG 大小可能变化）
        if total_chunks != buf["total"]:
            # 重新分配
            old_chunks = buf["chunks"]
            buf["chunks"] = [None] * total_chunks
            buf["total"] = total_chunks
            # 恢复旧数据
            for i, chunk in enumerate(old_chunks):
                if i < total_chunks and chunk is not None:
                    buf["chunks"][i] = chunk
                    buf["received"] += 1

        # 存储分片
        if chunk_index < total_chunks and buf["chunks"][chunk_index] is None:
            buf["chunks"][chunk_index] = data
            buf["received"] += 1

        buf["timestamp"] = now

        # 检查是否完整
        if buf["received"] == total_chunks:
            complete = b"".join(buf["chunks"])
            del self._buffers[frame_id]
            return complete

        return None


# ============================================================
# 推理服务器
# ============================================================

class InferenceServer:
    """猫识别推理服务器。

    运行在笔记本电脑上，接收树莓派转发的视频帧，
    返回推理结果。
    """

    def __init__(
        self,
        model_path: str,
        udp_port: int = 9001,
        tcp_port: int = 9002,
        conf_threshold: float = 0.25,
        backend: str = "tensorrt",
    ):
        self.udp_port = udp_port
        self.tcp_port = tcp_port
        self.conf_threshold = conf_threshold
        self.running = False

        # 初始化检测器
        print(f"初始化检测器: {model_path}")
        self.detector = CatDetector(
            model_path=model_path,
            backend=backend,
            conf_threshold=conf_threshold,
        )

        # 帧重组器
        self.reassembler = FrameReassembler()

        # 统计
        self.stats = {
            "frames_received": 0,
            "frames_processed": 0,
            "total_latency_ms": 0.0,
            "errors": 0,
        }
        self.stats_lock = threading.Lock()

        # TCP 客户端管理
        self.tcp_clients: list[socket.socket] = []
        self.tcp_clients_lock = threading.Lock()

    def start(self) -> None:
        """启动服务器。"""
        self.running = True

        # 启动 UDP 接收线程
        udp_thread = threading.Thread(target=self._udp_receiver, daemon=True)
        udp_thread.start()

        # 启动 TCP 服务线程
        tcp_thread = threading.Thread(target=self._tcp_server, daemon=True)
        tcp_thread.start()

        # 启动统计报告线程
        stats_thread = threading.Thread(target=self._stats_reporter, daemon=True)
        stats_thread.start()

        print(f"\n{'='*60}")
        print(f"猫识别推理服务已启动")
        print(f"  UDP 视频帧端口: {self.udp_port}")
        print(f"  TCP 结果端口:   {self.tcp_port}")
        print(f"  GPU: NVIDIA RTX A2000 Laptop GPU")
        print(f"{'='*60}\n")

        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self) -> None:
        """停止服务器。"""
        self.running = False
        with self.tcp_clients_lock:
            for client in self.tcp_clients:
                try:
                    client.close()
                except Exception:
                    pass
            self.tcp_clients.clear()
        print("\n服务器已停止")

    def _udp_receiver(self) -> None:
        """UDP 视频帧接收线程。"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", self.udp_port))
        sock.settimeout(1.0)

        print(f"[UDP] 监听端口 {self.udp_port}")

        while self.running:
            try:
                packet, addr = sock.recvfrom(65536)
                if len(packet) < UDP_HEADER_SIZE:
                    continue

                frame_id, total_chunks, chunk_index, data = unpack_frame_chunk(packet)

                with self.stats_lock:
                    self.stats["frames_received"] += 1

                # 尝试重组完整帧
                jpeg_data = self.reassembler.add_chunk(
                    frame_id, total_chunks, chunk_index, data
                )

                if jpeg_data is not None:
                    # 解码 JPEG → numpy array
                    np_arr = np.frombuffer(jpeg_data, dtype=np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                    if frame is not None:
                        # 异步推理（不阻塞 UDP 接收）
                        threading.Thread(
                            target=self._process_frame,
                            args=(frame, frame_id, addr),
                            daemon=True,
                        ).start()

            except socket.timeout:
                continue
            except Exception as e:
                with self.stats_lock:
                    self.stats["errors"] += 1
                print(f"[UDP] 错误: {e}")

        sock.close()

    def _process_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
        sender_addr: tuple,
    ) -> None:
        """处理一帧图像：推理 → 返回结果。"""
        try:
            detections, latency = self.detector.detect_with_timing(frame)

            with self.stats_lock:
                self.stats["frames_processed"] += 1
                self.stats["total_latency_ms"] += latency

            # 构造结果消息
            result = {
                "type": "inference_result",
                "frame_id": frame_id,
                "timestamp_ms": int(time.time() * 1000),
                "inference_latency_ms": round(latency, 2),
                "detections": [
                    {
                        "class": "cat",
                        "confidence": round(det["confidence"], 4),
                        "bbox": [det["x1"], det["y1"], det["x2"], det["y2"]],
                        "center": [
                            (det["x1"] + det["x2"]) // 2,
                            (det["y1"] + det["y2"]) // 2,
                        ],
                    }
                    for det in detections
                ],
            }

            # 通过 TCP 发送结果
            self._broadcast_result(result)

            # 日志
            if detections:
                confs = [d["confidence"] for d in detections]
                print(
                    f"[推理] frame={frame_id}, "
                    f"检测到 {len(detections)} 只猫, "
                    f"置信度={[f'{c:.2f}' for c in confs]}, "
                    f"延迟={latency:.1f}ms"
                )

        except Exception as e:
            with self.stats_lock:
                self.stats["errors"] += 1
            print(f"[推理] frame={frame_id} 错误: {e}")

    def _tcp_server(self) -> None:
        """TCP 服务器（发送推理结果，接收心跳）。"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", self.tcp_port))
        sock.listen(5)
        sock.settimeout(1.0)

        print(f"[TCP] 监听端口 {self.tcp_port}")

        while self.running:
            try:
                client_sock, addr = sock.accept()
                print(f"[TCP] 客户端连接: {addr[0]}:{addr[1]}")

                with self.tcp_clients_lock:
                    self.tcp_clients.append(client_sock)

                # 为每个客户端启动处理线程
                threading.Thread(
                    target=self._handle_tcp_client,
                    args=(client_sock, addr),
                    daemon=True,
                ).start()

            except socket.timeout:
                continue
            except Exception as e:
                print(f"[TCP] 错误: {e}")

        sock.close()

    def _handle_tcp_client(self, sock: socket.socket, addr: tuple) -> None:
        """处理 TCP 客户端（接收心跳）。"""
        try:
            sock.settimeout(30.0)
            while self.running:
                # 接收消息长度
                header = sock.recv(4)
                if not header:
                    break

                msg_len = struct.unpack("!I", header)[0]
                data = sock.recv(msg_len)
                if not data:
                    break

                message = json.loads(data.decode("utf-8"))
                msg_type = message.get("type")

                if msg_type == "heartbeat":
                    # 回复心跳确认
                    ack = pack_result_message({
                        "type": "heartbeat_ack",
                        "server_time_ms": int(time.time() * 1000),
                    })
                    sock.sendall(ack)

        except (ConnectionError, OSError):
            pass
        finally:
            print(f"[TCP] 客户端断开: {addr[0]}:{addr[1]}")
            with self.tcp_clients_lock:
                if sock in self.tcp_clients:
                    self.tcp_clients.remove(sock)
            try:
                sock.close()
            except Exception:
                pass

    def _broadcast_result(self, result: dict) -> None:
        """向所有已连接 TCP 客户端广播推理结果。"""
        message = pack_result_message(result)
        disconnected = []

        with self.tcp_clients_lock:
            for client in self.tcp_clients:
                try:
                    client.sendall(message)
                except (ConnectionError, OSError):
                    disconnected.append(client)

            for client in disconnected:
                self.tcp_clients.remove(client)

    def _stats_reporter(self) -> None:
        """定期报告推理统计。"""
        while self.running:
            time.sleep(10)
            with self.stats_lock:
                frames = self.stats["frames_processed"]
                total_lat = self.stats["total_latency_ms"]
                avg_lat = total_lat / frames if frames > 0 else 0

                print(
                    f"\n[统计] 接收帧: {self.stats['frames_received']}, "
                    f"处理帧: {frames}, "
                    f"平均延迟: {avg_lat:.2f}ms, "
                    f"错误: {self.stats['errors']}"
                )


# ============================================================
# 主入口
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="猫识别推理服务器")
    parser.add_argument("--model", default="./models/cat_yolov8n.engine", help="模型路径")
    parser.add_argument("--udp-port", type=int, default=9001, help="UDP 视频帧端口")
    parser.add_argument("--tcp-port", type=int, default=9002, help="TCP 结果端口")
    parser.add_argument("--backend", default="tensorrt", choices=["tensorrt", "onnx", "ultralytics"])
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--config", default="config.yaml", help="配置文件")
    args = parser.parse_args()

    config = load_config(args.config)

    server = InferenceServer(
        model_path=args.model,
        udp_port=args.udp_port,
        tcp_port=args.tcp_port,
        conf_threshold=args.conf,
        backend=args.backend,
    )

    server.start()


if __name__ == "__main__":
    main()
