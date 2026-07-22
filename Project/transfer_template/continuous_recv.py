"""
continuous_recv.py
==================
持续运行的图片接收服务器，用于不间断接收树莓派拍摄的图片并保存到电脑。

与 ref_recv.py 的区别：
  - ref_recv.py：作为 TCP 客户端，每次手动运行，接收完一批文件后退出
  - continuous_recv.py：作为 TCP 服务器，持续监听，树莓派随时连接发送图片，
    接收完毕后不退出，继续等待下一次连接

协议（兼容 ref_send.py 的发送格式）：
  1. 4 字节 Header 长度（大端 uint32）
  2. JSON Header：{"type": "file", "name": "...", "size": ..., "sha256": "...",
                    "index": ..., "total": ...}
  3. 文件数据（size 字节）
  4. 最后发送 {"type": "done", "file_count": ...} 表示本轮传输结束

用法：
    python continuous_recv.py
    python continuous_recv.py --port 2105 --output ./received_images
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import struct
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_PORT = 2105
CHUNK_SIZE = 1024 * 1024         # 每次 recv 的最大字节数
MAX_HEADER_SIZE = 1024 * 1024     # Header 最大长度
DEFAULT_BACKLOG = 5


# ── 线程安全的打印 ──────────────────────────────────────────────────
_print_lock = threading.Lock()


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    with _print_lock:
        print(f"[{timestamp}] {message}", flush=True)


# ── 协议层（与 ref_recv.py / ref_send.py 兼容）──────────────────────

def recv_exact(sock: socket.socket, length: int) -> bytes:
    """精确接收 length 字节数据"""
    data = bytearray()
    while len(data) < length:
        remaining = length - len(data)
        chunk = sock.recv(min(CHUNK_SIZE, remaining))
        if not chunk:
            raise ConnectionError(
                f"连接提前关闭：期望 {length} 字节，只收到 {len(data)} 字节"
            )
        data.extend(chunk)
    return bytes(data)


def recv_header(sock: socket.socket) -> dict[str, Any]:
    """接收并解析 JSON Header"""
    raw_length = recv_exact(sock, 4)
    header_length = struct.unpack("!I", raw_length)[0]
    if not (0 < header_length <= MAX_HEADER_SIZE):
        raise ValueError(f"不合理的 Header 长度：{header_length}")
    header_bytes = recv_exact(sock, header_length)
    return json.loads(header_bytes.decode("utf-8"))


def safe_file_name(raw_name: str) -> str:
    """从原始文件名提取安全的文件名"""
    name = Path(raw_name).name
    if not name or name in {".", ".."}:
        raise ValueError("无效文件名")
    return name


# ── 文件接收 ────────────────────────────────────────────────────────

def receive_one_file(
    sock: socket.socket,
    header: dict[str, Any],
    output_directory: Path,
) -> Path:
    """接收单个文件，校验 SHA-256，返回保存路径"""
    file_name = safe_file_name(str(header["name"]))
    file_size = int(header["size"])
    expected_hash = str(header["sha256"])
    index = int(header["index"])
    total = int(header["total"])

    if file_size < 0:
        raise ValueError("文件大小不能为负数")

    # 用时间戳重命名，避免同名覆盖
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stem = Path(file_name).stem
    suffix = Path(file_name).suffix or ".jpg"
    saved_name = f"{stem}_{timestamp}{suffix}"

    output_path = output_directory / saved_name
    temp_path = output_directory / (saved_name + ".part")

    log(f"  接收 {index}/{total}：{file_name} → {saved_name} "
        f"({file_size / 1024:.1f} KB)")

    remaining = file_size
    hasher = hashlib.sha256()

    with temp_path.open("wb") as f:
        while remaining > 0:
            chunk = sock.recv(min(CHUNK_SIZE, remaining))
            if not chunk:
                raise ConnectionError(f"接收 {file_name} 时连接中断")
            f.write(chunk)
            hasher.update(chunk)
            remaining -= len(chunk)

    # 校验 SHA-256
    actual_hash = hasher.hexdigest()
    if actual_hash != expected_hash:
        temp_path.unlink(missing_ok=True)
        raise ValueError(
            f"{file_name} SHA-256 校验失败（期望 {expected_hash[:16]}...，"
            f"实际 {actual_hash[:16]}...）"
        )

    temp_path.replace(output_path)
    log(f"  ✓ 保存完成：{output_path.name}")
    return output_path


# ── 客户端连接处理 ──────────────────────────────────────────────────

def handle_client(
    client_socket: socket.socket,
    client_address: tuple[str, int],
    output_directory: Path,
) -> None:
    """处理单个客户端连接：接收所有文件"""
    addr_str = f"{client_address[0]}:{client_address[1]}"
    log(f"[连接] {addr_str} 已连接")

    files_received = 0

    try:
        with client_socket:
            while True:
                header = recv_header(client_socket)
                message_type = header.get("type")

                if message_type == "file":
                    receive_one_file(client_socket, header, output_directory)
                    files_received += 1

                elif message_type == "done":
                    total = header.get("file_count", 0)
                    log(f"[完成] {addr_str} 本轮传输结束，"
                        f"共接收 {files_received}/{total} 个文件")
                    break

                else:
                    log(f"[警告] {addr_str} 未知消息类型：{message_type}，跳过")
                    break

    except (ConnectionResetError, ConnectionError, BrokenPipeError) as e:
        log(f"[断开] {addr_str} 连接中断（已接收 {files_received} 个文件）: {e}")
    except (ValueError, json.JSONDecodeError) as e:
        log(f"[错误] {addr_str} 协议解析失败: {e}")
    except Exception as e:
        log(f"[错误] {addr_str} 未预期异常: {e}")


# ── 主服务器逻辑 ────────────────────────────────────────────────────

def run_server(port: int, output_directory: Path) -> None:
    """启动 TCP 服务器，持续监听并接收文件"""
    output_directory.mkdir(parents=True, exist_ok=True)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("0.0.0.0", port))
        server.listen(DEFAULT_BACKLOG)

        log(f"图片接收服务器已启动，监听端口 {port}")
        log(f"保存目录：{output_directory}")
        log("等待树莓派连接...\n")

        try:
            while True:
                client_socket, client_address = server.accept()

                # 每个客户端连接在独立线程中处理
                thread = threading.Thread(
                    target=handle_client,
                    args=(client_socket, client_address, output_directory),
                    daemon=True,
                )
                thread.start()

                log(f"活跃连接数：{threading.active_count() - 2}")

        except KeyboardInterrupt:
            log("\n收到中断信号，正在关闭服务器...")
        finally:
            log("服务器已关闭。")


# ── CLI 入口 ────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="持续运行的图片接收服务器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python continuous_recv.py
  python continuous_recv.py --port 2105 --output ./received_images
  python continuous_recv.py --port 9999 --output D:/calib_data
        """,
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT,
        help=f"监听端口（默认 {DEFAULT_PORT}）",
    )
    parser.add_argument(
        "--output", default="./received_images",
        help="图片保存目录（默认 ./received_images）",
    )
    args = parser.parse_args()

    output_dir = Path(args.output).expanduser().resolve()

    run_server(port=args.port, output_directory=output_dir)


if __name__ == "__main__":
    main()
