"""
calib_transfer.py
=================
树莓派相机标定图片传输程序（程序 B）

功能：
  作为 TCP 服务器运行在树莓派上，将拍摄的标定图片批量传输至电脑端。
  电脑端可使用配套接收程序（或 netcat）连接并接收图片。

传输协议（与 ref_send.py / ref_recv.py 兼容）：
  1. 每个文件先发送 JSON Header（文件名、大小、SHA-256），使用 4 字节长度前缀
  2. 然后分块发送文件内容（每块 1 MB）
  3. 全部文件发送完毕后发送 {"type":"done"} 标记

用法（树莓派端）：
    python calib_transfer.py
    python calib_transfer.py --port 2105 --image-dir ./calib_captures

用法（电脑端接收，通过 Tailscale 连接树莓派）：
    # 方式一：使用配套的 ref_recv.py
    python ref_recv.py <pi-tailscale-ip> --port 2105 --output ./calib_images

    # 方式二：使用 Python 一行命令（单文件接收）
    python -c "
import socket,json,struct,hashlib,os
s=socket.socket();s.connect(('<pi-tailscale-ip>',2105))
while True:
    hl=struct.unpack('!I',s.recv(4))[0]
    h=json.loads(s.recv(hl).decode())
    if h.get('type')=='done':break
    os.makedirs('calib_images',exist_ok=True)
    rem=h['size'];hsh=hashlib.sha256()
    with open(f'calib_images/{h[\"name\"]}','wb') as f:
        while rem>0:
            c=s.recv(min(1048576,rem))
            if not c:break
            f.write(c);hsh.update(c);rem-=len(c)
    print(f'Received: {h[\"name\"]}')
print('Done')
s.close()
"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import struct
import threading
import time
from pathlib import Path
from typing import Any

# =============================================================================
# 配置
# =============================================================================

DEFAULT_PORT = 2105
CHUNK_SIZE = 1024 * 1024       # 1 MB 分块
MAX_HEADER_SIZE = 1024 * 1024  # 1 MB
PRINT_LOCK = threading.Lock()


def log(message: str) -> None:
    """线程安全的日志输出"""
    with PRINT_LOCK:
        print(message, flush=True)


# =============================================================================
# 文件准备
# =============================================================================

def calculate_sha256(file_path: Path) -> str:
    """计算文件的 SHA-256 校验值"""
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def scan_images(image_dir: Path) -> list[dict[str, Any]]:
    """
    扫描目录下所有 .jpg 文件，返回文件信息列表。
    按文件名排序，确保传输顺序一致。
    """
    image_paths = sorted(image_dir.glob("*.jpg"))
    if not image_paths:
        return []

    files = []
    for path in image_paths:
        log(f"  扫描: {path.name}  ({path.stat().st_size / 1024:.1f} KB)")
        files.append({
            "path": path,
            "name": path.name,
            "size": path.stat().st_size,
            "sha256": calculate_sha256(path),
        })
    return files


# =============================================================================
# 协议层：Header 发送
# =============================================================================

def send_header(sock: socket.socket, header: dict[str, Any]) -> None:
    """发送 JSON Header（4 字节长度前缀 + JSON 内容）"""
    header_bytes = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(header_bytes) > MAX_HEADER_SIZE:
        raise ValueError("Header 超出最大大小限制")
    sock.sendall(struct.pack("!I", len(header_bytes)))
    sock.sendall(header_bytes)


# =============================================================================
# 单文件发送
# =============================================================================

def send_one_file(
    sock: socket.socket,
    file_info: dict[str, Any],
    index: int,
    total: int,
) -> None:
    """发送单个文件到已连接的客户端"""
    file_path: Path = file_info["path"]
    file_size: int = file_info["size"]

    header = {
        "type": "file",
        "name": file_info["name"],
        "size": file_size,
        "sha256": file_info["sha256"],
        "index": index,
        "total": total,
    }
    send_header(sock, header)

    log(f"  发送 ({index}/{total}): {file_path.name}  ({file_size / 1024:.1f} KB)")

    start_time = time.monotonic()
    bytes_sent = 0

    with file_path.open("rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            sock.sendall(chunk)
            bytes_sent += len(chunk)

    elapsed = time.monotonic() - start_time
    speed = file_size / 1024 / elapsed if elapsed > 0 else 0
    log(f"  ✅ 完成: {file_path.name}  ({elapsed:.2f}s, {speed:.1f} KB/s)")


# =============================================================================
# 客户端处理（每个连接一个线程）
# =============================================================================

def handle_client(
    client_socket: socket.socket,
    client_address: tuple[str, int],
    files: list[dict[str, Any]],
) -> None:
    """处理单个客户端连接的完整传输流程"""
    try:
        with client_socket:
            total = len(files)
            log(f"\n[{client_address[0]}:{client_address[1]}] 已连接，共 {total} 张图片待发送")

            for index, file_info in enumerate(files, start=1):
                send_one_file(client_socket, file_info, index, total)

            # 发送结束标记
            send_header(client_socket, {"type": "done", "file_count": total})
            log(f"[{client_address[0]}:{client_address[1]}] 全部 {total} 张图片发送完成 ✅")

    except (BrokenPipeError, ConnectionResetError, ConnectionError) as e:
        log(f"[{client_address[0]}:{client_address[1]}] 连接中断: {e}")
    except Exception as e:
        log(f"[{client_address[0]}:{client_address[1]}] 发送失败: {e}")


# =============================================================================
# 主程序
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="将树莓派拍摄的标定图片通过 TCP 传输至电脑端",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  树莓派端（发送方）：
    python calib_transfer.py
    python calib_transfer.py --port 2105 --image-dir ./calib_captures

  电脑端（接收方）：
    python ref_recv.py <pi-tailscale-ip> --port 2105 --output ./calib_images
        """,
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"TCP 监听端口（默认 {DEFAULT_PORT}）")
    parser.add_argument("--image-dir", type=str, default="./calib_captures",
                        help="标定图片所在目录（默认 ./calib_captures）")
    parser.add_argument("--max-clients", type=int, default=5,
                        help="最大同时连接数（默认 5）")
    args = parser.parse_args()

    # 解析图片目录
    image_dir = Path(args.image_dir).resolve()
    if not image_dir.is_dir():
        print(f"[错误] 目录不存在: {image_dir}")
        print("  请先运行 calib_camera_server.py 拍摄标定图片")
        return

    # 扫描图片
    print("=" * 55)
    print("📦 树莓派标定图片传输服务")
    print("=" * 55)
    print(f"  图片目录:  {image_dir}")
    print(f"  监听端口:  {args.port}")

    files = scan_images(image_dir)
    if not files:
        print(f"\n[提示] 目录中没有 .jpg 文件，请先拍摄标定图片。")
        print(f"\n  服务器仍会启动，添加图片后重新连接即可。\n")
    else:
        total_size = sum(f["size"] for f in files)
        print(f"  图片数量:  {len(files)} 张")
        print(f"  总大小:    {total_size / 1024 / 1024:.2f} MB")
    print("=" * 55)

    # 显示 Tailscale IP（尝试自动获取）
    print("\n💡 电脑端连接方式：")
    print(f"   python ref_recv.py <树莓派IP> --port {args.port} --output ./calib_images")
    print()
    print("   树莓派 IP 地址参考：")
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        print(f"   局域网 IP:   {local_ip}")
    except Exception:
        pass
    print(f"   Tailscale IP: 在树莓派上运行 `tailscale ip -4` 查看")
    print(f"\n⏳ 等待电脑端连接... (按 Ctrl+C 停止)\n")

    # 启动 TCP 服务器
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(("", args.port))
        server_socket.listen(args.max_clients)

        threads: list[threading.Thread] = []

        try:
            while True:
                client_socket, client_address = server_socket.accept()

                # 每次连接前重新扫描目录（获取最新图片列表）
                files = scan_images(image_dir)

                thread = threading.Thread(
                    target=handle_client,
                    args=(client_socket, client_address, files),
                    daemon=True,
                )
                thread.start()
                threads.append(thread)

                # 清理已完成的线程
                threads = [t for t in threads if t.is_alive()]

        except KeyboardInterrupt:
            print("\n[信息] 收到中断信号，正在停止服务...")

        # 等待所有传输线程结束
        for t in threads:
            t.join(timeout=5)

    print("[信息] 服务已停止。")


if __name__ == "__main__":
    main()
