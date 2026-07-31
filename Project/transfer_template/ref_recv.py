from __future__ import annotations

import argparse
import hashlib
import json
import socket
import struct
import time
from pathlib import Path
from typing import Any


DEFAULT_PORT = 2105
CHUNK_SIZE = 1024 * 1024
MAX_HEADER_SIZE = 1024 * 1024


def recv_exact(sock: socket.socket, length: int) -> bytes:
    data = bytearray()

    while len(data) < length:
        remaining = length - len(data)
        chunk = sock.recv(min(CHUNK_SIZE, remaining))

        if not chunk:
            raise ConnectionError(
                f"连接提前关闭：期望 {length} 字节，"
                f"只收到 {len(data)} 字节"
            )

        data.extend(chunk)

    return bytes(data)


def recv_header(sock: socket.socket) -> dict[str, Any]:
    raw_header_length = recv_exact(sock, 4)
    header_length = struct.unpack("!I", raw_header_length)[0]

    if header_length <= 0 or header_length > MAX_HEADER_SIZE:
        raise ValueError(f"不合理的 Header 长度：{header_length}")

    header_bytes = recv_exact(sock, header_length)
    return json.loads(header_bytes.decode("utf-8"))


def safe_file_name(raw_name: str) -> str:
    name = Path(raw_name).name

    if not name or name in {".", ".."}:
        raise ValueError("服务器发送了无效文件名")

    return name


def receive_one_file(
    sock: socket.socket,
    header: dict[str, Any],
    output_directory: Path,
) -> None:
    file_name = safe_file_name(str(header["name"]))
    file_size = int(header["size"])
    expected_hash = str(header["sha256"])
    index = int(header["index"])
    total = int(header["total"])

    if file_size < 0:
        raise ValueError("文件大小不能为负数")

    output_path = output_directory / file_name
    temporary_path = output_directory / (file_name + ".part")

    print(f"\n开始接收 {index}/{total}：{file_name}")
    print(f"文件大小：{file_size / 1024 / 1024:.2f} MB")

    remaining = file_size
    bytes_received = 0
    hasher = hashlib.sha256()
    next_report = 25
    start_time = time.monotonic()

    with temporary_path.open("wb") as file:
        while remaining > 0:
            chunk = sock.recv(min(CHUNK_SIZE, remaining))

            if not chunk:
                raise ConnectionError(
                    f"接收 {file_name} 时连接提前关闭"
                )

            file.write(chunk)
            hasher.update(chunk)

            chunk_length = len(chunk)
            remaining -= chunk_length
            bytes_received += chunk_length

            if file_size > 0:
                progress = int(bytes_received * 100 / file_size)

                if progress >= next_report:
                    print(f"{file_name}：{progress}%")
                    while next_report <= progress:
                        next_report += 25

    actual_hash = hasher.hexdigest()
    elapsed = time.monotonic() - start_time
    speed = file_size / 1024 / 1024 / elapsed if elapsed > 0 else 0

    if actual_hash != expected_hash:
        temporary_path.unlink(missing_ok=True)
        raise ValueError(
            f"{file_name} 的 SHA-256 不一致，"
            "文件可能不完整或已损坏"
        )

    temporary_path.replace(output_path)

    print(f"接收完成：{output_path}")
    print(f"SHA-256 校验成功：{actual_hash}")
    print(f"耗时 {elapsed:.2f} 秒，平均 {speed:.2f} MB/s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TCP 大文件接收客户端"
    )

    parser.add_argument(
        "server_ip",
        help="服务器电脑的 IPv4 地址",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"服务器端口，默认 {DEFAULT_PORT}",
    )

    parser.add_argument(
        "--output",
        default="received_files",
        help="文件保存目录，默认 received_files",
    )

    args = parser.parse_args()

    output_directory = Path(args.output).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    print(f"正在连接 {args.server_ip}:{args.port}……")

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            client_socket.connect((args.server_ip, args.port))
            print("已连接服务器，等待接收文件。")

            while True:
                header = recv_header(client_socket)
                message_type = header.get("type")

                if message_type == "file":
                    receive_one_file(
                        client_socket,
                        header,
                        output_directory,
                    )

                elif message_type == "done":
                    print(
                        f"\n服务器已发送完全部 "
                        f"{header.get('file_count', 0)} 个文件。"
                    )
                    break

                else:
                    raise ValueError(f"未知消息类型：{message_type}")

    except ConnectionRefusedError:
        print(
            "连接被拒绝：请确认服务器已启动、"
            "IP 和端口正确、防火墙允许连接。"
        )

    except (
        ConnectionResetError,
        ConnectionError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(f"接收失败：{error}")

    else:
        print(f"所有文件已保存在：{output_directory}")


if __name__ == "__main__":
    main()