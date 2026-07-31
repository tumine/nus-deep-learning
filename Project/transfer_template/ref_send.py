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


HOST = ""
DEFAULT_PORT = 2105
EXPECTED_CLIENTS = 1
CHUNK_SIZE = 1024 * 1024
MAX_HEADER_SIZE = 1024 * 1024

PRINT_LOCK = threading.Lock()


def log(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def calculate_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()

    with file_path.open("rb") as file:
        while True:
            chunk = file.read(CHUNK_SIZE)
            if not chunk:
                break
            hasher.update(chunk)

    return hasher.hexdigest()


def prepare_files(raw_paths: list[str]) -> list[dict[str, Any]]:
    prepared_files: list[dict[str, Any]] = []

    for raw_path in raw_paths:
        path = Path(raw_path).expanduser().resolve()

        if not path.is_file():
            raise FileNotFoundError(f"找不到文件：{path}")

        log(f"正在计算 SHA-256：{path.name}")

        prepared_files.append(
            {
                "path": path,
                "name": path.name,
                "size": path.stat().st_size,
                "sha256": calculate_sha256(path),
            }
        )

    return prepared_files


def send_header(sock: socket.socket, header: dict[str, Any]) -> None:
    header_bytes = json.dumps(
        header,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    if len(header_bytes) > MAX_HEADER_SIZE:
        raise ValueError("Header 太大")

    sock.sendall(struct.pack("!I", len(header_bytes)))
    sock.sendall(header_bytes)


def send_one_file(
    client_socket: socket.socket,
    client_address: tuple[str, int],
    file_info: dict[str, Any],
    index: int,
    total: int,
) -> None:
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

    send_header(client_socket, header)

    log(
        f"[{client_address[0]}:{client_address[1]}] "
        f"开始发送 {index}/{total}：{file_path.name} "
        f"({file_size / 1024 / 1024:.2f} MB)"
    )

    bytes_sent = 0
    next_report = 25
    start_time = time.monotonic()

    with file_path.open("rb") as file:
        while True:
            chunk = file.read(CHUNK_SIZE)
            if not chunk:
                break

            client_socket.sendall(chunk)
            bytes_sent += len(chunk)

            if file_size > 0:
                progress = int(bytes_sent * 100 / file_size)

                if progress >= next_report:
                    log(
                        f"[{client_address[0]}:{client_address[1]}] "
                        f"{file_path.name}：{progress}%"
                    )
                    while next_report <= progress:
                        next_report += 25

    elapsed = time.monotonic() - start_time
    speed = file_size / 1024 / 1024 / elapsed if elapsed > 0 else 0

    log(
        f"[{client_address[0]}:{client_address[1]}] "
        f"发送完成：{file_path.name}，"
        f"耗时 {elapsed:.2f} 秒，平均 {speed:.2f} MB/s"
    )


def handle_client(
    client_socket: socket.socket,
    client_address: tuple[str, int],
    files: list[dict[str, Any]],
) -> None:
    try:
        with client_socket:
            total = len(files)

            for index, file_info in enumerate(files, start=1):
                send_one_file(
                    client_socket,
                    client_address,
                    file_info,
                    index,
                    total,
                )

            send_header(
                client_socket,
                {
                    "type": "done",
                    "file_count": total,
                },
            )

            log(
                f"[{client_address[0]}:{client_address[1]}] "
                "全部文件发送完成"
            )

    except (BrokenPipeError, ConnectionResetError, ConnectionError) as error:
        log(
            f"[{client_address[0]}:{client_address[1]}] "
            f"客户端连接中断：{error}"
        )

    except Exception as error:
        log(
            f"[{client_address[0]}:{client_address[1]}] "
            f"发送失败：{error}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="向一个 TCP 客户端发送一个或多个大文件"
    )

    parser.add_argument(
        "files",
        nargs="+",
        help="要发送的文件路径，可以指定多个文件",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"服务器监听端口，默认 {DEFAULT_PORT}",
    )

    args = parser.parse_args()

    try:
        files = prepare_files(args.files)
    except (FileNotFoundError, PermissionError) as error:
        print(f"文件准备失败：{error}")
        return

    print("\n准备发送以下文件：")

    for index, file_info in enumerate(files, start=1):
        print(
            f"{index}. {file_info['name']} "
            f"({file_info['size'] / 1024 / 1024:.2f} MB)"
        )

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1,
        )

        server_socket.bind((HOST, args.port))
        server_socket.listen(EXPECTED_CLIENTS)

        print(f"\n服务器已启动，监听 TCP 端口 {args.port}")
        print(f"等待 {EXPECTED_CLIENTS} 个客户端连接……")

        clients: list[tuple[socket.socket, tuple[str, int]]] = []

        while len(clients) < EXPECTED_CLIENTS:
            client_socket, client_address = server_socket.accept()
            clients.append((client_socket, client_address))

            print(
                f"客户端 {len(clients)}/{EXPECTED_CLIENTS} 已连接："
                f"{client_address[0]}:{client_address[1]}"
            )

        print("\n一个客户端均已连接，开始发送文件。\n")

        threads: list[threading.Thread] = []

        for client_socket, client_address in clients:
            thread = threading.Thread(
                target=handle_client,
                args=(client_socket, client_address, files),
                daemon=False,
            )
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

    print("\n服务器任务结束。")


if __name__ == "__main__":
    main()