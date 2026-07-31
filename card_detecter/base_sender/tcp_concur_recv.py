from socket import *
import json
import struct
import threading
import os

SERVER_HOST = '0.0.0.0'
SERVER_PORT = 2105
BUFFER_SIZE = 65536  # 64KB buffer for file transfer
OUTPUT_DIR = './received'  # 接收文件的保存目录

os.makedirs(OUTPUT_DIR, exist_ok=True)


def handle_client(conn: socket, addr):
    """处理单个客户端连接的完整文件接收流程。"""
    try:
        # --- 步骤1: 接收元数据（长度前缀协议）---
        raw_len = conn.recv(4)
        if len(raw_len) < 4:
            raise Exception('Failed to receive metadata length')
        metadata_len = struct.unpack('!I', raw_len)[0]

        metadata_bytes = b''
        while len(metadata_bytes) < metadata_len:
            chunk = conn.recv(min(metadata_len - len(metadata_bytes), BUFFER_SIZE))
            if not chunk:
                raise Exception('Connection closed while receiving metadata')
            metadata_bytes += chunk

        metadata = json.loads(metadata_bytes.decode())
        file_name = metadata['name']
        file_size = metadata['size']

        print(f'[{addr}] Receiving: {file_name}')
        print(f'[{addr}] Size: {file_size} bytes ({file_size / (1024**3):.2f} GB)')

        # --- 步骤2: 接收文件内容 ---
        output_path = os.path.join(OUTPUT_DIR, file_name)
        received_bytes = 0

        with open(output_path, 'wb') as f:
            while received_bytes < file_size:
                to_read = min(BUFFER_SIZE, file_size - received_bytes)
                chunk = conn.recv(to_read)
                if not chunk:
                    raise Exception('Connection closed prematurely')
                f.write(chunk)
                received_bytes += len(chunk)
                progress = received_bytes / file_size * 100
                print(f'\r[{addr}] Progress: {progress:.1f}% ({received_bytes}/{file_size} bytes)', end='')

        print()
        print(f'[{addr}] File saved to: {output_path}')

        # --- 步骤3: 发送确认 ---
        conn.sendall(f'OK: received {received_bytes} bytes'.encode())

    except Exception as e:
        error_msg = f'ERROR: {str(e)}'
        print(f'\n[{addr}] {error_msg}')
        try:
            conn.sendall(error_msg.encode())
        except Exception:
            pass

    finally:
        conn.close()
        print(f'[{addr}] Connection closed.')


# --- 主线程：持续监听并派发连接 ---
server_socket = socket(AF_INET, SOCK_STREAM)
server_socket.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
server_socket.bind((SERVER_HOST, SERVER_PORT))
server_socket.listen(10)  # 允许最多10个等待连接

print(f'[Server] Listening on {SERVER_HOST}:{SERVER_PORT} ...')
print(f'[Server] Press Ctrl+C to stop.')

try:
    while True:
        conn, addr = server_socket.accept()
        print(f'[Server] New connection from {addr}')
        # 为每个客户端创建一个新线程处理
        thread = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
        thread.start()
except KeyboardInterrupt:
    print('\n[Server] Shutting down...')
finally:
    server_socket.close()
    print('[Server] Stopped.')
