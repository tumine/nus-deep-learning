from socket import *
import json
import struct

SERVER_HOST = '0.0.0.0'
SERVER_PORT = 2105
BUFFER_SIZE = 65536  # 64KB buffer for file transfer
OUTPUT_DIR = './received'  # 接收文件的保存目录

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 启动服务器 ---
server_socket = socket(AF_INET, SOCK_STREAM)
server_socket.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
server_socket.bind((SERVER_HOST, SERVER_PORT))
server_socket.listen(1)

print(f'Listening on {SERVER_HOST}:{SERVER_PORT} ...')

conn, addr = server_socket.accept()
print(f'Connection from {addr}')

try:
    # --- 步骤1: 接收元数据（长度前缀协议）---
    # 读取4字节的元数据长度
    raw_len = conn.recv(4)
    if len(raw_len) < 4:
        raise Exception('Failed to receive metadata length')
    metadata_len = struct.unpack('!I', raw_len)[0]

    # 读取元数据JSON
    metadata_bytes = b''
    while len(metadata_bytes) < metadata_len:
        chunk = conn.recv(min(metadata_len - len(metadata_bytes), BUFFER_SIZE))
        if not chunk:
            raise Exception('Connection closed while receiving metadata')
        metadata_bytes += chunk

    metadata = json.loads(metadata_bytes.decode())
    file_name = metadata['name']
    file_size = metadata['size']

    print(f'Receiving: {file_name}')
    print(f'Size: {file_size} bytes ({file_size / (1024**3):.2f} GB)')

    # --- 步骤2: 接收文件内容 ---
    output_path = os.path.join(OUTPUT_DIR, file_name)
    received_bytes = 0

    with open(output_path, 'wb') as f:
        while received_bytes < file_size:
            # 计算本次要接收的字节数（不超过剩余量）
            to_read = min(BUFFER_SIZE, file_size - received_bytes)
            chunk = conn.recv(to_read)
            if not chunk:
                raise Exception('Connection closed prematurely')
            f.write(chunk)
            received_bytes += len(chunk)
            # 进度显示
            progress = received_bytes / file_size * 100
            print(f'\rProgress: {progress:.1f}% ({received_bytes}/{file_size} bytes)', end='')

    print()
    print(f'File saved to: {output_path}')

    # --- 步骤3: 发送确认 ---
    conn.sendall(f'OK: received {received_bytes} bytes'.encode())

except Exception as e:
    error_msg = f'ERROR: {str(e)}'
    print(f'\n{error_msg}')
    conn.sendall(error_msg.encode())

finally:
    conn.close()
    server_socket.close()
    print('Connection closed.')
