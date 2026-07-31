from socket import *
import json
import os
import struct

SERVER_HOST = '172.20.10.7'
SERVER_PORT = 2105
BUFFER_SIZE = 65536  # 64KB buffer for file transfer

# --- 待传输的文件路径 ---
FILE_PATH = input('Enter file path: ').strip()

if not os.path.exists(FILE_PATH):
    print(f'Error: file "{FILE_PATH}" not found.')
    exit(1)

file_name = os.path.basename(FILE_PATH)
file_size = os.path.getsize(FILE_PATH)

print(f'File: {file_name}')
print(f'Size: {file_size} bytes ({file_size / (1024**3):.2f} GB)')

# --- 连接服务器 ---
client_socket = socket(AF_INET, SOCK_STREAM)
client_socket.connect((SERVER_HOST, SERVER_PORT))

# --- 步骤1: 发送文件元数据（JSON格式，长度前缀协议）---
metadata = json.dumps({'name': file_name, 'size': file_size}).encode()
# 先发送4字节的元数据长度（网络字节序），再发送元数据内容
client_socket.sendall(struct.pack('!I', len(metadata)))
client_socket.sendall(metadata)
print('Metadata sent.')

# --- 步骤2: 分块发送文件内容 ---
sent_bytes = 0
with open(FILE_PATH, 'rb') as f:
    while True:
        chunk = f.read(BUFFER_SIZE)
        if not chunk:
            break
        client_socket.sendall(chunk)
        sent_bytes += len(chunk)
        # 进度显示
        progress = sent_bytes / file_size * 100
        print(f'\rProgress: {progress:.1f}% ({sent_bytes}/{file_size} bytes)', end='')

print()
print(f'File transfer complete. Total sent: {sent_bytes} bytes.')

# --- 步骤3: 接收服务器确认 ---
response = client_socket.recv(1024)
print(f'Server response: {response.decode()}')

client_socket.close()
