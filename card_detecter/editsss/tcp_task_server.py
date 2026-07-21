"""
task_server.py

Receive robot tasks from PC.
Run on Raspberry Pi.
"""

from socket import *
import json

HOST = "0.0.0.0"
PORT = 2105

server = socket(AF_INET, SOCK_STREAM)
server.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
server.bind((HOST, PORT))
server.listen(5)

print(f"[Pi] Listening on {HOST}:{PORT}")

while True:

    conn, addr = server.accept()

    print(f"[Pi] Connected: {addr}")

    try:

        data = conn.recv(4096)

        task = json.loads(data.decode())

        print("[Pi] Task received:")
        print(task)

        # ====================================================
        # TODO: Send command to Arduino
        # ====================================================

        item = task["item"]

        print(f"[Pi] Need to deliver: {item}")

        # Example:
        # serial.write(b"GO_TEACHER\n")

        conn.sendall(b"TASK_RECEIVED")

    except Exception as e:

        print(f"[Pi] Error: {e}")

    finally:

        conn.close()