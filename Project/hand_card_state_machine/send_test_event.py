import json
import socket
import time


PC_HOST = "127.0.0.1"
PC_PORT = 2106


def send_event(event, **payload):
    message = {
        "event": event,
        **payload,
    }

    data = (
        json.dumps(message, ensure_ascii=False) + "\n"
    ).encode("utf-8")

    with socket.create_connection(
        (PC_HOST, PC_PORT),
        timeout=3,
    ) as sock:
        sock.sendall(data)

        try:
            reply = sock.recv(1024)
            if reply:
                print(
                    "PC ACK:",
                    reply.decode("utf-8", errors="replace"),
                )
        except socket.timeout:
            pass

    print("Sent:", message)


if __name__ == "__main__":
    send_event(
        "scan_started",
        route_node=1,
        direction="left",
        timeout_seconds=10,
    )