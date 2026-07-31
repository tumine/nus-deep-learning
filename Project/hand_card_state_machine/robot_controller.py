"""robot_controller.py

Reliable one-command-per-connection TCP client for the Raspberry Pi.
"""

import json
import socket
from typing import Any, Dict


class RobotController:
    def __init__(self, host="100.84.2.68", port=2105, timeout=5.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.busy = False

    def execute(self, task: Dict[str, Any]) -> bool:
        if self.busy:
            print("[Robot] Controller is busy.")
            return False

        print("=" * 50)
        print("[Robot] Sending Task")
        print(task)
        print("=" * 50)

        self.busy = True
        try:
            payload = json.dumps(task, ensure_ascii=False).encode("utf-8")

            with socket.create_connection(
                (self.host, self.port),
                timeout=self.timeout,
            ) as sock:
                sock.settimeout(self.timeout)
                sock.sendall(payload)
                sock.shutdown(socket.SHUT_WR)

                reply_raw = sock.recv(4096)

            if not reply_raw:
                print("[Robot] Raspberry Pi closed without ACK.")
                return False

            reply = json.loads(reply_raw.decode("utf-8"))
            print(f"[Robot] Raspberry Pi ACK: {reply}")

            status = reply.get("status")
            return status in {"received", "accepted", "duplicate", "registered"}

        except (socket.timeout, TimeoutError):
            print(
                f"[Robot] TCP timeout connecting/sending to "
                f"{self.host}:{self.port}"
            )
            return False
        except ConnectionRefusedError:
            print(
                f"[Robot] Connection refused by {self.host}:{self.port}. "
                "Check that ws_car_control is listening on 2105."
            )
            return False
        except Exception as error:
            print(f"[Robot] TCP Error: {type(error).__name__}: {error}")
            return False
        finally:
            self.busy = False
