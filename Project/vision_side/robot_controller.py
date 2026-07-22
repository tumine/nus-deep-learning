"""
robot_controller.py

Send robot tasks to Raspberry Pi through TCP.
"""

import json
import socket


class RobotController:

    def __init__(
        self,
        host="100.84.2.68",
        port=2105,
        timeout=2.0
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.busy = False

    def execute(self, task):

        print("=" * 50)
        print("[Robot] Sending Task")
        print(task)
        print("=" * 50)

        self.busy = True

        try:
            data = json.dumps(task).encode("utf-8")

            with socket.socket(
                socket.AF_INET,
                socket.SOCK_STREAM
            ) as sock:

                sock.settimeout(self.timeout)

                sock.connect(
                    (self.host, self.port)
                )

                sock.sendall(data)

            print("[Robot] Task sent successfully.")

            return True

        except Exception as error:

            print(f"[Robot] TCP Error: {error}")

            return False

        finally:

            self.busy = False