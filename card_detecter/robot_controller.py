"""
robot_controller.py

Send robot tasks to Raspberry Pi through TCP.
"""

import socket
import json


class RobotController:

    def __init__(
        self,
        host="100.84.2.68",
        port=2105
    ):

        self.host = host
        self.port = port

        self.busy = False

    def execute(self, task):

        self.busy = True

        try:

            print("=" * 50)
            print("[Robot] Sending Task")
            print(task)
            print("=" * 50)

            self.send_task(task)

        except Exception as e:

            print(f"[Robot] TCP Error: {e}")

        finally:

            self.busy = False

    def send_task(self, task):

        payload = json.dumps(task)

        with socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM
        ) as s:

            s.connect(
                (
                    self.host,
                    self.port
                )
            )

            s.sendall(
                payload.encode("utf-8")
            )

            print(
                "[Robot] Task sent successfully."
            )