"""Asynchronously send confirmed classroom requests to the teacher monitor."""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
from typing import Final


REQUEST_DESCRIPTIONS: Final[dict[str, str]] = {
    "pencil": "学生请求铅笔",
    "eraser": "学生请求橡皮",
    "blocks": "学生请求积木块",
    "teacher": "学生请求教师帮助",
}


class TeacherNotifier:
    """Queue teacher notifications so detection never blocks the control loop."""

    def __init__(self, teacher_url: str) -> None:
        self.teacher_url = teacher_url
        self._messages: queue.Queue[dict[str, object] | None] = queue.Queue()
        self._message_counter = 0
        self._worker = threading.Thread(
            target=self._run,
            name="teacher-notifier",
            daemon=True,
        )
        self._worker.start()

    def notify_request(self, request: str | None) -> bool:
        """Queue one confirmed request in the format expected by teacher_client.py."""
        description = REQUEST_DESCRIPTIONS.get(request or "")
        if description is None:
            print(f"[TEACHER NOTIFIER] Ignoring unsupported request: {request!r}")
            return False

        self._message_counter += 1
        self._messages.put(
            {
                "message_id": (
                    f"card-{int(time.time() * 1000)}-{self._message_counter:04d}"
                ),
                "axis_x": -1,
                "axis_y": -1,
                "request": "教师协助" if request == "teacher" else "物品",
                "description": description,
            }
        )
        return True

    def close(self) -> None:
        """Stop the background worker during application shutdown."""
        self._messages.put(None)
        self._worker.join(timeout=1.0)

    def _run(self) -> None:
        while True:
            message = self._messages.get()
            if message is None:
                return
            asyncio.run(self._send(message))

    async def _send(self, message: dict[str, object]) -> None:
        try:
            import websockets

            async with websockets.connect(
                self.teacher_url,
                open_timeout=3,
                close_timeout=1,
            ) as websocket:
                await websocket.send(json.dumps(message, ensure_ascii=False))
            print(
                "[TEACHER NOTIFIER] Sent "
                f"{message['message_id']} to {self.teacher_url}"
            )
        except ImportError:
            print("[TEACHER NOTIFIER] Missing dependency: pip install websockets")
        except Exception as error:
            print(f"[TEACHER NOTIFIER] Send failed: {error}")