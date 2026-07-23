"""
ui_manager.py

Thread-safe communication bridge between main.py and the browser UI.

main.py publishes status/events to UIManager.
UIServer reads those events and broadcasts them to browser clients.
Browser control commands are placed in a separate command queue for main.py.
"""

from __future__ import annotations

import queue
import threading
from datetime import datetime
from typing import Any


class UIManager:
    """Store the latest UI state and manage event/command queues."""

    def __init__(self) -> None:
        self._event_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._command_queue: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()

        self._latest_state: dict[str, Any] = {
            "robot_state": "INITIALIZING",
            "scan_direction": "-",
            "route_node": "-",
            "current_request": "-",
            "pi_connected": False,
            "arduino_connected": False,
        }

        self._request_history: list[dict[str, Any]] = []

    def publish(self, event_type: str, data: dict[str, Any]) -> None:
        self._event_queue.put(
            {
                "type": event_type,
                "data": data,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def update_robot_state(self, state: str) -> None:
        with self._lock:
            self._latest_state["robot_state"] = state
        self.publish("robot_state", {"state": state})

    def update_scan(self, direction: str, route_node: str | int) -> None:
        with self._lock:
            self._latest_state["scan_direction"] = direction
            self._latest_state["route_node"] = route_node

        self.publish(
            "scan_update",
            {
                "direction": direction,
                "route_node": route_node,
            },
        )

    def update_request(
        self,
        request_type: str,
        description: str,
        *,
        message_id: str | None = None,
        axis_x: float | None = None,
        axis_y: float | None = None,
    ) -> None:
        request = {
            "message_id": message_id or f"REQ-{len(self._request_history) + 1:04d}",
            "axis_x": axis_x,
            "axis_y": axis_y,
            "request": request_type,
            "description": description,
            "received_at": datetime.now().isoformat(),
        }

        with self._lock:
            self._latest_state["current_request"] = description
            self._request_history.append(request)

        self.publish("new_request", request)

    def update_connection(self, device: str, connected: bool) -> None:
        key = f"{device}_connected"
        with self._lock:
            self._latest_state[key] = connected

        self.publish(
            "connection_update",
            {
                "device": device,
                "connected": connected,
            },
        )

    def get_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": self._latest_state.copy(),
                "requests": list(self._request_history),
            }

    def get_next_event(self, timeout: float | None = None) -> dict[str, Any] | None:
        try:
            return self._event_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def submit_command(self, command: str) -> None:
        self._command_queue.put(command.upper())

    def get_next_command(self) -> str | None:
        try:
            return self._command_queue.get_nowait()
        except queue.Empty:
            return None
