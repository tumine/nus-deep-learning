"""Blocking client for the phone audio endpoint exposed by ``ui_server.py``.

Timeouts are split on purpose:

* a short *connect* timeout (``ROBOT_AUDIO_CONNECT_TIMEOUT_SECONDS``,
  default 3 s) so a silently dropping network path (e.g. a missing Windows
  firewall rule on the laptop) fails fast instead of stalling the caller
  for the full playback timeout;
* a longer *read* timeout (``ROBOT_AUDIO_TIMEOUT_SECONDS``, default 20 s)
  because ``ui_server.py`` only answers after the phone browser confirms
  playback, which can legitimately take 5-20 s.
"""

from __future__ import annotations

import base64
import http.client
import json
import mimetypes
import os
import socket
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse


DEFAULT_SERVER_URL = os.environ.get(
    "ROBOT_AUDIO_SERVER_URL",
    "http://127.0.0.1:8000/api/audio",
)
AUDIO_DIRECTORY = Path(os.environ.get("ROBOT_AUDIO_DIRECTORY", Path(__file__).parent / "audio"))
DEFAULT_TIMEOUT_SECONDS = float(os.environ.get("ROBOT_AUDIO_TIMEOUT_SECONDS", "20.0"))
DEFAULT_CONNECT_TIMEOUT_SECONDS = float(
    os.environ.get("ROBOT_AUDIO_CONNECT_TIMEOUT_SECONDS", "3.0")
)
AUDIO_FILES: dict[int, str] = {
    1: "1.m4a",
    2: "2_item_request.m4a",
    3: "3_teacher_request.m4a",
    4: "4_teacher_loading.m4a",
    5: "5_student_unloading.m4a",
}


class AudioDispatcher:
    """Upload one prerecorded audio file and wait for browser playback to end."""

    def __init__(
        self,
        server_url: str = DEFAULT_SERVER_URL,
        audio_directory: Path = AUDIO_DIRECTORY,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        connect_timeout_seconds: float = DEFAULT_CONNECT_TIMEOUT_SECONDS,
    ) -> None:
        self.server_url = server_url
        self.audio_directory = Path(audio_directory)
        self.timeout_seconds = timeout_seconds
        self.connect_timeout_seconds = connect_timeout_seconds
        self._lock = threading.Lock()

    def play_audio_blocking(self, audio_id: int) -> bool:
        request_started = time.monotonic()
        trace_id = uuid.uuid4().hex[:8]
        try:
            filename = AUDIO_FILES[audio_id]
        except KeyError as error:
            raise ValueError(f"Unsupported audio ID: {audio_id}") from error

        audio_path = self.audio_directory / filename
        if not audio_path.is_file():
            raise FileNotFoundError(
                f"Audio ID {audio_id} is mapped to a missing file: {audio_path}"
            )

        audio_bytes = audio_path.read_bytes()
        print(
            f"[AUDIO {trace_id}] Preparing audio {audio_id} ({audio_path.name}, "
            f"{len(audio_bytes)} bytes) for {self.server_url}; "
            f"connect timeout={self.connect_timeout_seconds:.1f}s, "
            f"read timeout={self.timeout_seconds:.1f}s."
        )
        payload = {
            "audio_id": audio_id,
            "filename": audio_path.name,
            "media_type": mimetypes.guess_type(audio_path.name)[0] or "audio/mp4",
            "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
            "trace_id": trace_id,
        }
        body = json.dumps(payload).encode("utf-8")

        parsed_url = urlparse(self.server_url)
        request_path = parsed_url.path or "/"
        if parsed_url.query:
            request_path = f"{request_path}?{parsed_url.query}"
        connection_class = (
            http.client.HTTPSConnection
            if parsed_url.scheme == "https"
            else http.client.HTTPConnection
        )

        with self._lock:
            connection = connection_class(
                parsed_url.hostname,
                parsed_url.port,
                timeout=self.connect_timeout_seconds,
            )
            try:
                print(f"[AUDIO {trace_id}] POST {request_path} started.")
                connection.connect()
                # The socket is connected now; only the response may
                # legitimately take longer (server waits for phone playback).
                connection.sock.settimeout(self.timeout_seconds)
                connection.request(
                    "POST",
                    request_path,
                    body=body,
                    headers={"Content-Type": "application/json"},
                )
                response = connection.getresponse()
                response_text = response.read().decode("utf-8", errors="replace")
                elapsed = time.monotonic() - request_started
                if response.status != 200:
                    print(
                        f"[AUDIO {trace_id}] Playback rejected with HTTP "
                        f"{response.status} after {elapsed:.2f}s: {response_text}"
                    )
                    return False
                played = bool(json.loads(response_text).get("played"))
                print(
                    f"[AUDIO {trace_id}] Audio server replied HTTP {response.status} "
                    f"after {elapsed:.2f}s; played={played}."
                )
                return played
            except (TimeoutError, socket.timeout):
                elapsed = time.monotonic() - request_started
                print(
                    f"[AUDIO {trace_id}] Timed out after {elapsed:.2f}s "
                    "while contacting the audio server."
                )
            except (OSError, http.client.HTTPException) as network_error:
                elapsed = time.monotonic() - request_started
                print(
                    f"[AUDIO {trace_id}] Cannot reach audio server after "
                    f"{elapsed:.2f}s ({self.server_url}): {network_error}"
                )
            finally:
                connection.close()
        return False


_default_dispatcher = AudioDispatcher()


def play_audio_blocking(audio_id: int) -> bool:
    """Play ``audio_id`` and return only after browser playback completes or fails."""
    return _default_dispatcher.play_audio_blocking(audio_id)
