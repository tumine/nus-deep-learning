"""Blocking client for the phone audio endpoint exposed by ``ui_server.py``."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import threading
import time
import uuid
from pathlib import Path
from urllib import error as urlerror
from urllib import request


DEFAULT_SERVER_URL = os.environ.get(
    "ROBOT_AUDIO_SERVER_URL",
    "http://127.0.0.1:8000/api/audio",
)
AUDIO_DIRECTORY = Path(os.environ.get("ROBOT_AUDIO_DIRECTORY", Path(__file__).parent / "audio"))
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
        timeout_seconds: float = 20.0,
    ) -> None:
        self.server_url = server_url
        self.audio_directory = Path(audio_directory)
        self.timeout_seconds = timeout_seconds
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
            f"HTTP timeout={self.timeout_seconds:.1f}s."
        )
        payload = {
            "audio_id": audio_id,
            "filename": audio_path.name,
            "media_type": mimetypes.guess_type(audio_path.name)[0] or "audio/mp4",
            "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
            "trace_id": trace_id,
        }
        http_request = request.Request(
            self.server_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with self._lock:
            try:
                print(f"[AUDIO {trace_id}] POST /api/audio started.")
                with request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                    result = json.loads(response.read().decode("utf-8"))
                    played = bool(result.get("played"))
                    elapsed = time.monotonic() - request_started
                    print(
                        f"[AUDIO {trace_id}] Audio server replied HTTP {response.status} "
                        f"after {elapsed:.2f}s; played={played}."
                    )
                    return played
            except urlerror.HTTPError as response_error:
                detail = response_error.read().decode("utf-8", errors="replace")
                elapsed = time.monotonic() - request_started
                print(
                    f"[AUDIO {trace_id}] Playback rejected with HTTP "
                    f"{response_error.code} after {elapsed:.2f}s: {detail}"
                )
            except urlerror.URLError as network_error:
                elapsed = time.monotonic() - request_started
                print(
                    f"[AUDIO {trace_id}] Cannot reach audio server after "
                    f"{elapsed:.2f}s ({self.server_url}): {network_error.reason}"
                )
            except TimeoutError:
                elapsed = time.monotonic() - request_started
                print(
                    f"[AUDIO {trace_id}] Timed out after {elapsed:.2f}s "
                    "while waiting for the audio server response."
                )
        return False


_default_dispatcher = AudioDispatcher()


def play_audio_blocking(audio_id: int) -> bool:
    """Play ``audio_id`` and return only after browser playback completes or fails."""
    return _default_dispatcher.play_audio_blocking(audio_id)