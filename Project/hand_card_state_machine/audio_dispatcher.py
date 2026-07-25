"""Low-latency client for the phone audio endpoint exposed by ``ui_server.py``."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import threading
from pathlib import Path
from urllib import error as urlerror
from urllib import request


DEFAULT_SERVER_URL = os.environ.get(
    "ROBOT_AUDIO_SERVER_URL",
    "http://127.0.0.1:8000/api/audio",
)
DEFAULT_TIMEOUT_SECONDS = float(os.environ.get("ROBOT_AUDIO_TIMEOUT_SECONDS", "4"))
AUDIO_DIRECTORY = Path(os.environ.get("ROBOT_AUDIO_DIRECTORY", Path(__file__).parent / "audio"))
AUDIO_FILES: dict[int, str] = {
    1: "1.m4a",
    2: "2_item_request.m4a",
    3: "3_teacher_request.m4a",
    4: "4_teacher_loading.m4a",
    5: "5_student_unloading.m4a",
}


class AudioDispatcher:
    """Upload one prerecorded audio file and wait only for browser acceptance."""

    def __init__(
        self,
        server_url: str = DEFAULT_SERVER_URL,
        audio_directory: Path = AUDIO_DIRECTORY,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.server_url = server_url
        self.audio_directory = Path(audio_directory)
        self.timeout_seconds = timeout_seconds
        self._lock = threading.Lock()

    def play_audio_blocking(self, audio_id: int) -> bool:
        try:
            filename = AUDIO_FILES[audio_id]
        except KeyError as error:
            raise ValueError(f"Unsupported audio ID: {audio_id}") from error

        audio_path = self.audio_directory / filename
        if not audio_path.is_file():
            raise FileNotFoundError(
                f"Audio ID {audio_id} is mapped to a missing file: {audio_path}"
            )

        payload = {
            "audio_id": audio_id,
            "filename": audio_path.name,
            "media_type": mimetypes.guess_type(audio_path.name)[0] or "audio/mp4",
            "audio_base64": base64.b64encode(audio_path.read_bytes()).decode("ascii"),
        }
        http_request = request.Request(
            self.server_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with self._lock:
            try:
                with request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                    result = json.loads(response.read().decode("utf-8"))
                    return bool(result.get("accepted", result.get("played")))
            except urlerror.HTTPError as response_error:
                detail = response_error.read().decode("utf-8", errors="replace")
                print(f"[AUDIO] Playback rejected ({response_error.code}): {detail}")
            except urlerror.URLError as network_error:
                print(f"[AUDIO] Cannot reach audio server: {network_error.reason}")
            except TimeoutError:
                print("[AUDIO] Timed out waiting for browser acceptance.")
        return False

    def play_audio_async(self, audio_id: int, on_result=None) -> threading.Thread:
        """Start prompt delivery without blocking the robot control loop."""
        def deliver() -> None:
            try:
                played = self.play_audio_blocking(audio_id)
            except (FileNotFoundError, ValueError) as error:
                print(f"[AUDIO] Unable to dispatch audio {audio_id}: {error}")
                played = False
            if on_result is not None:
                on_result(played)

        thread = threading.Thread(
            target=deliver,
            name=f"audio-prompt-{audio_id}",
            daemon=True,
        )
        thread.start()
        return thread


_default_dispatcher = AudioDispatcher()


def play_audio_blocking(audio_id: int) -> bool:
    """Deliver ``audio_id`` and return after the browser accepts or rejects it."""
    return _default_dispatcher.play_audio_blocking(audio_id)


def play_audio_async(audio_id: int, on_result=None) -> threading.Thread:
    """Deliver ``audio_id`` in the background and return immediately."""
    return _default_dispatcher.play_audio_async(audio_id, on_result)