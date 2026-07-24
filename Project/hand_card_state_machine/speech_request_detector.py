"""
speech_request_detector.py

Non-blocking speech request detector for the classroom robot.

Recognised requests:
    "I want a pencil" -> pencil
    "I need an eraser" -> eraser
    "Can I have some blocks" -> blocks
    "I need a teacher" / "help" -> teacher

The microphone and speech recognition run in a background thread, so the
camera loop, UI, TCP communication and robot state machine are not blocked.
"""

from __future__ import annotations

import queue
import re
import threading
import time
from typing import Any

import speech_recognition as sr


class SpeechRequestDetector:
    """Listen for spoken classroom requests without blocking main.py."""

    def __init__(
        self,
        microphone_index: int | None = None,
        language: str = "en-US",
        listen_timeout: float = 1.0,
        phrase_time_limit: float = 4.0,
        ambient_adjust_seconds: float = 1.0,
    ) -> None:
        self.microphone_index = microphone_index
        self.language = language
        self.listen_timeout = listen_timeout
        self.phrase_time_limit = phrase_time_limit
        self.ambient_adjust_seconds = ambient_adjust_seconds

        self.recognizer = sr.Recognizer()
        self.recognizer.dynamic_energy_threshold = True

        self._results: queue.Queue[dict[str, Any]] = queue.Queue()
        self._stop_event = threading.Event()
        self._enabled_event = threading.Event()

        self._thread: threading.Thread | None = None
        self._latched = False
        self._lock = threading.Lock()

    @staticmethod
    def list_microphones() -> list[str]:
        """Return microphone names; list index is the device_index."""
        return sr.Microphone.list_microphone_names()

    @staticmethod
    def parse_request(text: str) -> str | None:
        """Convert recognised English text into the project's request names."""
        normalised = re.sub(r"[^a-z0-9\s]", " ", text.lower())
        words = set(normalised.split())

        if "pencil" in words or "pencils" in words:
            return "pencil"

        if "eraser" in words or "erasers" in words or "rubber" in words:
            return "eraser"

        if (
            "block" in words
            or "blocks" in words
            or "lego" in words
            or "legos" in words
        ):
            return "blocks"

        if (
            "teacher" in words
            or "help" in words
            or "emergency" in words
        ):
            return "teacher"

        return None

    def start(self) -> None:
        """Start the background listener once."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._worker,
            name="SpeechRequestDetector",
            daemon=True,
        )
        self._thread.start()

    def enable(self) -> None:
        """Allow speech requests, normally when state == WAIT_CARD."""
        with self._lock:
            self._latched = False
        self._clear_queue()
        self._enabled_event.set()

    def disable(self) -> None:
        """Ignore speech requests outside the request-waiting state."""
        self._enabled_event.clear()
        self._clear_queue()

    def clear(self) -> None:
        """Prepare for a new student/request round."""
        with self._lock:
            self._latched = False
        self._clear_queue()

    def poll(self) -> dict[str, Any] | None:
        """Return one new confirmed speech result, or None immediately."""
        try:
            return self._results.get_nowait()
        except queue.Empty:
            return None

    def stop(self) -> None:
        """Stop the background listener."""
        self._stop_event.set()
        self._enabled_event.set()

        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _clear_queue(self) -> None:
        while True:
            try:
                self._results.get_nowait()
            except queue.Empty:
                break

    def _worker(self) -> None:
        try:
            microphone = sr.Microphone(
                device_index=self.microphone_index
            )
        except Exception as exc:
            print(f"[Speech] Cannot open microphone: {exc}")
            return

        # Calibrate once when the module starts.
        try:
            with microphone as source:
                print("[Speech] Calibrating ambient noise...")
                self.recognizer.adjust_for_ambient_noise(
                    source,
                    duration=self.ambient_adjust_seconds,
                )
                print(
                    "[Speech] Ready. Energy threshold:",
                    round(self.recognizer.energy_threshold, 1),
                )
        except Exception as exc:
            print(f"[Speech] Calibration failed: {exc}")

        while not self._stop_event.is_set():
            # Do not listen during PATROL / DELIVERY / RETURN states.
            if not self._enabled_event.is_set():
                time.sleep(0.05)
                continue

            with self._lock:
                if self._latched:
                    time.sleep(0.05)
                    continue

            try:
                with microphone as source:
                    audio = self.recognizer.listen(
                        source,
                        timeout=self.listen_timeout,
                        phrase_time_limit=self.phrase_time_limit,
                    )
            except sr.WaitTimeoutError:
                continue
            except Exception as exc:
                print(f"[Speech] Microphone error: {exc}")
                time.sleep(0.5)
                continue

            try:
                text = self.recognizer.recognize_google(
                    audio,
                    language=self.language,
                )
            except sr.UnknownValueError:
                continue
            except sr.RequestError as exc:
                print(f"[Speech] Recognition service error: {exc}")
                time.sleep(1.0)
                continue
            except Exception as exc:
                print(f"[Speech] Recognition error: {exc}")
                continue

            request = self.parse_request(text)

            print(f"[Speech] Heard: {text}")

            if request is None:
                print("[Speech] No supported request keyword")
                continue

            result = {
                "id": {
                    "blocks": 0,
                    "pencil": 1,
                    "eraser": 2,
                    "teacher": 3,
                }[request],
                "source": "speech",
                "request": request,
                "text": text,
                "confidence": 1.0,
                "confirmed": True,
            }

            with self._lock:
                if self._latched:
                    continue
                self._latched = True

            self._results.put(result)
            print(f"[Speech] Request confirmed: {request}")
