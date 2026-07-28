"""
speech_request_detector.py

External-audio speech request detector for the classroom assistant robot.

This version does NOT open the PC microphone. Audio must be supplied by
voice_transmission_server_speech.py through:

    detector.feed_audio(waveform, sample_rate)

The public API remains compatible with main_speech_integrated.py:

    start()
    enable()
    disable()
    clear()
    poll()
    stop()

Speech requests are accepted only while enable() is active, which should only
happen in the WAIT_CARD state.
"""

from __future__ import annotations

import queue
import re
import threading
from typing import Any

import numpy as np
import speech_recognition as sr


REQUEST_IDS = {
    "blocks": 0,
    "pencil": 1,
    "eraser": 2,
    "teacher": 3,
}


class SpeechRequestDetector:
    """Recognize request keywords from externally supplied audio."""

    TARGET_SAMPLE_RATE = 16000

    def __init__(
        self,
        microphone_index: int | None = None,
        language: str = "en-US",
        window_seconds: float = 3.0,
        hop_seconds: float = 2.0,
        min_rms: float = 0.008,
        **_: Any,
    ) -> None:
        # Kept only for compatibility with older main code.
        self.microphone_index = microphone_index

        self.language = language
        self.window_seconds = max(1.0, float(window_seconds))
        self.hop_seconds = max(0.25, float(hop_seconds))
        self.min_rms = max(0.0, float(min_rms))

        self.recognizer = sr.Recognizer()

        self._results: queue.Queue[dict[str, Any]] = queue.Queue()
        self._jobs: queue.Queue[tuple[np.ndarray, int] | None] = queue.Queue()

        self._stop_event = threading.Event()
        self._enabled_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._lock = threading.Lock()
        self._audio_buffer = np.empty(0, dtype=np.float32)
        self._latched = False

        # Incremented whenever a session is reset/disabled.
        # A recognition result from an older session is discarded.
        self._session_id = 0
        self._recognition_pending = False

        self._window_samples = int(
            round(self.window_seconds * self.TARGET_SAMPLE_RATE)
        )
        self._hop_samples = int(
            round(self.hop_seconds * self.TARGET_SAMPLE_RATE)
        )

    @staticmethod
    def list_microphones() -> list[str]:
        """Compatibility helper; this external-audio version uses no microphone."""
        return []

    @staticmethod
    def parse_request(text: str) -> str | None:
        """Map recognized English text to the project's request names."""
        normalized = re.sub(r"[^a-z0-9\s]", " ", text.lower())
        words = set(normalized.split())

        if "pencil" in words or "pencils" in words:
            return "pencil"

        if "eraser" in words or "erasers" in words or "rubber" in words:
            return "eraser"

        if {"block", "blocks", "lego", "legos"} & words:
            return "blocks"

        if {"teacher", "help", "emergency"} & words:
            return "teacher"

        return None

    def start(self) -> None:
        """Start the background recognition worker once."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._worker,
            name="SpeechRequestDetector",
            daemon=True,
        )
        self._thread.start()

        print(
            "[SpeechRequestDetector] Started in external-audio mode. "
            "Waiting for feed_audio()."
        )

    def enable(self) -> None:
        """Enable recognition for one WAIT_CARD request session."""
        with self._lock:
            self._session_id += 1
            self._latched = False
            self._recognition_pending = False
            self._audio_buffer = np.empty(0, dtype=np.float32)

        self._clear_results()
        self._clear_jobs()
        self._enabled_event.set()
        print("[SpeechRequestDetector] Enabled for WAIT_CARD.")

    def disable(self) -> None:
        """Disable recognition outside WAIT_CARD and discard buffered audio."""
        self._enabled_event.clear()

        with self._lock:
            self._session_id += 1
            self._recognition_pending = False
            self._audio_buffer = np.empty(0, dtype=np.float32)

        self._clear_results()
        self._clear_jobs()

    def clear(self) -> None:
        """Clear one request session without changing enabled state."""
        with self._lock:
            self._session_id += 1
            self._latched = False
            self._recognition_pending = False
            self._audio_buffer = np.empty(0, dtype=np.float32)

        self._clear_results()
        self._clear_jobs()

    def poll(self) -> dict[str, Any] | None:
        """Return one confirmed speech result without blocking."""
        try:
            return self._results.get_nowait()
        except queue.Empty:
            return None

    def stop(self) -> None:
        """Stop the background recognition worker."""
        self._enabled_event.clear()
        self._stop_event.set()
        self._jobs.put(None)

        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def feed_audio(
        self,
        waveform: np.ndarray,
        sample_rate: int = TARGET_SAMPLE_RATE,
    ) -> None:
        """
        Supply one external audio chunk.

        Args:
            waveform:
                Mono or multi-channel NumPy audio. Integer PCM and floating
                point arrays are supported.
            sample_rate:
                Sample rate of the supplied chunk.
        """
        if self._stop_event.is_set() or not self._enabled_event.is_set():
            return

        audio = self._prepare_audio(waveform, sample_rate)
        if audio.size == 0:
            return

        with self._lock:
            if self._latched:
                return

            self._audio_buffer = np.concatenate((self._audio_buffer, audio))

            # Keep at most two windows to prevent unbounded growth.
            max_buffer = self._window_samples * 2
            if self._audio_buffer.size > max_buffer:
                self._audio_buffer = self._audio_buffer[-max_buffer:]

            if (
                self._recognition_pending
                or self._audio_buffer.size < self._window_samples
            ):
                return

            window = self._audio_buffer[: self._window_samples].copy()
            self._audio_buffer = self._audio_buffer[self._hop_samples :]

            rms = float(np.sqrt(np.mean(np.square(window), dtype=np.float64)))
            if rms < self.min_rms:
                return

            session_id = self._session_id
            self._recognition_pending = True

        self._jobs.put((window, session_id))
        print(
            f"[SpeechRequestDetector] Audio window queued "
            f"({self.window_seconds:.1f}s, RMS={rms:.4f})."
        )

    def _worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                job = self._jobs.get(timeout=0.2)
            except queue.Empty:
                continue

            if job is None:
                break

            waveform, session_id = job

            try:
                text = self._recognize_waveform(waveform)
            except sr.UnknownValueError:
                text = None
            except sr.RequestError as exc:
                print(
                    "[SpeechRequestDetector] Recognition service error:",
                    exc,
                )
                text = None
            except Exception as exc:
                print(
                    "[SpeechRequestDetector] Recognition error:",
                    exc,
                )
                text = None

            with self._lock:
                self._recognition_pending = False

                # Ignore a result returned after WAIT_CARD ended/reset.
                if (
                    session_id != self._session_id
                    or not self._enabled_event.is_set()
                    or self._latched
                ):
                    continue

            if not text:
                continue

            print(f"[SpeechRequestDetector] Heard: {text}")

            request = self.parse_request(text)
            if request is None:
                print(
                    "[SpeechRequestDetector] "
                    "No supported request keyword."
                )
                continue

            result = {
                "id": REQUEST_IDS[request],
                "source": "speech",
                "request": request,
                "text": text,
                "confidence": 1.0,
                "confirmed": True,
                "center": None,
            }

            with self._lock:
                if (
                    session_id != self._session_id
                    or not self._enabled_event.is_set()
                    or self._latched
                ):
                    continue
                self._latched = True

            self._results.put(result)
            print(
                f"[SpeechRequestDetector] "
                f"Request confirmed: {request}"
            )

    def _recognize_waveform(self, waveform: np.ndarray) -> str:
        pcm16 = np.clip(waveform, -1.0, 1.0)
        pcm16 = (pcm16 * 32767.0).astype("<i2", copy=False)

        audio_data = sr.AudioData(
            pcm16.tobytes(),
            self.TARGET_SAMPLE_RATE,
            sample_width=2,
        )

        return self.recognizer.recognize_google(
            audio_data,
            language=self.language,
        )

    @classmethod
    def _prepare_audio(
        cls,
        waveform: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

        audio = np.asarray(waveform)
        if audio.size == 0:
            return np.empty(0, dtype=np.float32)

        # Convert multi-channel audio to mono.
        if audio.ndim == 2:
            # Accept either (channels, samples) or (samples, channels).
            if audio.shape[0] <= 8:
                audio = audio.mean(axis=0)
            else:
                audio = audio.mean(axis=1)
        elif audio.ndim > 2:
            audio = audio.reshape(-1)

        if np.issubdtype(audio.dtype, np.integer):
            info = np.iinfo(audio.dtype)
            scale = float(max(abs(info.min), info.max))
            audio = audio.astype(np.float32) / scale
        else:
            audio = audio.astype(np.float32, copy=False)

        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        audio = np.clip(audio, -1.0, 1.0)

        if sample_rate == cls.TARGET_SAMPLE_RATE:
            return np.ascontiguousarray(audio, dtype=np.float32)

        target_length = int(
            round(audio.size * cls.TARGET_SAMPLE_RATE / sample_rate)
        )
        if target_length <= 0:
            return np.empty(0, dtype=np.float32)

        old_positions = np.arange(audio.size, dtype=np.float64)
        new_positions = np.linspace(
            0,
            max(audio.size - 1, 0),
            target_length,
            dtype=np.float64,
        )
        resampled = np.interp(
            new_positions,
            old_positions,
            audio,
        ).astype(np.float32)

        return np.ascontiguousarray(resampled)

    def _clear_results(self) -> None:
        while True:
            try:
                self._results.get_nowait()
            except queue.Empty:
                break

    def _clear_jobs(self) -> None:
        while True:
            try:
                item = self._jobs.get_nowait()
                if item is None:
                    self._jobs.put(None)
                    break
            except queue.Empty:
                break
