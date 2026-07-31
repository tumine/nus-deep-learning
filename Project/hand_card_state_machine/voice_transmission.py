"""
voice_transmission.py

WebRTC microphone transmission manager for the classroom assistant robot.
Receives audio from the browser's microphone via WebRTC and feeds it into
the SpeechRequestDetector for voice request recognition.

This module is designed to be integrated into the existing FastAPI server
(ui_server.py) on port 8000, eliminating the need for a separate aiohttp
server on port 8080.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import numpy as np

logger = logging.getLogger("voice_transmission")


# ---------------------------------------------------------------------------
# Optional aiortc import
# ---------------------------------------------------------------------------
try:
    from aiortc import RTCPeerConnection, RTCSessionDescription

    HAS_AIORTC = True
except ImportError:
    HAS_AIORTC = False
    RTCPeerConnection = None  # type: ignore[assignment]
    RTCSessionDescription = None  # type: ignore[assignment]


class VoiceTransmissionManager:
    """Manages a single WebRTC peer connection for receiving browser microphone
    audio and feeding it into the shared SpeechRequestDetector.

    Only one WebRTC connection is allowed at a time.  A new offer
    automatically closes any existing connection.

    Parameters
    ----------
    speech_detector:
        The shared ``SpeechRequestDetector`` instance.  May be ``None`` when
        used standalone (audio is received but discarded).
    """

    def __init__(self, speech_detector: Any = None) -> None:
        if not HAS_AIORTC:
            logger.warning(
                "aiortc is not installed – microphone transmission disabled. "
                "Install with: pip install aiortc"
            )

        self.speech_detector = speech_detector
        self.pc: RTCPeerConnection | None = None
        self.audio_task: asyncio.Task | None = None

        self._active: bool = False
        self._sample_rate: int | None = None
        self._total_frames: int = 0
        self._start_time: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        return self._active and self.pc is not None

    async def handle_offer(self, sdp: str, sdp_type: str) -> dict[str, str]:
        """Process a WebRTC offer from the browser and return an SDP answer.

        If a previous connection exists it is closed first.
        """
        if not HAS_AIORTC:
            raise RuntimeError("aiortc is not installed")

        # Tear down any existing connection so only one mic stream is active.
        await self._close()

        self.pc = RTCPeerConnection()

        # ---- connection state callback -----------------------------------
        @self.pc.on("connectionstatechange")
        async def _on_state_change() -> None:
            state = self.pc.connectionState
            logger.info("WebRTC connection state: %s", state)
            if state in ("failed", "disconnected", "closed"):
                self._active = False

        # ---- incoming track handler --------------------------------------
        @self.pc.on("track")
        async def _on_track(track: Any) -> None:
            logger.info("Received track: %s", track.kind)
            if track.kind == "audio":
                logger.info("Audio track ready – starting processing loop")
                self._active = True
                self._start_time = time.time()
                self._total_frames = 0
                self.audio_task = asyncio.create_task(
                    self._process_audio(track)
                )

        # ---- SDP exchange ------------------------------------------------
        offer = RTCSessionDescription(sdp=sdp, type=sdp_type)
        await self.pc.setRemoteDescription(offer)

        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)

        logger.info("WebRTC answer created")
        return {
            "sdp": self.pc.localDescription.sdp,
            "type": self.pc.localDescription.type,
        }

    async def stop(self) -> None:
        """Stop the current WebRTC connection and free resources."""
        await self._close()

    def get_status(self) -> dict[str, Any]:
        """Return a lightweight status dict for the browser UI."""
        elapsed: float = 0.0
        if self._start_time is not None and self._active:
            elapsed = time.time() - self._start_time

        return {
            "active": self._active,
            "sample_rate": self._sample_rate,
            "total_frames": self._total_frames,
            "elapsed_seconds": round(elapsed, 1),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _process_audio(self, track: Any) -> None:
        """Continuously receive audio frames and feed them to the speech
        detector (if available)."""
        logger.info("Audio processing loop started")
        try:
            while True:
                try:
                    frame = await asyncio.wait_for(track.recv(), timeout=5.0)
                except asyncio.TimeoutError:
                    continue

                self._total_frames += 1

                if self.speech_detector is not None:
                    self._feed_frame_to_detector(frame)

                if self._total_frames == 1:
                    self._sample_rate = frame.sample_rate
                    logger.info(
                        "Audio stream established: %d Hz, %d channel(s)",
                        frame.sample_rate,
                        frame.layout.nb_channels,
                    )

        except asyncio.CancelledError:
            logger.info("Audio processing task cancelled")
        except Exception:
            logger.exception("Audio processing error")
        finally:
            self._active = False
            logger.info("Audio processing loop ended")

    def _feed_frame_to_detector(self, frame: Any) -> None:
        """Convert an aiortc AudioFrame into a float32 mono waveform and
        feed it to the SpeechRequestDetector."""
        audio_array = frame.to_ndarray()  # (channels, samples) int16
        num_channels = frame.layout.nb_channels

        # Convert to mono
        if num_channels >= 2:
            mono = audio_array.mean(axis=0).astype(np.int16)
        else:
            mono = audio_array[0].astype(np.int16)

        # int16 → float32, normalized to [-1, 1]
        float_audio = mono.astype(np.float32) / 32768.0

        self.speech_detector.feed_audio(
            float_audio,
            sample_rate=frame.sample_rate,
        )

    async def _close(self) -> None:
        """Cancel the audio task and close the peer connection."""
        if self.audio_task is not None and not self.audio_task.done():
            self.audio_task.cancel()
            try:
                await self.audio_task
            except asyncio.CancelledError:
                pass
            self.audio_task = None

        if self.pc is not None:
            await self.pc.close()
            self.pc = None

        self._active = False
        self._sample_rate = None
        logger.info("Voice transmission closed")
