#!/usr/bin/env python3
"""Start the robot with one shared speech detector and one phone UI on port 8000.

The former standalone 8080/8081 voice webpage is no longer started. The port-8000
UI handles both prompt playback and WebRTC microphone upload.
"""

from main_speech_integrated import main as robot_main
from speech_request_detector import SpeechRequestDetector


def main() -> None:
    speech_detector = SpeechRequestDetector(
        microphone_index=None,
        language="en-US",
    )
    speech_detector.start()
    speech_detector.disable()

    try:
        robot_main(speech_detector=speech_detector)
    finally:
        speech_detector.disable()
        speech_detector.stop()
        print("[RUNNER] Program stopped.")


if __name__ == "__main__":
    main()
