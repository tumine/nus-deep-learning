#!/usr/bin/env python3
"""
run_main_with_voice.py

统一启动机器人主程序与 iPhone WebRTC 语音输入。

语音请求仅在 WAIT_CARD 状态识别：
- WAIT_CARD: main_speech_integrated.py 调用 enable()
- 其他状态: detector 保持 disable()，传入音频会被立即丢弃
"""

from __future__ import annotations

import argparse
import asyncio
import threading
import time

from main_speech_integrated import main as robot_main
from speech_request_detector import SpeechRequestDetector
from voice_transmission_server_speech import (
    DEFAULT_CRY_THRESHOLD,
    VoiceServer,
    generate_ssl_cert,
)


def run_voice_server(server: VoiceServer) -> None:
    try:
        asyncio.run(server.start())
    except Exception as error:
        print(
            "[VOICE SERVER ERROR] "
            f"{type(error).__name__}: {error}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run robot main with iPhone WebRTC speech input."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--voice-port", type=int, default=8080)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 全工程共享唯一一份 detector。
    speech_detector = SpeechRequestDetector(
        microphone_index=None,
        language="en-US",
    )
    speech_detector.start()
    speech_detector.disable()

    ssl_ctx = generate_ssl_cert()

    voice_server = VoiceServer(
        host=args.host,
        port=args.voice_port,
        threshold=DEFAULT_CRY_THRESHOLD,
        ssl_ctx=ssl_ctx,
        teacher_url=None,
        speech_detector=speech_detector,
        enable_cry=False,
    )

    server_thread = threading.Thread(
        target=run_voice_server,
        args=(voice_server,),
        daemon=True,
        name="WebRTCVoiceServer",
    )
    server_thread.start()

    time.sleep(1.0)

    try:
        robot_main(speech_detector=speech_detector)
    finally:
        speech_detector.disable()
        speech_detector.stop()
        print("[RUNNER] Program stopped.")


if __name__ == "__main__":
    main()
