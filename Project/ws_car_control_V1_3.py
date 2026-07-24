"""Voice-aware vehicle controller.

Run this on the vehicle host. Set ``ROBOT_AUDIO_SERVER_URL`` to the laptop's
``main_speech.py`` address when the vehicle host is different from the laptop.
"""

from __future__ import annotations

import queue
import sys
import threading
import time

import ws_car_control_V1_2 as base
from hand_card_state_machine.audio_dispatcher import play_audio_blocking


class VoiceCarController(base.CarController):
    """Preserve the V1_2 route logic while blocking at spoken interaction points."""

    def _play_prompt(self, audio_id: int, context: str) -> bool:
        print(f"[AUDIO] {context}: playing prompt {audio_id}.")
        try:
            played = play_audio_blocking(audio_id)
        except (FileNotFoundError, ValueError) as error:
            print(f"[AUDIO WARNING] {error}")
            return False

        if not played:
            print("[AUDIO WARNING] Playback failed or timed out; continuing safely.")
        return played

    def detect_and_handle_student(self, branch_x: int, current_y: int) -> None:
        """Perform the delivery loop with speech before each relevant wait."""
        print(" -> [流程触发] 启动超声波循迹探路...")
        self.go_detect_obstacle()
        self.pos_x = branch_x
        self._play_prompt(1, "Student reached")
        base.send_status_to_pc("arrived_student")

        self.wait_for_signal(["go_teacher"])

        print(" -> [退回分叉点] 线上180度掉头并返回交叉口...")
        self.execute_cmd("PN")
        self.heading = (self.heading + 2) % 4
        self.execute_cmd("TF")
        self.pos_x = 0

        print(f" -> [连续返回起点] 正在从交叉口 {current_y} 连续开往真正的起点...")
        self.turn_to(base.SOUTH)
        self.execute_cmd(f"TF {current_y}")
        self.pos_y = 0
        self.turn_to(base.WEST)
        self.execute_cmd("TF")
        self.pos_x = -1
        self._play_prompt(4, "Teacher reached")
        base.send_status_to_pc("arrived_teacher")

        self.wait_for_signal(["return_student"])

        print(" -> [连续重返现场] 正在从起点连续开回主路并直达交叉口...")
        self.execute_cmd("PU")
        self.heading = base.EAST
        self.execute_cmd("TF")
        self.pos_x = 0
        self.continuous_move_to_y(current_y)

        target_heading = base.WEST if branch_x == -1 else base.EAST
        self.turn_to(target_heading)
        print(" -> [前行至现场] 重新驶入侧边栏靠近障碍物点...")
        self.go_detect_obstacle()
        self.pos_x = branch_x
        self._play_prompt(5, "Student return reached")
        base.send_status_to_pc("arrived_student")

        self.wait_for_signal(["return_patrol"])

        print(" -> [姿态补偿] 线上180度掉头并退回交叉口...")
        self.execute_cmd("PN")
        self.heading = (self.heading + 2) % 4
        self.execute_cmd("TF")
        self.pos_x = 0
        base.send_status_to_pc("route_rejoined")


def main() -> None:
    hardware = base.CarHardware(baudrate=9600)
    threading.Thread(
        target=base.background_input_listener,
        args=(hardware,),
        daemon=True,
    ).start()
    threading.Thread(
        target=base.tcp_server_listener,
        args=(hardware,),
        daemon=True,
    ).start()

    while True:
        base.stop_event.clear()
        while not base.signal_queue.empty():
            base.signal_queue.get_nowait()

        print("\n" + "=" * 50)
        if not base.network_ready_event.is_set():
            print("⏳ 正在等待电脑端视觉程序接入网络...")
            base.network_ready_event.wait()
            time.sleep(0.5)

        print("👉 请在终端输入 'start' 并回车以开始整个完整的过程:")
        while True:
            try:
                start_command = base.signal_queue.get(timeout=0.2)
                if start_command.lower() == "start":
                    print("▶️ [启动成功] 系统控制权限已激活，流程正式开始！")
                    break
                print(f"[错误] 指令 '{start_command}' 未识别，必须先输入 'start'。")
            except queue.Empty:
                continue

        controller = VoiceCarController(hardware)
        try:
            controller.run()
        except RuntimeError as error:
            print(f"\n🛑 [业务流中断] {error}")
            print("[系统提示] 小车已安全制动。正在重置业务流程...\n")
        except KeyboardInterrupt:
            print("\n[系统提示] 检测到 Ctrl+C，强制退出程序。")
            sys.exit(0)


if __name__ == "__main__":
    main()