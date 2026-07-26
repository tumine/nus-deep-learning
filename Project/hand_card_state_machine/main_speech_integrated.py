"""
main_speech.py

Classroom Assistant Robot Main Controller with Visual and Speech Requests

Workflow:
PATROL -> SCAN -> APPROACH_STUDENT -> WAIT_CARD (vision OR speech) -> GO_TEACHER
-> WAIT_LOADING -> RETURN_STUDENT -> WAIT_UNLOAD -> RETURN_PATROL -> PATROL

Automated network triggers are now active: 
Listens to TCP messages from the Raspberry Pi / Robot to transition states.
Temporary keyboard triggers remain as a manual override/fallback.
"""

import cv2
import os
import socket
import threading
import queue
import sys
import time

from camera import Camera
from audio_dispatcher import play_audio_blocking
from card_detector_classifier import CardDetector
from speech_request_detector import SpeechRequestDetector
from hand_detector import HandDetector
from request_manager import RequestManager
# from robot_controller import RobotController # 已被 TCP 网络通信替代
from state_machine import StateMachine, RobotState
from task_queue import TaskQueue

from ui_manager import UIManager
from ui_server import UIServer

try:
    import whisper
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False
    whisper = None
    print("[WARNING] openai-whisper not installed. Speech recognition via "
          "the web UI will be unavailable. Install: pip install openai-whisper")

# ==============================================================================
# ⚠️ 系统及网络配置区 
# ==============================================================================
CAMERA_URL = "http://100.84.2.68:5000/video_feed"
# CAMERA_URL = 0

HAND_MODEL_PATH = "yolov8n-pose.pt"
WINDOW_NAME = "Classroom Assistant"

DEFAULT_ROUTE_NODE = 1
DEFAULT_APPROACH_SECONDS = 1.5

# 请将这里的 IP 地址改为小车连接 WiFi 后分配到的真实 IP 地址！
ROBOT_IP = "100.84.2.68" 
ROBOT_PORT = 9999

# 摄像头容错配置
MAX_CAMERA_FAILURES = 5       # 连续失败帧数阈值
MAX_CAMERA_RECONNECTS = 3     # 最多重连次数
CAMERA_RECONNECT_DELAY = 2.0  # 重连前等待秒数

# 视频录制配置
RECORDING_DIR = "recordings"  # 录制视频保存目录

# 语音识别超时配置
SPEECH_TIMEOUT_SECONDS = 8.0  # 等待语音识别的最大秒数，超时后启用图像识别 fallback
# ==============================================================================


def print_separator():
    print("=" * 55)


# Speech results must be converted into the same dictionary structure used by
# RequestManager. These IDs preserve the original request mapping.
SPEECH_REQUEST_IDS = {
    "blocks": 0,
    "pencil": 1,
    "eraser": 2,
    "teacher": 3,
}


def normalize_speech_event(speech_event, state_machine):
    """
    Convert SpeechRequestDetector output into a CardDetector-compatible event.

    RequestManager currently expects:
        result["request"], result["id"], result["center"]

    A speech request has no image center, so the stored student target is used
    when available. Otherwise, center remains None.
    """
    request = speech_event.get("request")

    if request not in SPEECH_REQUEST_IDS:
        print(f"[SPEECH WARNING] Unsupported request: {request}")
        return None

    student_target = state_machine.get_context_value("student_target")
    center = student_target if isinstance(student_target, (tuple, list)) else None

    return {
        "id": SPEECH_REQUEST_IDS[request],
        "request": request,
        "center": center,
        "source": "speech",
        "text": speech_event.get("text", ""),
        "confidence": speech_event.get("confidence", 1.0),
        "confirmed": True,
    }


def send_robot_command(sock, command_dict):
    """向小车发送 TCP 动作指令"""
    if sock is None:
        print("[ROBOT] ⚠️ 网络未连接，无法向小车下发指令。")
        return False

    # 提取字典中的 "command" 字段（例如 "approach_student", "go_teacher"）发送给小车
    cmd_str = command_dict.get("command", "")

    print_separator()
    print("[ROBOT COMMAND]")
    print(f"📡 正在通过网络发送指令: {cmd_str}")
    print(f"📦 完整动作参数: {command_dict}")
    print_separator()

    try:
        # 发送指令并加上换行符，对应小车端的按行解析逻辑
        sock.sendall((cmd_str + '\n').encode('utf-8'))
        return True
    except Exception as e:
        print(f"❌ 发送指令失败: {e}")
        return False


def handle_hand_event(hand_event, state_machine):
    """Save student position and enter APPROACH_STUDENT."""

    target = hand_event.get("target")
    confidence = hand_event.get("confidence", 0.0)
    direction = state_machine.get_context_value("scan_direction", "front")

    state_machine.update_context(
        student_target=target,
        student_confidence=confidence,
        approach_command={
            "direction": direction,
            "target": target,
            "forward_seconds": DEFAULT_APPROACH_SECONDS,
        },
    )

    print_separator()
    print("[HAND EVENT]")
    print(f"Target         : {target}")
    print(f"Confidence     : {confidence:.2f}")
    print(f"Route node     : {state_machine.get_context_value('route_node')}")
    print(f"Scan direction : {direction}")
    print_separator()

    state_machine.set_state(RobotState.APPROACH_STUDENT)


def play_audio_with_feedback(audio_id, ui_manager):
    """Play prerecorded audio in the browser and report its result."""
    start_message = f"[AUDIO] Starting browser playback for audio {audio_id}."
    print(start_message)
    ui_manager.publish("audio_playback", {"message": start_message})

    try:
        played = play_audio_blocking(audio_id)
    except (FileNotFoundError, ValueError) as audio_error:
        result_message = f"[AUDIO WARNING] {audio_error}"
        print(result_message)
        ui_manager.publish("audio_playback", {"message": result_message})
        return False

    if played:
        result_message = f"[AUDIO] Browser playback completed for audio {audio_id}."
    else:
        result_message = f"[AUDIO WARNING] Browser playback failed or timed out for audio {audio_id}."

    print(result_message)
    ui_manager.publish("audio_playback", {"message": result_message})
    return played


def handle_request_event(request_event, request_manager, task_queue, state_machine, ui_manager):
    """
    Create and store one task from either visual classification or speech.

    Both sources are normalized to the structure required by RequestManager.
    A valid event always advances WAIT_CARD -> GO_TEACHER.
    """
    audio_id = 3 if request_event.get("request") == "teacher" else 2
    play_audio_with_feedback(audio_id, ui_manager)

    task = request_manager.create_task(request_event)
    task["student_context"] = state_machine.get_context()
    task["request_source"] = request_event.get("source", "vision")

    task_queue.add(task)
    state_machine.set_task(task)

    source = request_event.get("source", "vision")

    print_separator()
    print("[REQUEST EVENT]")
    print(f"Source    : {source}")
    print(f"Request   : {request_event.get('request', 'unknown')}")
    print(f"Request ID: {request_event.get('id')}")
    print(f"Center    : {request_event.get('center')}")
    if source == "speech":
        print(f"Heard text: {request_event.get('text', '')}")
    print(f"Task      : {task}")
    print_separator()

    state_machine.set_state(RobotState.GO_TEACHER)


def draw_system_status(frame, state_machine):
    """Draw current state and instructions."""

    state = state_machine.get_state()

    instructions = {
        RobotState.PATROL: "Patrolling - waiting for intersection_reached (or press I)",
        RobotState.SCAN: "Scanning hands (J=left, K=right)",
        RobotState.APPROACH_STUDENT: "Approaching student - waiting for robot arrival",
        RobotState.WAIT_CARD: "Waiting for object classification OR speech request",
        RobotState.GO_TEACHER: "Going to teacher - waiting for robot arrival",
        RobotState.WAIT_LOADING: "Waiting for vehicle button or UI loading confirmation",
        RobotState.RETURN_STUDENT: "Returning to student - waiting for robot arrival",
        RobotState.WAIT_UNLOAD: "Waiting for vehicle button or UI unloading confirmation",
        RobotState.RETURN_PATROL: "Returning to route - waiting for robot arrival",
    }

    cv2.rectangle(frame, (10, 10), (760, 110), (0, 0, 0), -1)

    cv2.putText(
        frame,
        f"STATE: {state.name}",
        (20, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        instructions.get(state, "Unknown state"),
        (20, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        (
            f"Route node: {state_machine.get_context_value('route_node')} | "
            f"Scan: {state_machine.get_context_value('scan_direction')}"
        ),
        (20, 98),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    return frame


def tcp_receive_thread(sock, network_queue, ui_manager, stop_event):
    """后台独立运行的网络接收线程，专门监听小车发回的状态信息"""
    buffer = ""
    while not stop_event.is_set():
        try:
            sock.settimeout(0.5)  # 每 0.5 秒检查一次 stop_event
            data = sock.recv(1024).decode('utf-8')
            if not data:
                print("⚠️ [网络通信] 与小车的连接已断开，请检查网络！")
                ui_manager.update_connection("pi", False)
                network_queue.put("connection_lost")
                break

            buffer += data
            # 解决 TCP 粘包问题，按行拆分指令
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                line = line.strip()
                if line:
                    print(f"\n📥 [网络通信] 接收到小车反馈状态: '{line}'")
                    network_queue.put(line)
        except socket.timeout:
            continue  # 超时后检查 stop_event 再循环
        except OSError as e:
            # 主线程关闭了 socket，这是正常的退出信号
            if stop_event.is_set():
                print("[网络通信] 接收线程收到关闭信号，正常退出。")
            else:
                print(f"❌ [网络通信] 接收数据异常退出: {e}")
                ui_manager.update_connection("pi", False)
                network_queue.put("connection_lost")
            break
        except Exception as e:
            print(f"❌ [网络通信] 接收数据异常退出: {e}")
            ui_manager.update_connection("pi", False)
            network_queue.put("connection_lost")
            break

def pc_input_listener(sock):
    """后台独立运行的终端输入监听线程，用于随时接收 S 键暂停指令"""
    while True:
        try:
            # 阻塞并读取电脑端终端的输入
            line = sys.stdin.readline().strip()
            if not line:
                continue
            if line.upper() == 'S':
                print("\n🛑 [电脑端终端控制] 检测到输入 'S'，正在向小车发送暂停指令...")
                # 借助已有的 send_robot_command 向小车发送 S
                send_robot_command(sock, {"command": "S"})
        except Exception:
            break


def main(speech_detector=None):
    print_separator()
    print("Starting Classroom Assistant...")
    print_separator()

    camera = None
    tcp_socket = None
    owns_speech_detector = speech_detector is None
    network_queue = queue.Queue()
    tcp_stop_event = threading.Event()

    ui_manager = UIManager()

    ui_server = UIServer(
        ui_manager=ui_manager,
        host="0.0.0.0",
        port=8000,
    )

    ui_server.start_in_thread()

    # Sync the audio dispatcher's URL protocol with the UI server.
    # When SSL certificates are available, the UI server only speaks
    # HTTPS, but audio_dispatcher.py defaults to HTTP.  An HTTP
    # request sent to an HTTPS server results in an immediate TCP
    # RST (WinError 10054) because the server expects a TLS handshake.
    from audio_dispatcher import _default_dispatcher as _audio_disp
    protocol = "https" if ui_server._use_https else "http"
    _audio_disp.server_url = (
        f"{protocol}://127.0.0.1:{ui_server.port}/api/audio"
    )

    # 尝试连接小车的 TCP Server
    print(f"🔌 正在尝试连接到小车控制端 ({ROBOT_IP}:{ROBOT_PORT})...")
    try:
        tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp_socket.settimeout(5.0)  # 连接超时时间 5 秒
        tcp_socket.connect((ROBOT_IP, ROBOT_PORT))
        tcp_socket.settimeout(None) # 连接成功后取消超时，改为持续阻塞监听
        print("✅ 成功连接到小车网络！")
        ui_manager.update_connection("pi", True)
        
        # 启动后台接收线程
        tcp_stop_event.clear()
        threading.Thread(target=tcp_receive_thread, args=(tcp_socket, network_queue, ui_manager, tcp_stop_event), daemon=True).start()
        # === 新增：启动电脑端终端键盘输入监听线程 ===
        threading.Thread(target=pc_input_listener, args=(tcp_socket,), daemon=True).start()

    except Exception as e:
        print(f"❌ 无法连接到小车: {e}")
        print("⚠️ 电脑端将以【离线/单机模式】运行，网络自动触发将失效，只能使用键盘按键模拟。")
        tcp_socket = None
        ui_manager.update_connection("pi", False)

    try:
        camera = Camera(CAMERA_URL)
        hand_detector = HandDetector(model_path=HAND_MODEL_PATH, conf=0.5)
        card_detector = CardDetector()
        request_manager = RequestManager()
        task_queue = TaskQueue()
        state_machine = StateMachine()
        ui_manager.update_robot_state(state_machine.get_state().name)
        last_ui_state = state_machine.get_state()
        if speech_detector is None:
            speech_detector = SpeechRequestDetector(
                microphone_index=None,
                language="en-US",
            )
            speech_detector.start()

        # 无论检测器由 main 创建还是由统一启动脚本传入，
        # 初始状态都保持关闭；只有进入 WAIT_CARD 才会 enable()。
        speech_detector.disable()

        # Load Whisper model and wire it to the UI server for speech
        # recognition.  The browser records audio via MediaRecorder,
        # uploads it to /api/upload_speech, and the transcribed result
        # is fed back into the SpeechRequestDetector via push_result().
        if HAS_WHISPER:
            print("[WHISPER] Loading Whisper model (base)...")
            whisper_model = whisper.load_model("base")
            ui_server.whisper_model = whisper_model
            ui_server.speech_detector_for_whisper = speech_detector
            print("[WHISPER] Model loaded and wired to UI server.")
        else:
            print("[WHISPER] Not available — web speech recording disabled.")

        # WAIT_CARD session guards:
        # - request_session_active: whether visual/speech detectors are armed
        # - request_accepted: guarantees only one request is accepted per student
        request_session_active = False
        request_accepted = False

        # 语音优先策略：
        # - speech_phase_start_time: 语音阶段开始时间（用于超时判断）
        # - image_fallback_active:  语音超时后是否已启用图像识别
        speech_phase_start_time = None
        image_fallback_active = False

        # 记录上一次发送指令时的状态，防止在同一个状态下一帧一帧疯狂重复发指令
        command_sent_for_state = None
        audio_prompt_state = None

        # 视频录制相关变量
        recording_writer = None
        recording_active = False
        recording_filename = None

        print("\n[MAIN] 状态触发控制说明:")
        print("  ▶ 正常情况：小车通过网络自动发送触发信号，自动跳转流程。")
        print("  ▶ 键盘测试：如果没连上小车，可使用以下按键手动模拟触发：")
        print("    I: inspection point reached (开启扫描)")
        print("    J/K: scan left/right (左右转头扫描)")
        print("    A: 模拟抵达学生 (arrived_student)")
        print("    T: 模拟抵达老师 (arrived_teacher)")
        print("    L: 老师已放好物品（备用键盘确认）")
        print("    B: 模拟返回学生 (returned to student)")
        print("    U: 学生已取走物品（备用键盘确认）")
        print("    R: 模拟回到主路线 (patrol route rejoined)")
        print("    Q: 退出程序")
        print("  ▶ 正常装载/卸载确认：优先使用小车物理按钮 button_pressed。")
        print("  ▶ 网页 UI 同时提供 Loading Complete / Unloading Complete 备用按钮。\n")

        camera_failures = 0
        camera_reconnects = 0

        while True:
            # 0. 处理网页 UI 发来的控制命令
            ui_command = ui_manager.get_next_command()

            if ui_command == "STOP":
                print("\n🛑 [WEB UI] 收到紧急停止命令。")
                send_robot_command(tcp_socket, {"command": "S"})

            elif ui_command == "LOAD_COMPLETE":
                if state_machine.is_state(RobotState.WAIT_LOADING):
                    print(
                        "▶️ [WEB UI] 老师确认装载完成，"
                        "开始返回学生处 "
                        "(WAIT_LOADING -> RETURN_STUDENT)"
                    )
                    state_machine.set_state(RobotState.RETURN_STUDENT)
                else:
                    print(
                        "[WEB UI WARNING] 忽略 LOAD_COMPLETE："
                        f"当前状态为 {state_machine.get_state().name}，"
                        "不是 WAIT_LOADING。"
                    )

            elif ui_command == "UNLOAD_COMPLETE":
                if state_machine.is_state(RobotState.WAIT_UNLOAD):
                    print(
                        "▶️ [WEB UI] 学生确认取件完成，"
                        "开始返回巡逻路线 "
                        "(WAIT_UNLOAD -> RETURN_PATROL)"
                    )
                    state_machine.set_state(RobotState.RETURN_PATROL)
                else:
                    print(
                        "[WEB UI WARNING] 忽略 UNLOAD_COMPLETE："
                        f"当前状态为 {state_machine.get_state().name}，"
                        "不是 WAIT_UNLOAD。"
                    )

            elif ui_command in ("PAUSE", "RESUME"):
                print(
                    f"[WEB UI] {ui_command} 尚未接入底盘协议，"
                    "当前仅记录。"
                )

            # 1. 优先处理网络发来的小车状态事件
            while not network_queue.empty():
                net_msg = network_queue.get()
                current_state = state_machine.get_state()

                if net_msg.startswith("scan_started:"):
                    parts = net_msg.split(":")
                    direction = parts[1] if len(parts) > 1 else "front"
                    route_node = parts[2] if len(parts) > 2 else DEFAULT_ROUTE_NODE

                    state_machine.update_context(
                        route_node=route_node,
                        scan_direction=direction,
                    )
                    if current_state == RobotState.PATROL:
                        state_machine.set_state(RobotState.SCAN)

                    ui_manager.update_scan(direction, route_node)

                elif net_msg.startswith("scan_finished:"):
                    if current_state == RobotState.SCAN:
                        print("▶️ [自动触发] 双侧扫描结束，继续巡逻。")
                        state_machine.reset()

                elif net_msg == "connection_lost":
                    print("🛑 [安全提示] 小车连接已断开。")

                elif net_msg == "intersection_reached":
                    if current_state == RobotState.PATROL:
                        print("▶️ [自动触发] 小车已到达巡逻点，开始扫描 (PATROL -> SCAN)")
                        state_machine.update_context(
                            route_node=DEFAULT_ROUTE_NODE,
                            scan_direction="front",
                        )
                        state_machine.set_state(RobotState.SCAN)
                
                if net_msg == "arrived_student":
                    if current_state == RobotState.APPROACH_STUDENT:
                        print("▶️ [自动触发] 小车已到达学生身边，开始识别物品或语音需求 (APPROACH_STUDENT -> WAIT_CARD)")
                        state_machine.set_state(RobotState.WAIT_CARD)
                    elif current_state == RobotState.RETURN_STUDENT:
                        print("▶️ [自动触发] 小车已带回物品到达学生身边，等待取件 (RETURN_STUDENT -> WAIT_UNLOAD)")
                        state_machine.set_state(RobotState.WAIT_UNLOAD)
                        
                elif net_msg == "arrived_teacher":
                    if current_state == RobotState.GO_TEACHER:
                        print("▶️ [自动触发] 小车已到达老师身边，等待放件 (GO_TEACHER -> WAIT_LOADING)")
                        state_machine.set_state(RobotState.WAIT_LOADING)
                        
                elif net_msg == "route_rejoined":
                    if current_state == RobotState.RETURN_PATROL:
                        print("▶️ [自动触发] 小车已回正到主巡逻路线，继续巡逻 (RETURN_PATROL -> PATROL)")
                        state_machine.reset()

                elif net_msg == "button_pressed":
                    if current_state == RobotState.WAIT_LOADING:
                        print(
                            "▶️ [小车物理按钮] 老师已完成装载，"
                            "开始返回学生处 "
                            "(WAIT_LOADING -> RETURN_STUDENT)"
                        )
                        state_machine.set_state(RobotState.RETURN_STUDENT)

                    elif current_state == RobotState.WAIT_UNLOAD:
                        print(
                            "▶️ [小车物理按钮] 学生已完成取件，"
                            "开始返回巡逻路线 "
                            "(WAIT_UNLOAD -> RETURN_PATROL)"
                        )
                        state_machine.set_state(RobotState.RETURN_PATROL)

                    else:
                        print(
                            "[BUTTON WARNING] 收到 button_pressed，"
                            f"但当前状态为 {current_state.name}，"
                            "本次按键已忽略。"
                        )

            # 2. 读取一帧摄像头画面
            frame = camera.read()
            if frame is None:
                camera_failures += 1
                print(f"[CAMERA] 读取帧失败 ({camera_failures}/{MAX_CAMERA_FAILURES})")

                if camera_failures >= MAX_CAMERA_FAILURES:
                    if camera_reconnects < MAX_CAMERA_RECONNECTS:
                        print(f"[CAMERA] 连续 {camera_failures} 次失败，尝试重连 "
                              f"({camera_reconnects + 1}/{MAX_CAMERA_RECONNECTS})...")
                        if camera.reconnect():
                            camera_failures = 0
                            camera_reconnects += 1
                            continue
                        else:
                            camera_reconnects += 1
                            camera_failures = 0
                    else:
                        print(f"[FATAL] 摄像头重连 {MAX_CAMERA_RECONNECTS} 次均失败，程序退出。")
                        break
                    time.sleep(CAMERA_RECONNECT_DELAY)
                continue

            # 帧读取成功，重置失败计数器
            camera_failures = 0

            # 如果正在录制，将当前帧写入视频文件
            if recording_active and recording_writer is not None:
                recording_writer.write(frame)

            # 3. 核心业务状态机流转
            current_state = state_machine.get_state()
            if command_sent_for_state != current_state:
                command_sent_for_state = None

            # Arrival prompts are synchronous gates. No movement, detector, or
            # physical-button step is unlocked until the phone confirms playback.
            if current_state != audio_prompt_state:
                if current_state == RobotState.WAIT_LOADING:
                    if not play_audio_with_feedback(4, ui_manager):
                        continue
                    send_robot_command(
                        tcp_socket,
                        {"command": "arm_loading_button"},
                    )
                elif current_state == RobotState.WAIT_UNLOAD:
                    if not play_audio_with_feedback(5, ui_manager):
                        continue
                    send_robot_command(
                        tcp_socket,
                        {"command": "arm_unload_button"},
                    )
                audio_prompt_state = current_state

            # --------------------------------------------------------------
            # Open exactly one visual/speech request session per student.
            # This works for both TCP arrival and keyboard simulation.
            # --------------------------------------------------------------
            if current_state == RobotState.WAIT_CARD and not request_session_active:
                print("[AUDIO] Robot arrived at student; playing request prompt.")
                audio_ok = play_audio_with_feedback(1, ui_manager)
                if not audio_ok:
                    print("[AUDIO WARNING] Request prompt playback failed. "
                          "Proceeding with speech detection anyway.")

                request_accepted = False
                request_session_active = True

                # 开始录制摄像头视频流
                os.makedirs(RECORDING_DIR, exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                recording_filename = os.path.join(RECORDING_DIR, f"recording_{timestamp}.avi")
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                recording_writer = cv2.VideoWriter(recording_filename, fourcc, 20.0, (w, h))
                recording_active = True
                print(f"[RECORDING] 开始录制视频: {recording_filename}")

                # 语音优先：先启动语音识别，暂不启动图像识别
                speech_detector.clear()
                speech_detector.enable()
                speech_phase_start_time = time.monotonic()
                image_fallback_active = False

                print_separator()
                print("[REQUEST SESSION] Speech recognition started (priority).")
                print(f"[REQUEST SESSION] Image recognition will start after "
                      f"{SPEECH_TIMEOUT_SECONDS:.0f}s if no speech is detected.")
                print_separator()

            elif current_state != RobotState.WAIT_CARD and request_session_active:
                # 离开 WAIT_CARD 但没有识别到结果，停止录制并丢弃视频
                if recording_active and recording_writer is not None:
                    recording_writer.release()
                    recording_writer = None
                    recording_active = False
                    print(f"[RECORDING] 未识别到结果，停止录制。")
                    # 可选择删除未使用的录制文件
                    try:
                        os.remove(recording_filename)
                        print(f"[RECORDING] 已删除未使用的视频: {recording_filename}")
                    except OSError:
                        pass

                speech_detector.disable()
                speech_detector.clear()
                request_session_active = False
                image_fallback_active = False
                speech_phase_start_time = None

            if current_state == RobotState.PATROL:
                # 预留给未来拓展
                pass

            elif current_state == RobotState.SCAN:
                hand_events = hand_detector.detect(frame)
                if hand_events:
                    handle_hand_event(hand_events[0], state_machine)
                frame = hand_detector.draw(frame)

            elif current_state == RobotState.APPROACH_STUDENT:
                if command_sent_for_state is None:
                    command = {
                        "command": "approach_student",
                        "route_node": state_machine.get_context_value("route_node"),
                        "scan_direction": state_machine.get_context_value("scan_direction"),
                        "student_target": state_machine.get_context_value("student_target"),
                        "approach": state_machine.get_context_value("approach_command"),
                    }
                    if send_robot_command(tcp_socket, command):
                        command_sent_for_state = current_state

            elif current_state == RobotState.WAIT_CARD:
                # ── 意图识别策略：语音优先，超时后回退到图像 ──
                # 1. 每帧首先检查语音结果（始终最高优先级）
                # 2. 若语音在 SPEECH_TIMEOUT_SECONDS 内返回结果 → 直接接受
                # 3. 若语音超时无结果 → 启动图像识别作为 fallback
                # 4. 图像 fallback 启动后，语音仍保持监听（后续若有人说话，语音仍优先）
                if not request_accepted:
                    # ── 语音识别：始终优先检查 ──
                    speech_raw = speech_detector.poll()
                    speech_event = None

                    if speech_raw is not None:
                        speech_event = normalize_speech_event(
                            speech_raw,
                            state_machine,
                        )

                    # ── 图像识别：仅在语音超时后启用 ──
                    visual_events = []
                    if speech_event is None:
                        if not image_fallback_active:
                            elapsed = time.monotonic() - speech_phase_start_time
                            if elapsed >= SPEECH_TIMEOUT_SECONDS:
                                print(
                                    f"[FALLBACK] Speech timed out after "
                                    f"{elapsed:.1f}s, enabling image recognition."
                                )
                                image_fallback_active = True
                                if hasattr(card_detector, "reset_session"):
                                    card_detector.reset_session()
                        else:
                            visual_events = card_detector.detect(frame)

                    # ── 结果判定：语音优先 ──
                    accepted_event = None

                    if speech_event is not None:
                        accepted_event = speech_event
                    elif visual_events:
                        accepted_event = dict(visual_events[0])
                        accepted_event.setdefault("source", "vision")

                    if accepted_event is not None:
                        # Main-level lock: prevents visual and speech from
                        # creating two tasks in the same student interaction.
                        request_accepted = True

                        # 停止录制并保存视频
                        if recording_active and recording_writer is not None:
                            recording_writer.release()
                            recording_writer = None
                            recording_active = False
                            print(f"[RECORDING] 识别到结果，停止录制，视频已保存: {recording_filename}")

                        # Stop listening immediately after the first accepted
                        # request. The next student session will re-enable it.
                        speech_detector.disable()
                        speech_detector.clear()

                        handle_request_event(
                            accepted_event,
                            request_manager,
                            task_queue,
                            state_machine,
                            ui_manager,
                        )

                        center = accepted_event.get("center")
                        axis_x = center[0] if center else None
                        axis_y = center[1] if center else None
                        source = accepted_event.get("source", "vision")
                        confidence = accepted_event.get("confidence")

                        ui_manager.update_request(
                            request_type="语音" if source == "speech" else "物品",
                            description=accepted_event.get("request", "unknown"),
                            message_id=(
                                f"{source.upper()}-"
                                f"{accepted_event.get('id', 'unknown')}"
                            ),
                            axis_x=axis_x,
                            axis_y=axis_y,
                            confidence=confidence,
                        )

                # draw() only displays the latest classifier result and must
                # not run classification a second time.
                frame = card_detector.draw(frame)

            elif current_state == RobotState.GO_TEACHER:
                if command_sent_for_state is None:
                    if not task_queue.has_task():
                        print("[WARNING] GO_TEACHER entered with empty queue.")
                        state_machine.reset()
                    else:
                        task = task_queue.next_task()
                        command = {
                            "command": "go_teacher",
                            "task": task,
                            "route_node": state_machine.get_context_value("route_node"),
                        }
                        if send_robot_command(tcp_socket, command):
                            command_sent_for_state = current_state
                        else:
                            # 发送失败则把任务塞回队列
                            task_queue.add(task)

            elif current_state == RobotState.WAIT_LOADING:
                # 等待小车物理按钮、网页确认按钮或备用键盘 L
                pass

            elif current_state == RobotState.RETURN_STUDENT:
                if command_sent_for_state is None:
                    command = {
                        "command": "return_student",
                        "task": state_machine.get_task(),
                        "route_node": state_machine.get_context_value("route_node"),
                        "scan_direction": state_machine.get_context_value("scan_direction"),
                        "student_target": state_machine.get_context_value("student_target"),
                        "approach": state_machine.get_context_value("approach_command"),
                    }
                    if send_robot_command(tcp_socket, command):
                        command_sent_for_state = current_state

            elif current_state == RobotState.WAIT_UNLOAD:
                # 等待小车物理按钮、网页确认按钮或备用键盘 U
                pass

            elif current_state == RobotState.RETURN_PATROL:
                if command_sent_for_state is None:
                    command = {
                        "command": "return_patrol",
                        "route_node": state_machine.get_context_value("route_node"),
                        "scan_direction": state_machine.get_context_value("scan_direction"),
                        "approach": state_machine.get_context_value("approach_command"),
                    }
                    if send_robot_command(tcp_socket, command):
                        command_sent_for_state = current_state

            else:
                print(f"[WARNING] Unknown state: {current_state}")
                state_machine.reset()

            # 状态发生变化时发布给网页 UI
            current_ui_state = state_machine.get_state()
            if current_ui_state != last_ui_state:
                ui_manager.update_robot_state(current_ui_state.name)
                last_ui_state = current_ui_state

            # 4. 界面绘制与键盘事件捕捉
            frame = draw_system_status(frame, state_machine)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("[MAIN] Quit key pressed.")
                break

            elif key == ord("i"):
                if state_machine.is_state(RobotState.PATROL):
                    state_machine.update_context(
                        route_node=DEFAULT_ROUTE_NODE,
                        scan_direction="front",
                    )
                    state_machine.set_state(RobotState.SCAN)

            elif key == ord("j"):
                if state_machine.is_state(RobotState.SCAN):
                    state_machine.update_context(scan_direction="left")
                    ui_manager.update_scan("left", state_machine.get_context_value("route_node"))
                    print("[SIMULATION] Scanning left.")

            elif key == ord("k"):
                if state_machine.is_state(RobotState.SCAN):
                    state_machine.update_context(scan_direction="right")
                    ui_manager.update_scan("right", state_machine.get_context_value("route_node"))
                    print("[SIMULATION] Scanning right.")

            elif key == ord("a"):
                if state_machine.is_state(RobotState.APPROACH_STUDENT):
                    print("[SIMULATION - 手动模拟] Robot arrived at student.")
                    state_machine.set_state(RobotState.WAIT_CARD)

            elif key == ord("t"):
                if state_machine.is_state(RobotState.GO_TEACHER):
                    print("[SIMULATION - 手动模拟] Robot arrived at teacher.")
                    state_machine.set_state(RobotState.WAIT_LOADING)

            elif key == ord("l"):
                if state_machine.is_state(RobotState.WAIT_LOADING):
                    print("[MAIN - 人工确认] Teacher loading completed.")
                    state_machine.set_state(RobotState.RETURN_STUDENT)

            elif key == ord("b"):
                if state_machine.is_state(RobotState.RETURN_STUDENT):
                    print("[SIMULATION - 手动模拟] Robot returned to student.")
                    state_machine.set_state(RobotState.WAIT_UNLOAD)

            # === 新增：在 OpenCV 图像界面按 S 键直接触发暂停下发 ===
            elif key in [ord("s"), ord("S")]:
                print("\n🛑 [图形界面控制] 检测到 'S' 键，正在向小车下发暂停指令...")
                send_robot_command(tcp_socket, {"command": "S"})

            elif key == ord("u"):
                if state_machine.is_state(RobotState.WAIT_UNLOAD):
                    print("[MAIN - 人工确认] Student unloading completed.")
                    state_machine.set_state(RobotState.RETURN_PATROL)

            elif key == ord("r"):
                if state_machine.is_state(RobotState.RETURN_PATROL):
                    print("[SIMULATION - 手动模拟] Patrol route rejoined.")
                    state_machine.reset()

    except KeyboardInterrupt:
        print("\n[MAIN] Program interrupted by user.")

    except Exception as error:
        print(f"[FATAL ERROR] {type(error).__name__}: {error}")
        raise

    finally:
        print("[MAIN] Releasing resources...")

        if speech_detector is not None:
            try:
                speech_detector.disable()
                # 外部传入的共享 detector 由 run_main_with_voice.py 统一关闭。
                if owns_speech_detector:
                    speech_detector.stop()
            except Exception as speech_error:
                print(f"[SPEECH WARNING] Failed to stop cleanly: {speech_error}")

        # 1. 先通知接收线程停止，避免 socket 关闭时的竞态错误
        tcp_stop_event.set()

        if tcp_socket is not None:
            try:
                print("[SAFETY] Sending stop command before shutdown...")
                tcp_socket.sendall(b"S\n")
            except OSError:
                pass

        if camera is not None:
            camera.release()

        if recording_writer is not None:
            recording_writer.release()

        # 2. 等待接收线程退出后再关闭 socket
        time.sleep(0.8)
        if tcp_socket is not None:
            try:
                tcp_socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            tcp_socket.close()
        cv2.destroyAllWindows()
        print("[MAIN] Program closed.")


if __name__ == "__main__":
    main()  