"""
main.py

Classroom Assistant Robot Main Controller

Workflow:
PATROL -> SCAN -> APPROACH_STUDENT -> WAIT_CARD -> GO_TEACHER
-> WAIT_LOADING -> RETURN_STUDENT -> WAIT_UNLOAD -> RETURN_PATROL -> PATROL

Automated network triggers are now active: 
Listens to TCP messages from the Raspberry Pi / Robot to transition states.
Temporary keyboard triggers remain as a manual override/fallback.
"""

import cv2
import socket
import threading
import queue
import sys

from camera import Camera
from card_detector_classifier import CardDetector
from hand_detector import HandDetector
from request_manager import RequestManager
# from robot_controller import RobotController # 已被 TCP 网络通信替代
from state_machine import StateMachine, RobotState
from task_queue import TaskQueue

from ui_manager import UIManager
from ui_server import UIServer

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
# ==============================================================================


def print_separator():
    print("=" * 55)


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


def handle_card_event(card_event, request_manager, task_queue, state_machine):
    """Create and store one delivery task from a confirmed card."""

    task = request_manager.create_task(card_event)
    task["student_context"] = state_machine.get_context()

    task_queue.add(task)
    state_machine.set_task(task)

    print_separator()
    print("[CARD EVENT]")
    print(f"Marker ID : {card_event.get('id')}")
    print(f"Request   : {card_event.get('request', 'unknown')}")
    print(f"Center    : {card_event.get('center')}")
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
        RobotState.WAIT_CARD: "Waiting for ArUco request card",
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


def tcp_receive_thread(sock, network_queue, ui_manager):
    """后台独立运行的网络接收线程，专门监听小车发回的状态信息"""
    buffer = ""
    while True:
        try:
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


def main():
    print_separator()
    print("Starting Classroom Assistant...")
    print_separator()

    camera = None
    tcp_socket = None
    network_queue = queue.Queue()

    ui_manager = UIManager()

    ui_server = UIServer(
        ui_manager=ui_manager,
        host="0.0.0.0",
        port=8000,
    )

    ui_server.start_in_thread()

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
        threading.Thread(target=tcp_receive_thread, args=(tcp_socket, network_queue, ui_manager), daemon=True).start()
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

        # 记录上一次发送指令时的状态，防止在同一个状态下一帧一帧疯狂重复发指令
        command_sent_for_state = None

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
                        print("▶️ [自动触发] 小车已到达学生身边，开始识别需求卡 (APPROACH_STUDENT -> WAIT_CARD)")
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
                print("[ERROR] Failed to read camera frame.")
                break

            # 3. 核心业务状态机流转
            current_state = state_machine.get_state()
            if command_sent_for_state != current_state:
                command_sent_for_state = None

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
                card_events = card_detector.detect(frame)
                if card_events:
                    handle_card_event(
                        card_events[0],
                        request_manager,
                        task_queue,
                        state_machine,
                    )

                    card_event = card_events[0]
                    center = card_event.get("center")
                    axis_x = center[0] if center else None
                    axis_y = center[1] if center else None
                    ui_manager.update_request(
                        request_type="物品",
                        description=card_event.get("request", "unknown"),
                        message_id=f"ARUCO-{card_event.get('id', 'unknown')}",
                        axis_x=axis_x,
                        axis_y=axis_y,
                    )
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
        if tcp_socket is not None:
            try:
                print("[SAFETY] Sending stop command before shutdown...")
                tcp_socket.sendall(b"S\n")
            except OSError:
                pass
        if camera is not None:
            camera.release()
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