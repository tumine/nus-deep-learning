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

from camera import Camera
from card_detector import CardDetector
from hand_detector import HandDetector
from request_manager import RequestManager
# from robot_controller import RobotController # 已被 TCP 网络通信替代
from state_machine import StateMachine, RobotState
from task_queue import TaskQueue

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
        RobotState.WAIT_LOADING: "Press L after teacher finishes loading",
        RobotState.RETURN_STUDENT: "Returning to student - waiting for robot arrival",
        RobotState.WAIT_UNLOAD: "Press U after student takes the item",
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


def tcp_receive_thread(sock, network_queue):
    """后台独立运行的网络接收线程，专门监听小车发回的状态信息"""
    buffer = ""
    while True:
        try:
            data = sock.recv(1024).decode('utf-8')
            if not data:
                print("⚠️ [网络通信] 与小车的连接已断开，请检查网络！")
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
            break


def main():
    print_separator()
    print("Starting Classroom Assistant...")
    print_separator()

    camera = None
    tcp_socket = None
    network_queue = queue.Queue()

    # 尝试连接小车的 TCP Server
    print(f"🔌 正在尝试连接到小车控制端 ({ROBOT_IP}:{ROBOT_PORT})...")
    try:
        tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp_socket.settimeout(5.0)  # 连接超时时间 5 秒
        tcp_socket.connect((ROBOT_IP, ROBOT_PORT))
        tcp_socket.settimeout(None) # 连接成功后取消超时，改为持续阻塞监听
        print("✅ 成功连接到小车网络！")
        
        # 启动后台接收线程
        threading.Thread(target=tcp_receive_thread, args=(tcp_socket, network_queue), daemon=True).start()
    except Exception as e:
        print(f"❌ 无法连接到小车: {e}")
        print("⚠️ 电脑端将以【离线/单机模式】运行，网络自动触发将失效，只能使用键盘按键模拟。")
        tcp_socket = None

    try:
        camera = Camera(CAMERA_URL)
        hand_detector = HandDetector(model_path=HAND_MODEL_PATH, conf=0.5)
        card_detector = CardDetector()
        request_manager = RequestManager()
        task_queue = TaskQueue()
        state_machine = StateMachine()

        # 记录上一次发送指令时的状态，防止在同一个状态下一帧一帧疯狂重复发指令
        command_sent_for_state = None

        print("\n[MAIN] 状态触发控制说明:")
        print("  ▶ 正常情况：小车通过网络自动发送触发信号，自动跳转流程。")
        print("  ▶ 键盘测试：如果没连上小车，可使用以下按键手动模拟触发：")
        print("    I: inspection point reached (开启扫描)")
        print("    J/K: scan left/right (左右转头扫描)")
        print("    A: 模拟抵达学生 (arrived_student)")
        print("    T: 模拟抵达老师 (arrived_teacher)")
        print("    L: 老师已放好物品 (loading complete)")
        print("    S: 模拟返回学生 (returned to student)")
        print("    U: 学生已取走物品 (unloading complete)")
        print("    R: 模拟回到主路线 (patrol route rejoined)")
        print("    Q: 退出程序\n")

        while True:
            # 1. 优先处理网络发来的小车状态事件
            while not network_queue.empty():
                net_msg = network_queue.get()
                current_state = state_machine.get_state()

                if net_msg == "intersection_reached":
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
                # 老师放好物品，需人工按 L 键确认
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
                # 学生拿走物品，需人工按 U 键确认
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
                    print("[SIMULATION] Scanning left.")

            elif key == ord("k"):
                if state_machine.is_state(RobotState.SCAN):
                    state_machine.update_context(scan_direction="right")
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

            elif key == ord("s"):
                if state_machine.is_state(RobotState.RETURN_STUDENT):
                    print("[SIMULATION - 手动模拟] Robot returned to student.")
                    state_machine.set_state(RobotState.WAIT_UNLOAD)

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
        if camera is not None:
            camera.release()
        if tcp_socket is not None:
            tcp_socket.close()
        cv2.destroyAllWindows()
        print("[MAIN] Program closed.")


if __name__ == "__main__":
    main()