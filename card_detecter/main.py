"""
main.py

Classroom Assistant Robot Main Controller

Workflow:
1. PATROL
   Detect a student raising their hand.

2. WAIT_CARD
   Wait for the student to show an ArUco request card.

3. GO_TEACHER
   Send the delivery task to the robot and move to the teacher station.

4. WAIT_LOADING
   Wait for the teacher to load the requested item.

5. GO_CHILD
   Return to the child.

6. WAIT_UNLOAD
   Wait for the child to take the item.

7. RETURN_PATROL
   Clear the current task and return to patrol mode.
"""

import cv2

from camera import Camera
from card_detector import CardDetector
from hand_detector import HandDetector
from request_manager import RequestManager
from robot_controller import RobotController
from state_machine import StateMachine, RobotState
from task_queue import TaskQueue


# ============================================================
# Configuration
# ============================================================

# CAMERA_URL = "http://100.84.2.68:5000/video_feed"
CAMERA_URL = 0

# 修改成实际的 YOLO 举手识别模型路径
HAND_MODEL_PATH = "yolov8_hand_raise.pt"

WINDOW_NAME = "Classroom Assistant"


# ============================================================
# Helper functions
# ============================================================

def print_separator():
    """Print a separator line in the terminal."""
    print("=" * 50)


def handle_hand_event(hand_event, state_machine):
    """
    Handle a confirmed hand-raise event.

    Parameters
    ----------
    hand_event : dict
        Event returned by HandDetector.

    state_machine : StateMachine
        Controls the current robot workflow state.
    """

    target = hand_event.get("target")
    confidence = hand_event.get("confidence", 0.0)

    print_separator()
    print("[HAND EVENT]")
    print(f"Target     : {target}")
    print(f"Confidence : {confidence:.2f}")
    print("Student raised hand.")
    print("Robot should move towards the student.")
    print_separator()

    # TODO:
    # 等运动控制接口完成后，可以改成：
    #
    # robot.move_to_student(target)
    #
    # 目前先模拟机器人已经到达学生附近。

    print("[MAIN] Robot arrived near the student.")
    print("[MAIN] Waiting for an ArUco request card.")

    state_machine.set_state(RobotState.WAIT_CARD)


def handle_card_event(
    card_event,
    request_manager,
    task_queue,
    state_machine
):
    """
    Convert a confirmed ArUco card event into a delivery task.

    Parameters
    ----------
    card_event : dict
        Confirmed ArUco detection result.

    request_manager : RequestManager
        Converts card results into task dictionaries.

    task_queue : TaskQueue
        Stores tasks waiting to be executed.

    state_machine : StateMachine
        Controls the current robot workflow state.
    """

    request = card_event.get("request", "unknown")
    marker_id = card_event.get("id")
    center = card_event.get("center")

    task = request_manager.create_task(card_event)

    task_queue.add(task)

    # 保存当前正在执行的任务
    state_machine.set_task(task)

    print_separator()
    print("[CARD EVENT]")
    print(f"Marker ID : {marker_id}")
    print(f"Request   : {request}")
    print(f"Center    : {center}")
    print(f"Task      : {task}")
    print("Task added to TaskQueue.")
    print_separator()

    state_machine.set_state(RobotState.GO_TEACHER)


def send_task_to_robot(robot, task_queue, state_machine):
    """
    Send the next queued task to the robot.

    This function is used in the GO_TEACHER state.
    """

    if robot.busy:
        return

    if not task_queue.has_task():
        print("[WARNING] GO_TEACHER state entered, but TaskQueue is empty.")

        state_machine.clear_task()
        state_machine.set_state(RobotState.PATROL)

        return

    task = task_queue.next_task()

    print_separator()
    print("[TASK EXECUTION]")
    print(f"Sending task to robot: {task}")
    print_separator()

    try:
        robot.execute(task)

    except Exception as error:
        print(f"[ERROR] Failed to send task to robot: {error}")

        # 发送失败时重新放回队列，避免任务丢失。
        task_queue.add(task)

        return

    print("[MAIN] Task sent successfully.")
    print("[MAIN] Simulating arrival at teacher station.")

    state_machine.set_state(RobotState.WAIT_LOADING)


def draw_system_status(frame, state_machine):
    """
    Draw the current state and instruction on the video frame.
    """

    state_name = state_machine.state.name

    cv2.putText(
        frame,
        f"STATE: {state_name}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    if state_machine.state == RobotState.PATROL:
        instruction = "Detecting raised hands"

    elif state_machine.state == RobotState.WAIT_CARD:
        instruction = "Waiting for ArUco card"

    elif state_machine.state == RobotState.GO_TEACHER:
        instruction = "Going to teacher station"

    elif state_machine.state == RobotState.WAIT_LOADING:
        instruction = "Press L after teacher loads item"

    elif state_machine.state == RobotState.GO_CHILD:
        instruction = "Returning to child"

    elif state_machine.state == RobotState.WAIT_UNLOAD:
        instruction = "Press U after child takes item"

    elif state_machine.state == RobotState.RETURN_PATROL:
        instruction = "Returning to patrol route"

    else:
        instruction = "Unknown state"

    cv2.putText(
        frame,
        instruction,
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return frame


# ============================================================
# Main program
# ============================================================

def main():
    """Run the classroom assistant robot control program."""

    print_separator()
    print("Starting Classroom Assistant...")
    print_separator()

    camera = None

    try:
        # ----------------------------------------------------
        # Initialize components
        # ----------------------------------------------------

        camera = Camera(CAMERA_URL)

        hand_detector = HandDetector(
            model_path=HAND_MODEL_PATH,
            conf=0.5
        )

        card_detector = CardDetector()

        request_manager = RequestManager()

        robot = RobotController()

        task_queue = TaskQueue()

        state_machine = StateMachine()

        # StateMachine 默认已经是 PATROL。
        # 这里保留明确的启动信息即可。
        print(f"[MAIN] Initial state: {state_machine.state.name}")
        print("[MAIN] All components initialized.")
        print("[MAIN] Press Q to quit.")
        print("[MAIN] Press L after teacher finishes loading.")
        print("[MAIN] Press U after child finishes unloading.")

        # ----------------------------------------------------
        # Main loop
        # ----------------------------------------------------

        while True:

            frame = camera.read()

            if frame is None:
                print("[ERROR] Failed to read camera frame.")
                break

            current_state = state_machine.get_state()

            # =================================================
            # State 1: PATROL
            # =================================================

            if current_state == RobotState.PATROL:

                hand_events = hand_detector.detect(frame)

                if hand_events:
                    # 一次只处理一个已经确认的举手事件。
                    hand_event = hand_events[0]

                    handle_hand_event(
                        hand_event=hand_event,
                        state_machine=state_machine
                    )

                frame = hand_detector.draw(frame)

            # =================================================
            # State 2: WAIT_CARD
            # =================================================

            elif current_state == RobotState.WAIT_CARD:

                card_events = card_detector.detect(frame)

                if card_events:
                    # 一次只处理一张已经确认的请求卡。
                    card_event = card_events[0]

                    handle_card_event(
                        card_event=card_event,
                        request_manager=request_manager,
                        task_queue=task_queue,
                        state_machine=state_machine
                    )

                frame = card_detector.draw(frame)

            # =================================================
            # State 3: GO_TEACHER
            # =================================================

            elif current_state == RobotState.GO_TEACHER:

                send_task_to_robot(
                    robot=robot,
                    task_queue=task_queue,
                    state_machine=state_machine
                )

            # =================================================
            # State 4: WAIT_LOADING
            # =================================================

            elif current_state == RobotState.WAIT_LOADING:

                # 当前使用键盘 L 模拟老师装载完成。
                # 后续可替换为按钮、语音或传感器输入。
                pass

            # =================================================
            # State 5: GO_CHILD
            # =================================================

            elif current_state == RobotState.GO_CHILD:

                print("[MAIN] Simulating movement back to the child.")

                # TODO:
                # 后续接入真实运动接口，例如：
                #
                # robot.go_to_child(state_machine.get_task())

                state_machine.set_state(RobotState.WAIT_UNLOAD)

            # =================================================
            # State 6: WAIT_UNLOAD
            # =================================================

            elif current_state == RobotState.WAIT_UNLOAD:

                # 当前使用键盘 U 模拟学生取走物品。
                # 后续可替换为按钮、语音或传感器输入。
                pass

            # =================================================
            # State 7: RETURN_PATROL
            # =================================================

            elif current_state == RobotState.RETURN_PATROL:

                print("[MAIN] Simulating return to patrol route.")

                # TODO:
                # 后续接入真实运动接口，例如：
                #
                # robot.return_patrol()

                state_machine.clear_task()
                state_machine.set_state(RobotState.PATROL)

            # =================================================
            # Unknown state protection
            # =================================================

            else:

                print(f"[WARNING] Unknown robot state: {current_state}")

                state_machine.clear_task()
                state_machine.set_state(RobotState.PATROL)

            # -------------------------------------------------
            # Draw current system status
            # -------------------------------------------------

            frame = draw_system_status(
                frame=frame,
                state_machine=state_machine
            )

            # -------------------------------------------------
            # Display frame
            # -------------------------------------------------

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF

            # -------------------------------------------------
            # Keyboard controls
            # -------------------------------------------------

            if key == ord("q"):

                print("[MAIN] Quit key pressed.")
                break

            elif key == ord("l"):

                if state_machine.state == RobotState.WAIT_LOADING:

                    print("[MAIN] Teacher loading completed.")

                    state_machine.set_state(RobotState.GO_CHILD)

            elif key == ord("u"):

                if state_machine.state == RobotState.WAIT_UNLOAD:

                    print("[MAIN] Child unloading completed.")

                    state_machine.set_state(RobotState.RETURN_PATROL)

    except KeyboardInterrupt:

        print("\n[MAIN] Program interrupted by user.")

    except Exception as error:

        print(f"[FATAL ERROR] {error}")
        raise

    finally:

        # ----------------------------------------------------
        # Release resources
        # ----------------------------------------------------

        print("[MAIN] Releasing resources...")

        if camera is not None:
            camera.release()

        cv2.destroyAllWindows()

        print("[MAIN] Program closed.")


if __name__ == "__main__":
    main()