"""
main.py

Classroom Assistant Robot Main Controller

Workflow:
PATROL -> SCAN -> APPROACH_STUDENT -> WAIT_CARD -> GO_TEACHER
-> WAIT_LOADING -> RETURN_STUDENT -> WAIT_UNLOAD -> RETURN_PATROL -> PATROL

Temporary keyboard triggers are provided for robot actions that the
robotics group has not implemented yet.
"""

import cv2

from camera import Camera
from card_detector import CardDetector
from hand_detector import HandDetector
from request_manager import RequestManager
from robot_controller import RobotController
from state_machine import StateMachine, RobotState
from task_queue import TaskQueue


CAMERA_URL = "http://100.84.2.68:5000/video_feed"
# CAMERA_URL = 0

HAND_MODEL_PATH = "yolov8n-pose.pt"
WINDOW_NAME = "Classroom Assistant"

DEFAULT_ROUTE_NODE = 1
DEFAULT_APPROACH_SECONDS = 1.5


def print_separator():
    print("=" * 55)


def send_robot_command(robot, command):
    """Send one TCP command and return whether sending succeeded."""

    if robot.busy:
        print("[ROBOT] Controller is busy.")
        return False

    print_separator()
    print("[ROBOT COMMAND]")
    print(command)
    print_separator()

    return bool(robot.execute(command))


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
    """Draw current state and the temporary trigger instruction."""

    state = state_machine.get_state()

    instructions = {
        RobotState.PATROL: "Patrolling - press I at an inspection point",
        RobotState.SCAN: "Scanning hands (J=left, K=right)",
        RobotState.APPROACH_STUDENT: "Approaching student - press A when arrived",
        RobotState.WAIT_CARD: "Waiting for ArUco request card",
        RobotState.GO_TEACHER: "Going to teacher - press T when arrived",
        RobotState.WAIT_LOADING: "Press L after teacher finishes loading",
        RobotState.RETURN_STUDENT: "Returning to student - press S when arrived",
        RobotState.WAIT_UNLOAD: "Press U after student takes the item",
        RobotState.RETURN_PATROL: "Returning to route - press R when rejoined",
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


def main():
    print_separator()
    print("Starting Classroom Assistant...")
    print_separator()

    camera = None

    try:
        camera = Camera(CAMERA_URL)
        hand_detector = HandDetector(model_path=HAND_MODEL_PATH, conf=0.5)
        card_detector = CardDetector()
        request_manager = RequestManager()
        robot = RobotController()
        task_queue = TaskQueue()
        state_machine = StateMachine()

        # Prevent one movement command from being sent every frame.
        command_sent_for_state = None

        print("[MAIN] Temporary triggers:")
        print("  I: inspection point reached")
        print("  J/K: scan left/right")
        print("  A: arrived at student")
        print("  T: arrived at teacher")
        print("  L: loading complete")
        print("  S: returned to student")
        print("  U: unloading complete")
        print("  R: patrol route rejoined")
        print("  Q: quit")

        while True:
            frame = camera.read()

            if frame is None:
                print("[ERROR] Failed to read camera frame.")
                break

            current_state = state_machine.get_state()

            if command_sent_for_state != current_state:
                command_sent_for_state = None

            if current_state == RobotState.PATROL:
                # Real trigger later: Raspberry Pi sends intersection_reached.
                # Temporary trigger now: press I.
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

                    if send_robot_command(robot, command):
                        command_sent_for_state = current_state

                # Real trigger later: arrived_student from Raspberry Pi.
                # Temporary trigger now: press A.

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

                        if send_robot_command(robot, command):
                            command_sent_for_state = current_state
                        else:
                            task_queue.add(task)

                # Real trigger later: arrived_teacher.
                # Temporary trigger now: press T.

            elif current_state == RobotState.WAIT_LOADING:
                # Temporary trigger: press L.
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

                    if send_robot_command(robot, command):
                        command_sent_for_state = current_state

                # Real trigger later: arrived_student.
                # Temporary trigger now: press S.

            elif current_state == RobotState.WAIT_UNLOAD:
                # Temporary trigger: press U.
                pass

            elif current_state == RobotState.RETURN_PATROL:
                if command_sent_for_state is None:
                    command = {
                        "command": "return_patrol",
                        "route_node": state_machine.get_context_value("route_node"),
                        "scan_direction": state_machine.get_context_value("scan_direction"),
                        "approach": state_machine.get_context_value("approach_command"),
                    }

                    if send_robot_command(robot, command):
                        command_sent_for_state = current_state

                # Real trigger later: route_rejoined.
                # Temporary trigger now: press R.

            else:
                print(f"[WARNING] Unknown state: {current_state}")
                state_machine.reset()

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
                    print("[SIMULATION] Robot arrived at student.")
                    state_machine.set_state(RobotState.WAIT_CARD)

            elif key == ord("t"):
                if state_machine.is_state(RobotState.GO_TEACHER):
                    print("[SIMULATION] Robot arrived at teacher.")
                    state_machine.set_state(RobotState.WAIT_LOADING)

            elif key == ord("l"):
                if state_machine.is_state(RobotState.WAIT_LOADING):
                    print("[MAIN] Teacher loading completed.")
                    state_machine.set_state(RobotState.RETURN_STUDENT)

            elif key == ord("s"):
                if state_machine.is_state(RobotState.RETURN_STUDENT):
                    print("[SIMULATION] Robot returned to student.")
                    state_machine.set_state(RobotState.WAIT_UNLOAD)

            elif key == ord("u"):
                if state_machine.is_state(RobotState.WAIT_UNLOAD):
                    print("[MAIN] Student unloading completed.")
                    state_machine.set_state(RobotState.RETURN_PATROL)

            elif key == ord("r"):
                if state_machine.is_state(RobotState.RETURN_PATROL):
                    print("[SIMULATION] Patrol route rejoined.")
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

        cv2.destroyAllWindows()
        print("[MAIN] Program closed.")


if __name__ == "__main__":
    main()
