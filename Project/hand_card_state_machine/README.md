# Classroom Assistant Robot

An intelligent classroom assistant robot that autonomously patrols the classroom, detects students raising their hands using YOLOv8 Pose, recognizes ArUco request cards, and delivers classroom items between students and the teacher.

The system combines computer vision, TCP communication, and a finite state machine (FSM) to coordinate the Raspberry Pi mobile robot and the PC vision controller.

---

# Features

- Autonomous classroom patrol
- Automatic left/right scanning at each patrol point
- YOLOv8 Pose hand-raising detection
- ArUco request card recognition
- TCP socket communication between PC and Raspberry Pi
- Event-driven finite state machine
- Automatic delivery workflow
- Automatic return to patrol after task completion

---

# Project Structure

```
hand_card_state_machine/
│
├── main.py                  # PC main controller
├── camera.py                # Camera interface
├── hand_detector.py         # YOLO Pose detector
├── hand_tracker.py          # Hand tracking
├── card_detector.py         # ArUco detector
├── request_manager.py       # Request generation
├── request_mapping.py       # Marker mapping
├── state_machine.py         # Robot state machine
├── task_queue.py            # Delivery task queue
├── config.py                # Global configuration
│
├── ws_car_control.py        # Raspberry Pi robot controller
├── webstream.py             # MJPEG camera server
│
├── send_test_event.py       # TCP communication test
├── test_hand_detector.py
│
├── requirements.txt
└── yolov8n-pose.pt
```

---

# System Architecture

```
                Raspberry Pi Robot
             (Motion Controller)

        ▲                         │
        │ Robot Status            │ Robot Commands
        │                         ▼
+--------------------------------------------------+
|                    main.py                       |
|          Finite State Machine (FSM)             |
+--------------------------------------------------+
          │                         │
          │                         │
          ▼                         ▼
   YOLOv8 Pose                ArUco Detector
 Hand Raise Detection       Request Card Detection
          │                         │
          └──────────┬──────────────┘
                     ▼
               Request Manager
                     │
                 Task Queue
```

---

# State Machine

```
PATROL
    │
    ▼
SCAN
    │
    ▼
APPROACH_STUDENT
    │
    ▼
WAIT_CARD
    │
    ▼
GO_TEACHER
    │
    ▼
WAIT_LOADING
    │
    ▼
RETURN_STUDENT
    │
    ▼
WAIT_UNLOAD
    │
    ▼
RETURN_PATROL
    │
    ▼
PATROL
```

---

# Patrol Workflow

```
Robot Patrol

      │
      ▼

Reach Patrol Point

      │
      ▼

Turn Left

      │
      ▼

Send:

scan_started:left

      │
      ▼

PC enters SCAN

      │
      ▼

YOLO Pose detects raised hand

      │
      ├──────────────┐
      │              │
No hand          Hand detected
      │              │
      ▼              ▼

Turn Right      APPROACH_STUDENT

      │
      ▼

Send:

scan_started:right

      │
      ▼

Still no hand

      │
      ▼

Send:

scan_finished

      │
      ▼

Continue PATROL
```

---

# Delivery Workflow

```
Student raises hand

        │
        ▼

APPROACH_STUDENT

        │
        ▼

WAIT_CARD

        │
        ▼

ArUco detected

        │
        ▼

GO_TEACHER

        │
        ▼

WAIT_LOADING

        │
        ▼

RETURN_STUDENT

        │
        ▼

WAIT_UNLOAD

        │
        ▼

RETURN_PATROL

        │
        ▼

PATROL
```

---

# TCP Communication

## Raspberry Pi → PC

The Raspberry Pi actively reports robot status to the PC.

```
scan_started:left:1

scan_started:right:1

scan_finished:1

arrived_student

arrived_teacher

route_rejoined
```

These messages automatically trigger state transitions inside the PC controller.

---

## PC → Raspberry Pi

The PC sends robot action commands.

```
approach_student

go_teacher

return_student

return_patrol
```

These commands are received by the Raspberry Pi and executed by the robot controller.

---

# Automatic Scan Trigger

Unlike previous versions that required manually pressing **I** to enter the scanning state, the current implementation automatically starts scanning.

Workflow:

```
Robot reaches patrol point

↓

Turn left

↓

Arduino returns "Done"

↓

Raspberry Pi sends

scan_started:left

↓

PC automatically enters SCAN

↓

YOLO Pose begins detecting raised hands
```

If no hand is detected:

```
Turn right

↓

scan_started:right

↓

Continue scanning
```

If neither side contains a raised hand:

```
scan_finished

↓

PATROL
```

---

# Running the System

## 1. Start Raspberry Pi

Run the camera server:

```bash
python webstream.py
```

Run the robot controller:

```bash
python ws_car_control.py
```

---

## 2. Start PC

Run the main controller:

```bash
python main.py
```

The PC automatically connects to the Raspberry Pi through TCP.

---

# Dependencies

- Python 3.10+
- OpenCV
- Ultralytics YOLOv8
- PySerial
- NumPy

Install:

```bash
pip install -r requirements.txt
```

---

# Future Improvements

- Automatic intersection detection using onboard sensors
- Dynamic path planning
- Multi-task scheduling
- Voice interaction
- Multiple robot collaboration
- Hardware emergency stop support