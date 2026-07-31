# Classroom Assistant Robot - ArUco Request Card Module

## Project Overview

This module is part of the **Classroom Assistant Robot** project.

It is responsible for recognizing **ArUco request cards** shown by children after the robot stops at their desk. Once a request card is confirmed, the module generates a delivery task and sends it to the robot control system.

The current supported classroom requests include:

| Marker ID | Request |
|-----------|---------|
| 0 | Building Blocks |
| 1 | Pencil |
| 2 | Eraser |
| 3 | Teacher Assistance |

---

## Workflow

```
Camera
    │
    ▼
ArUco Detection
    │
    ▼
Request Mapping
    │
    ▼
Task Generation
    │
    ▼
Task Queue
    │
    ▼
Robot Controller
```

The robot workflow is:

1. Patrol along the predefined route.
2. Detect a child raising a hand (implemented by another module).
3. Stop beside the child.
4. Wait for the child to display an ArUco request card.
5. Recognize the request.
6. Generate a delivery task.
7. Return to the teacher's station to collect the requested item.
8. Deliver the item to the child.
9. Resume patrol.

---

## Project Structure

```
card_detector/

│── main.py                 # Main program
│── camera.py               # Camera interface
│── card_detector.py        # ArUco detection module
│── request_mapping.py      # Marker ID → classroom request
│── request_manager.py      # Request → delivery task
│── task_queue.py           # Robot task queue
│── robot_controller.py     # Robot task execution interface
│── state_machine.py        # Robot state manager (future integration)
│── config.py               # Configuration parameters
```

---

## Current Features

- ArUco marker detection
- Multi-frame confirmation
- Classroom request mapping
- Delivery task generation
- Task queue management
- Robot controller interface
- Modular architecture for future integration

---

## Future Integration

This module will be integrated with:

### YOLO Pose Detection

Detect children raising their hands during patrol.

### Robot Navigation

Control the robot to:

- Return to the teacher
- Collect requested materials
- Navigate back to the child
- Resume patrol

### Raspberry Pi

The camera stream can be provided either by:

- Local USB camera
- Raspberry Pi video streaming

Only the camera source needs to be changed.

---

## Dependencies

- Python 3.10+
- OpenCV
- OpenCV ArUco Module

Install:

```bash
pip install opencv-python
pip install opencv-contrib-python
```

---

## Usage

Run

```bash
python main.py
```

Press **Q** to quit.

---

## Notes

This module currently focuses on ArUco-based classroom request recognition.

The robot motion control, YOLO hand-raising detection, and Raspberry Pi communication will be integrated in subsequent development stages.

---

## Version

Current Version: v0.2

### Completed

- Camera module
- ArUco detector
- Request mapping
- Task generation
- Task queue
- Robot controller interface

### Under Development

- YOLO hand-raising detection
- State machine
- Raspberry Pi video streaming
- Robot communication
- Material delivery workflow