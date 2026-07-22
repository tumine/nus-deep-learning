"""
test_pose_realtime.py

Real-time test for YOLO pose hand-raise detection.

Press Q to quit.
"""

import cv2
import torch
from ultralytics import YOLO


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "yolov8n-pose.pt"

# 本地电脑摄像头
CAMERA_SOURCE = 0

# 如果测试树莓派视频流，改成：
# CAMERA_SOURCE = "http://100.84.2.68:5000/video_feed"

PERSON_CONFIDENCE = 0.5
KEYPOINT_CONFIDENCE = 0.3

# 手腕需要至少高于肩膀多少像素，才判定为举手
WRIST_SHOULDER_MARGIN = 15


# COCO Pose keypoint indices
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10


def keypoint_is_valid(
    point,
    confidence_array,
    point_index
):
    """
    Check whether a keypoint has valid coordinates
    and sufficient confidence.
    """

    x, y = point

    if x <= 0 or y <= 0:
        return False

    if confidence_array is not None:

        confidence = float(
            confidence_array[point_index]
        )

        if confidence < KEYPOINT_CONFIDENCE:
            return False

    return True


def is_hand_raised(
    wrist,
    shoulder,
    confidence_array,
    wrist_index,
    shoulder_index
):
    """
    Determine whether one wrist is above its shoulder.
    """

    wrist_valid = keypoint_is_valid(
        wrist,
        confidence_array,
        wrist_index
    )

    shoulder_valid = keypoint_is_valid(
        shoulder,
        confidence_array,
        shoulder_index
    )

    if not wrist_valid or not shoulder_valid:
        return False

    wrist_y = wrist[1]
    shoulder_y = shoulder[1]

    return (
        wrist_y
        < shoulder_y - WRIST_SHOULDER_MARGIN
    )


def draw_point(
    frame,
    point,
    color,
    label
):
    """
    Draw one pose keypoint.
    """

    x, y = map(int, point)

    if x <= 0 or y <= 0:
        return

    cv2.circle(
        frame,
        (x, y),
        6,
        color,
        -1
    )

    cv2.putText(
        frame,
        label,
        (x + 7, y - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA
    )


def main():

    device = 0 if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("YOLO Pose Real-Time Test")
    print("=" * 60)
    print(
        "[TEST] Device:",
        "GPU" if torch.cuda.is_available() else "CPU"
    )
    print(f"[TEST] Model: {MODEL_PATH}")
    print(f"[TEST] Camera source: {CAMERA_SOURCE}")
    print("[TEST] Press Q to quit.")
    print("=" * 60)

    model = YOLO(MODEL_PATH)

    camera = cv2.VideoCapture(CAMERA_SOURCE)

    if not camera.isOpened():
        print("[ERROR] Cannot open camera source.")
        return

    frame_number = 0

    try:

        while True:

            success, frame = camera.read()

            if not success:
                print("[ERROR] Failed to read frame.")
                break

            frame_number += 1

            results = model.predict(
                source=frame,
                conf=PERSON_CONFIDENCE,
                device=device,
                verbose=False
            )

            person_count = 0
            raised_count = 0

            for result in results:

                if (
                    result.boxes is None
                    or result.keypoints is None
                ):
                    continue

                boxes = (
                    result.boxes.xyxy
                    .cpu()
                    .numpy()
                )

                box_confidences = (
                    result.boxes.conf
                    .cpu()
                    .numpy()
                )

                keypoints_xy = (
                    result.keypoints.xy
                    .cpu()
                    .numpy()
                )

                keypoints_conf = None

                if result.keypoints.conf is not None:
                    keypoints_conf = (
                        result.keypoints.conf
                        .cpu()
                        .numpy()
                    )

                for index, person_points in enumerate(
                    keypoints_xy
                ):

                    if index >= len(boxes):
                        continue

                    person_count += 1

                    x1, y1, x2, y2 = map(
                        int,
                        boxes[index]
                    )

                    person_confidence = float(
                        box_confidences[index]
                    )

                    point_confidences = None

                    if keypoints_conf is not None:
                        point_confidences = (
                            keypoints_conf[index]
                        )

                    left_shoulder = person_points[
                        LEFT_SHOULDER
                    ]

                    right_shoulder = person_points[
                        RIGHT_SHOULDER
                    ]

                    left_elbow = person_points[
                        LEFT_ELBOW
                    ]

                    right_elbow = person_points[
                        RIGHT_ELBOW
                    ]

                    left_wrist = person_points[
                        LEFT_WRIST
                    ]

                    right_wrist = person_points[
                        RIGHT_WRIST
                    ]

                    left_raised = is_hand_raised(
                        wrist=left_wrist,
                        shoulder=left_shoulder,
                        confidence_array=point_confidences,
                        wrist_index=LEFT_WRIST,
                        shoulder_index=LEFT_SHOULDER
                    )

                    right_raised = is_hand_raised(
                        wrist=right_wrist,
                        shoulder=right_shoulder,
                        confidence_array=point_confidences,
                        wrist_index=RIGHT_WRIST,
                        shoulder_index=RIGHT_SHOULDER
                    )

                    hand_raised = (
                        left_raised
                        or right_raised
                    )

                    if hand_raised:
                        raised_count += 1

                    box_color = (
                        (0, 255, 0)
                        if hand_raised
                        else (255, 255, 0)
                    )

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        box_color,
                        2
                    )

                    draw_point(
                        frame,
                        left_shoulder,
                        (255, 0, 0),
                        "LS"
                    )

                    draw_point(
                        frame,
                        right_shoulder,
                        (255, 0, 0),
                        "RS"
                    )

                    draw_point(
                        frame,
                        left_elbow,
                        (0, 255, 255),
                        "LE"
                    )

                    draw_point(
                        frame,
                        right_elbow,
                        (0, 255, 255),
                        "RE"
                    )

                    draw_point(
                        frame,
                        left_wrist,
                        (0, 0, 255),
                        "LW"
                    )

                    draw_point(
                        frame,
                        right_wrist,
                        (0, 0, 255),
                        "RW"
                    )

                    # 绘制肩膀到手腕的连线
                    if keypoint_is_valid(
                        left_shoulder,
                        point_confidences,
                        LEFT_SHOULDER
                    ) and keypoint_is_valid(
                        left_wrist,
                        point_confidences,
                        LEFT_WRIST
                    ):

                        cv2.line(
                            frame,
                            tuple(
                                map(
                                    int,
                                    left_shoulder
                                )
                            ),
                            tuple(
                                map(
                                    int,
                                    left_wrist
                                )
                            ),
                            (0, 255, 255),
                            2
                        )

                    if keypoint_is_valid(
                        right_shoulder,
                        point_confidences,
                        RIGHT_SHOULDER
                    ) and keypoint_is_valid(
                        right_wrist,
                        point_confidences,
                        RIGHT_WRIST
                    ):

                        cv2.line(
                            frame,
                            tuple(
                                map(
                                    int,
                                    right_shoulder
                                )
                            ),
                            tuple(
                                map(
                                    int,
                                    right_wrist
                                )
                            ),
                            (0, 255, 255),
                            2
                        )

                    if hand_raised:

                        status = "HAND RAISED"

                    else:

                        status = "NO HAND RAISE"

                    cv2.putText(
                        frame,
                        (
                            f"{status} "
                            f"{person_confidence:.2f}"
                        ),
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        box_color,
                        2,
                        cv2.LINE_AA
                    )

                    # 每隔 10 帧打印一次，避免终端刷屏太快
                    if frame_number % 10 == 0:

                        print(
                            f"[Frame {frame_number}] "
                            f"Person {index + 1} | "
                            f"confidence={person_confidence:.2f} | "
                            f"left_raised={left_raised} | "
                            f"right_raised={right_raised}"
                        )

            cv2.putText(
                frame,
                f"Persons: {person_count}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                frame,
                f"Raised: {raised_count}",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                frame,
                (
                    "GPU"
                    if torch.cuda.is_available()
                    else "CPU"
                ),
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.imshow(
                "YOLO Pose Real-Time Test",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("[TEST] Quit key pressed.")
                break

    except KeyboardInterrupt:
        print("\n[TEST] Interrupted by user.")

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("[TEST] Resources released.")


if __name__ == "__main__":
    main()