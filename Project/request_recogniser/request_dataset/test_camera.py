from __future__ import annotations

from collections import Counter, deque
from pathlib import Path

import cv2
from ultralytics import YOLO


MODEL_PATH = Path(
    r"D:\Downloads\object_detect\runs\classify\train\weights\best.pt"
)

CAMERA_ID = 0
IMAGE_SIZE = 224

MIN_CONFIDENCE = 0.75

WINDOW_SIZE = 10
MIN_MATCH_COUNT = 7
MIN_AVERAGE_CONFIDENCE = 0.80

CONFIRM_COOLDOWN_FRAMES = 30

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

ROI_LEFT_RATIO = 0.25
ROI_RIGHT_RATIO = 0.75
ROI_TOP_RATIO = 0.20
ROI_BOTTOM_RATIO = 0.85


def get_stable_prediction(
    history: deque[tuple[str, float]],
) -> tuple[str | None, float, int]:
    """根据最近多帧结果判断类别是否稳定。"""

    if len(history) < WINDOW_SIZE:
        return None, 0.0, 0

    valid_results = [
        (class_name, confidence)
        for class_name, confidence in history
        if class_name != "unknown"
    ]

    if not valid_results:
        return None, 0.0, 0

    class_counts = Counter(
        class_name for class_name, _ in valid_results
    )

    candidate_class, match_count = class_counts.most_common(1)[0]

    candidate_confidences = [
        confidence
        for class_name, confidence in valid_results
        if class_name == candidate_class
    ]

    average_confidence = (
        sum(candidate_confidences) / len(candidate_confidences)
    )

    if (
        match_count >= MIN_MATCH_COUNT
        and average_confidence >= MIN_AVERAGE_CONFIDENCE
    ):
        return candidate_class, average_confidence, match_count

    return None, average_confidence, match_count


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"找不到模型文件：{MODEL_PATH}\n"
            "请把 MODEL_PATH 修改成实际的 best.pt 路径。"
        )

    model = YOLO(str(MODEL_PATH))

    print("模型路径：", MODEL_PATH)
    print("模型类别：", model.names)

    camera = cv2.VideoCapture(CAMERA_ID)

    if not camera.isOpened():
        raise RuntimeError(
            "无法打开摄像头，请尝试把 CAMERA_ID 改成 1 或 2。"
        )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    prediction_history: deque[tuple[str, float]] = deque(
        maxlen=WINDOW_SIZE
    )

    last_confirmed_class: str | None = None
    cooldown_frames = 0

    print("分类摄像头测试已启动")
    print("请把一个物品放到中央矩形区域")
    print("按 q 退出，按 c 清空识别历史")

    while True:
        success, frame = camera.read()

        if not success:
            print("无法读取摄像头画面")
            break

        height, width = frame.shape[:2]

        x1 = int(width * ROI_LEFT_RATIO)
        x2 = int(width * ROI_RIGHT_RATIO)
        y1 = int(height * ROI_TOP_RATIO)
        y2 = int(height * ROI_BOTTOM_RATIO)

        roi = frame[y1:y2, x1:x2]

        result = model.predict(
            source=roi,
            imgsz=IMAGE_SIZE,
            device=0,
            verbose=False,
        )[0]

        class_id = int(result.probs.top1)
        confidence = float(result.probs.top1conf)
        raw_class_name = str(model.names[class_id])

        current_class = (
            raw_class_name
            if confidence >= MIN_CONFIDENCE
            else "unknown"
        )

        prediction_history.append((current_class, confidence))

        stable_class, average_confidence, match_count = (
            get_stable_prediction(prediction_history)
        )

        if cooldown_frames > 0:
            cooldown_frames -= 1

        if stable_class is not None:
            status_text = (
                f"CONFIRMED: {stable_class} "
                f"avg={average_confidence:.2f} "
                f"({match_count}/{WINDOW_SIZE})"
            )

            if (
                stable_class != last_confirmed_class
                or cooldown_frames == 0
            ):
                print("=" * 50)
                print(f"Request confirmed: {stable_class}")
                print(
                    f"Average confidence: "
                    f"{average_confidence:.3f}"
                )
                print(
                    f"Matched frames: "
                    f"{match_count}/{WINDOW_SIZE}"
                )

                last_confirmed_class = stable_class
                cooldown_frames = CONFIRM_COOLDOWN_FRAMES
        else:
            status_text = (
                f"WAITING: best={raw_class_name} "
                f"{confidence:.2f} "
                f"({match_count}/{WINDOW_SIZE})"
            )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Current: {current_class} {confidence:.2f}",
            (x1, max(30, y1 - 45)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            status_text,
            (x1, max(60, y1 - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0)
            if stable_class is not None
            else (0, 165, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            "Place one object inside this area",
            (x1 + 10, y1 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(
            "Stationery Classification Test",
            frame,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("c"):
            prediction_history.clear()
            last_confirmed_class = None
            cooldown_frames = 0
            print("已清空多帧识别历史")

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
