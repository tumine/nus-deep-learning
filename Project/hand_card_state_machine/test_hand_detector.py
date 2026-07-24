import cv2
import torch
from ultralytics import YOLO


MODEL_PATH = "yolov8_hand_raise.pt"


def main():

    model = YOLO(MODEL_PATH)

    device = 0 if torch.cuda.is_available() else "cpu"

    print(
        "[TEST] Device:",
        "GPU" if torch.cuda.is_available() else "CPU"
    )

    print("[TEST] Model classes:", model.names)

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    while True:

        success, frame = camera.read()

        if not success:
            print("[ERROR] Cannot read frame.")
            break

        results = model.predict(
            source="clear.png",
            conf=0.25,
            save=True
        )

        annotated_frame = results[0].plot()

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                class_name = model.names[class_id]

                print(
                    f"Detected: {class_name}, "
                    f"confidence: {confidence:.2f}"
                )

        cv2.imshow(
            "Custom Hand Raise Model Test",
            annotated_frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()