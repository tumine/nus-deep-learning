"""
main.py

Classroom Request Card Detection
"""

import cv2

from camera import Camera
from card_detector import CardDetector


def main():

    # 初始化
    camera = Camera()

    detector = CardDetector()

    while True:

        # 获取摄像头画面
        frame = camera.read()

        if frame is None:
            break

        # 检测新的请求（只有确认5帧后才会返回）
        results = detector.detect(frame)

        # 处理新的请求
        for result in results:

            print(f"[NEW REQUEST] {result['request']}")

            # 后续这里可以发送给机器人控制程序
            # robot.handle_request(result)

        # 绘制检测结果
        frame = detector.draw(frame)

        # 显示画面
        cv2.imshow("Classroom Assistant", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    # 释放资源
    camera.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":

    main()