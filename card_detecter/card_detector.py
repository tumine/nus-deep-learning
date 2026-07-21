"""
card_detector.py

Detect classroom request cards.

Author : DL-V2
"""

import cv2

from config import *
from request_mapping import get_request


class CardDetector:

    def __init__(self):

        # ArUco detector
        dictionary = cv2.aruco.getPredefinedDictionary(
            ARUCO_DICT
        )

        parameters = cv2.aruco.DetectorParameters()

        self.detector = cv2.aruco.ArucoDetector(
            dictionary,
            parameters
        )

        # 每个Marker自己的状态
        #
        # marker_id :
        # {
        #     "count":0,
        #     "confirmed":False
        # }
        #
        self.marker_states = {}

    # ============================================================
    # Detect cards
    # ============================================================

    def detect(self, frame):

        confirmed_results = []

        corners, ids, rejected = self.detector.detectMarkers(frame)

        # 没检测到Marker
        if ids is None:

            self.marker_states.clear()

            return confirmed_results

        ids = ids.flatten()

        current_ids = set(ids)

        # 删除已经离开画面的Marker
        for marker_id in list(self.marker_states.keys()):

            if marker_id not in current_ids:

                del self.marker_states[marker_id]

        # 遍历检测结果
        for markerCorner, markerID in zip(corners, ids):

            markerID = int(markerID)

            # ---------- 初始化 ----------
            if markerID not in self.marker_states:

                self.marker_states[markerID] = {

                    "count": 0,

                    "confirmed": False

                }

            state = self.marker_states[markerID]

            # 连续检测帧数
            state["count"] += 1

            pts = markerCorner.reshape((4, 2)).astype(int)

            topLeft, topRight, bottomRight, bottomLeft = pts

            cX = int((topLeft[0] + bottomRight[0]) / 2)

            cY = int((topLeft[1] + bottomRight[1]) / 2)

            result = {

                "id": markerID,

                "request": get_request(markerID),

                "center": (cX, cY),

                "corners": pts,

                "count": state["count"],

                "confirmed": state["confirmed"]

            }

            # 已确认，不再重复发送
            if state["confirmed"]:

                continue

            # 达到确认条件
            if state["count"] >= CONFIRM_FRAMES:

                state["confirmed"] = True

                result["confirmed"] = True

                confirmed_results.append(result)

        return confirmed_results

    # ============================================================
    # Draw Marker
    # ============================================================

    def draw(self, frame):

        corners, ids, rejected = self.detector.detectMarkers(frame)

        if ids is None:

            return frame

        ids = ids.flatten()

        for markerCorner, markerID in zip(corners, ids):

            markerID = int(markerID)

            pts = markerCorner.reshape((4, 2)).astype(int)

            topLeft, topRight, bottomRight, bottomLeft = pts

            cv2.line(frame, tuple(topLeft), tuple(topRight), (0, 255, 0), 2)
            cv2.line(frame, tuple(topRight), tuple(bottomRight), (0, 255, 0), 2)
            cv2.line(frame, tuple(bottomRight), tuple(bottomLeft), (0, 255, 0), 2)
            cv2.line(frame, tuple(bottomLeft), tuple(topLeft), (0, 255, 0), 2)

            cX = int((topLeft[0] + bottomRight[0]) / 2)

            cY = int((topLeft[1] + bottomRight[1]) / 2)

            cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)

            if markerID in self.marker_states:

                state = self.marker_states[markerID]

                if state["confirmed"]:

                    text = f"{get_request(markerID)} ✓"

                else:

                    text = f"{get_request(markerID)} ({state['count']}/{CONFIRM_FRAMES})"

            else:

                text = get_request(markerID)

            cv2.putText(

                frame,

                text,

                (topLeft[0], topLeft[1] - 15),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (255, 0, 0),

                2

            )

        return frame