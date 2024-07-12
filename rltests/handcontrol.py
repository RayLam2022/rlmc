"""
@File    :   handcontrol.py
@Time    :   2024/07/10 17:04:26
@Author  :   RayLam
@Contact :   1027196450@qq.com
"""

import argparse
import math
import time
import copy

import cv2
import pynput
from pynput.mouse import Button
from pynput.keyboard import Key
import numpy as np
import mediapipe as md
from PIL import ImageGrab


parser = argparse.ArgumentParser("handcontrol")
parser.add_argument(
    "-s",
    "--is_show",
    action="store_true",
    help="show or not",
)
parser.add_argument(
    "-f",
    "--is_show_face",
    action="store_true",
)
parser.add_argument(
    "-d",
    "--delay",
    type=float,
    default=0.2,
    help="delay seconds",
)

args = parser.parse_args()


class Command:
    def __init__(self):
        self.mouse = pynput.mouse.Controller()
        self.keyboard = pynput.keyboard.Controller()

    def _distance(self, point1, point2):
        return (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2

    def condition(self, collector, img, screen_width, screen_height):
        """judge the condition to do which command

        Args:
            collector (Dict):dict collected mediapipe results, including hand landmarks
            hand_collector = {
            0: ["WRIST", x,y]
            4: ["THUMB_TIP",x,y]
            ...
            }

            img (np.ndarray): image captured by camera

        Returns:
            img (np.ndarray)
        """
        if len(collector) >= 2:  # two hands catched by the camera
            ...

        else:  # only one hand catched by the camera
            key = list(collector.keys())[0]
            collector = collector[key]
            wrist = collector[0][1:]
            index_finger_mcp = collector[5][1:]
            thumb_tip = collector[4][1:]
            index_finger_tip = collector[8][1:]
            middle_finger_tip = collector[12][1:]
            middle_finger_pip = collector[10][1:]
            ring_finger_tip = collector[16][1:]
            # pinky_tip= collector[20][1:]

            base = self._distance(wrist, index_finger_mcp) * 0.75  # 基准距离
            if (
                self._distance(thumb_tip, middle_finger_pip) * 1.5 < base
                and self._distance(index_finger_tip, index_finger_mcp) < base
                and self._distance(middle_finger_tip, index_finger_mcp) * 0.85 < base
                and self._distance(ring_finger_tip, index_finger_mcp) * 0.7 < base
            ):
                # 握拳 移动
                self._move(wrist)

            elif (
                self._distance(thumb_tip, middle_finger_pip) * 1.5 >= base
                and self._distance(index_finger_tip, index_finger_mcp) < base
                and self._distance(middle_finger_tip, index_finger_mcp) * 0.85 < base
                and self._distance(ring_finger_tip, index_finger_mcp) * 0.7 < base
            ):
                # 竖拇指 滚动 手腕在屏幕下方1/4处往下滾动，否则往上滚动
                self._scroll(wrist, screen_height)

            elif (
                self._distance(thumb_tip, index_finger_mcp) * 1.5 > base
                and self._distance(thumb_tip, index_finger_tip) * 1.5 < base
                and self._distance(index_finger_tip, index_finger_mcp) * 1.8 >= base
                and self._distance(middle_finger_tip, index_finger_mcp) * 1.3 >= base
                and self._distance(ring_finger_tip, index_finger_mcp) * 1.1 >= base
            ):
                # 五指头合抓 压下按键（用于拖拽，要用移动和释放配合）  不灵敏待调整
                self._drag(wrist)

            elif (
                self._distance(thumb_tip, index_finger_mcp) * 1.5 > base
                and self._distance(thumb_tip, index_finger_tip) * 1.2 > base
                and self._distance(index_finger_tip, index_finger_mcp) * 1.5 >= base
                and self._distance(middle_finger_tip, index_finger_mcp) * 1.1 >= base
                and self._distance(ring_finger_tip, index_finger_mcp) * 0.9 >= base
            ):
                # 五指头分开 释放 不灵敏待调整
                self._release()

            elif (
                self._distance(thumb_tip, middle_finger_pip) * 1.3 < base
                and self._distance(index_finger_tip, index_finger_mcp) * 1.3 >= base
                and self._distance(middle_finger_tip, index_finger_mcp) * 0.85 < base
                and self._distance(ring_finger_tip, index_finger_mcp) * 0.7 < base
            ):
                # 竖食指 单击
                self._lclick()

            elif (
                self._distance(thumb_tip, index_finger_mcp) * 1.5 > base
                and self._distance(thumb_tip, index_finger_tip) * 1.2 > base
                and self._distance(index_finger_tip, index_finger_mcp) * 1.5 > base
                and self._distance(middle_finger_tip, index_finger_mcp) * 0.85 < base
                and self._distance(ring_finger_tip, index_finger_mcp) * 0.7 < base
            ):
                # 竖拇指，食指 双击
                self._double_click()

            elif (
                self._distance(thumb_tip, middle_finger_pip) * 1.2 < base
                and self._distance(index_finger_tip, index_finger_mcp) * 1.5 >= base
                and self._distance(middle_finger_tip, index_finger_mcp) * 1.2 >= base
                and self._distance(ring_finger_tip, index_finger_mcp) * 0.7 < base
            ):
                # 竖食指中指 右击
                self._rclick()

            elif (
                self._distance(thumb_tip, index_finger_mcp) * 1.3 > base
                and self._distance(thumb_tip, index_finger_tip) > base
                and self._distance(index_finger_tip, index_finger_mcp) * 1.5 >= base
                and self._distance(middle_finger_tip, index_finger_mcp) * 0.9 > base
                and self._distance(ring_finger_tip, index_finger_mcp) * 0.75 < base
            ):
                # 竖拇指，食指，中指 按alt + tab
                self._switch()

            else:
                pass

        return img

    def _move(self, point):
        self.mouse.position = (point[0], point[1])
        # time.sleep(1)

    def _lclick(self):
        self.mouse.click(Button.left)
        time.sleep(1)

    def _rclick(self):
        self.mouse.click(Button.right)
        time.sleep(1)

    def _double_click(self):
        self.mouse.click(Button.left, 2)
        time.sleep(1)

    def _drag(self, point):
        self.mouse.press(Button.left)
        time.sleep(1)

    def _release(self):
        self.mouse.release(Button.left)
        time.sleep(1)

    def _scroll(self, point, screen_height):
        if point[1] > screen_height * 0.75:
            self.mouse.scroll(0, -1)
        else:
            self.mouse.scroll(0, 1)

    def _switch(self):
        self.keyboard.press(Key.alt)
        self.keyboard.press(Key.tab)
        self.keyboard.release(Key.alt)
        self.keyboard.release(Key.tab)
        time.sleep(1)

    def _sound_control(self):
        pass


class HandControl(Command):
    def __init__(self, camera_id=0):
        super().__init__()
        self.cap = cv2.VideoCapture(camera_id)
        self.cap_width = int(self.cap.get(3))
        self.cap_height = int(self.cap.get(4))
        # self.cap_fps = self.cap.get(7)
        self.hand_detector = md.solutions.hands.Hands(max_num_hands=2)
        self.md_hands = md.solutions.hands

        self.face_mesh = md.solutions.face_mesh.FaceMesh(refine_landmarks=True)

        self.zone_size = 0.6
        self.bias = 0.1
        self.width_bias = int(self.cap_width * self.bias)
        self.height_bias = int(self.cap_height * self.bias)
        action_zone_width = int(self.cap_width * self.zone_size)
        action_zone_height = int(self.cap_height * self.zone_size)
        self.action_zone = [
            self.cap_width - action_zone_width - self.width_bias,
            self.cap_height - action_zone_height - self.height_bias,
            self.cap_width - self.width_bias,
            self.cap_height - self.height_bias,
        ]

        self.screen_width, self.screen_height = self._get_screen_size()
        self.area_radius = int(min(self.screen_width, self.screen_height) * 0.1)
        self.scale = (
            self.screen_width / action_zone_width,
            self.screen_height / action_zone_height,
        )

        self.hand_collector = {
            0: "WRIST",
            4: "THUMB_TIP",
            5: "INDEX_FINGER_MCP",
            8: "INDEX_FINGER_TIP",
            10: "MIDDLE_FINGER_PIP",
            12: "MIDDLE_FINGER_TIP",
            16: "RING_FINGER_TIP",
            20: "PINKY_TIP",
        }

        # self.hand_dict = {
        #     "WRIST": 0,
        #     "THUMB_CMC": 1,
        #     "THUMB_MCP": 2,
        #     "THUMB_IP": 3,
        #     "THUMB_TIP": 4,
        #     "INDEX_FINGER_MCP": 5,
        #     "INDEX_FINGER_PIP": 6,
        #     "INDEX_FINGER_DIP": 7,
        #     "INDEX_FINGER_TIP": 8,
        #     "MIDDLE_FINGER_MCP": 9,
        #     "MIDDLE_FINGER_PIP": 10,
        #     "MIDDLE_FINGER_DIP": 11,
        #     "MIDDLE_FINGER_TIP": 12,
        #     "RING_FINGER_MCP": 13,
        #     "RING_FINGER_PIP": 14,
        #     "RING_FINGER_DIP": 15,
        #     "RING_FINGER_TIP": 16,
        #     "PINKY_MCP": 17,
        #     "PINKY_PIP": 18,
        #     "PINKY_DIP": 19,
        #     "PINKY_TIP": 20,
        # }

    def _get_screen_size(self):
        screen = ImageGrab.grab()
        screen_width, screen_height = screen.size
        return int(screen_width), int(screen_height)

    def _hand(self, img):
        self.hand_data = self.hand_detector.process(img)
        cv2.rectangle(
            img,
            self.action_zone[:2],
            self.action_zone[2:],
            (170, 234, 242),
            5,
            lineType=cv2.LINE_AA,
        )
        if self.hand_data.multi_hand_landmarks:
            hands_collector = dict()
            for i, handlms in enumerate(self.hand_data.multi_hand_landmarks):
                if args.is_show:
                    md.solutions.drawing_utils.draw_landmarks(
                        img, handlms, self.md_hands.HAND_CONNECTIONS
                    )

                if i == 1:
                    left_hand_collector = dict()
                    # hand_dict = dict()
                    for idx, lm in enumerate(handlms.landmark):
                        # hand_dict[md.solutions.hands.HandLandmark(idx).name] = idx
                        # print(hand_dict)
                        if idx in self.hand_collector.keys():
                            x, y = int(lm.x * self.cap_width), int(
                                lm.y * self.cap_height
                            )
                            # cv2.putText(img, 'left', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                            x, y = self._mapping(x, y)
                            left_hand_collector[idx] = [self.hand_collector[idx], x, y]

                    if len(self.hand_collector) == len(left_hand_collector):
                        hands_collector["left"] = left_hand_collector
                elif i == 0:
                    right_hand_collector = dict()
                    # hand_dict = dict()
                    for idx, lm in enumerate(handlms.landmark):
                        # hand_dict[md.solutions.hands.HandLandmark(idx).name] = idx
                        # print(hand_dict)
                        if idx in self.hand_collector.keys():
                            x, y = int(lm.x * self.cap_width), int(
                                lm.y * self.cap_height
                            )
                            # cv2.putText(img, 'right', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                            x, y = self._mapping(x, y)
                            # print(x,y)
                            right_hand_collector[idx] = [self.hand_collector[idx], x, y]
                    if len(self.hand_collector) == len(right_hand_collector):
                        hands_collector["right"] = right_hand_collector
                else:
                    print("too many hands")

            # 因后续以wrist作为移动点，将wrist超屏幕范围的剔除
            if "left" in hands_collector.keys():
                point = hands_collector["left"][0][1:]
                if not self._check_screen_xy(point[0], point[1]):
                    del hands_collector["left"]

            if "right" in hands_collector.keys():
                point = hands_collector["right"][0][1:]
                if not self._check_screen_xy(point[0], point[1]):
                    del hands_collector["right"]
            # command
            if len(hands_collector) >= 1:
                img = self.condition(
                    hands_collector, img, self.screen_width, self.screen_height
                )

        return img
    
    def _face(self, img):
        results = self.face_mesh.process(img)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 脸网格
                md.solutions.drawing_utils.draw_landmarks(image=img,
                                        landmark_list=face_landmarks,
                                        connections=md.solutions.face_mesh.FACEMESH_TESSELATION,
                                        landmark_drawing_spec=None,
                                        connection_drawing_spec=md.solutions.drawing_styles.get_default_face_mesh_tesselation_style())
                # 人脸
                md.solutions.drawing_utils.draw_landmarks(image=img,
                                        landmark_list=face_landmarks,
                                        connections=md.solutions.face_mesh.FACEMESH_CONTOURS,
                                        landmark_drawing_spec=None,
                                        connection_drawing_spec=md.solutions.drawing_styles.get_default_face_mesh_contours_style())
                # 瞳孔
                md.solutions.drawing_utils.draw_landmarks(image=img,
                                        landmark_list=face_landmarks,
                                        connections=md.solutions.face_mesh.FACEMESH_IRISES,
                                        landmark_drawing_spec=None,
                                        connection_drawing_spec=md.solutions.drawing_styles.get_default_face_mesh_iris_connections_style())

        return img

    def _mapping(self, x, y):
        return int((x - self.action_zone[0]) * self.scale[0]), int(
            (y - self.action_zone[1]) * self.scale[1]
        )

    def _check_is_in_area(self, point, area_center_point, area_radius):
        distance = (point[0] - area_center_point[0]) ** 2 + (
            point[1] - area_center_point[1]
        ) ** 2
        return distance <= area_radius**2

    def _check_screen_xy(self, x, y):
        if 0 <= x <= self.screen_width and 0 <= y <= self.screen_height:
            return True
        else:
            return False

    def run(self):
        while True:
            success, img = self.cap.read()
            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rgb = self._hand(img_rgb)

            if args.is_show_face:
                img_rgb=self._face(img_rgb)

            if args.is_show:
                img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                cv2.imshow("Camera", img)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            time.sleep(args.delay)

        self.cap.release()
        cv2.destroyAllWindows()


def main():
    handcontrol = HandControl()
    handcontrol.run()


if __name__ == "__main__":
    main()
