import cv2
import numpy as np
import time
import autopy
import pyautogui
from hand_detector import HandDetector

class VirtualMouseController:
    def __init__(self, camera_index=0, camera_width=640, camera_height=480, frame_reduction=100, smoothing=7):
        self.camera_index = camera_index
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.frame_reduction = frame_reduction
        self.smoothing = smoothing

        self.previous_location_x = 0
        self.previous_location_y = 0
        self.current_location_x = 0
        self.current_location_y = 0
        self.previous_time = 0

        self.webcam = cv2.VideoCapture(self.camera_index)
        self.webcam.set(3, self.camera_width)
        self.webcam.set(4, self.camera_height)

        self.hand_detector = HandDetector(max_hands=1)
        self.screen_width, self.screen_height = autopy.screen.size()

    def run(self):
        while True:
            success, image = self.webcam.read()
            if not success:
                continue

            image = self.hand_detector.find_hands(image)
            landmark_list, _ = self.hand_detector.find_positions(image)

            if landmark_list:
                index_finger_x, index_finger_y = landmark_list[8][1:]
                fingers = self.hand_detector.fingers_up()

                cv2.rectangle(
                    image,
                    (self.frame_reduction, self.frame_reduction),
                    (self.camera_width - self.frame_reduction, self.camera_height - self.frame_reduction),
                    (255, 0, 255),
                    2
                )
                # Index only (mouse move)
                if fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
                    mapped_x = np.interp(index_finger_x, (self.frame_reduction, self.camera_width - self.frame_reduction), (0, self.screen_width))
                    mapped_y = np.interp(index_finger_y, (self.frame_reduction, self.camera_height - self.frame_reduction), (0, self.screen_height))

                    self.current_location_x = self.previous_location_x + (mapped_x - self.previous_location_x) / self.smoothing
                    self.current_location_y = self.previous_location_y + (mapped_y - self.previous_location_y) / self.smoothing

                    autopy.mouse.move(self.screen_width - self.current_location_x, self.current_location_y)
                    cv2.circle(image, (index_finger_x, index_finger_y), 15, (255, 0, 255), cv2.FILLED)

                    self.previous_location_x = self.current_location_x
                    self.previous_location_y = self.current_location_y

                # Index + thumb (click)
                if fingers[1] == 1 and fingers.count(1) == 1:
                    autopy.mouse.click()
                    time.sleep(0.3)

                # Index + pinky (right click)
                if fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 1:
                    autopy.mouse.click(button=autopy.mouse.Button.RIGHT)
                    time.sleep(0.3)

                # All fingers open (scroll up)
                if fingers == [0, 1, 1, 1, 1]:
                    pyautogui.scroll(100)
                    time.sleep(0.2)

                # All fingers folded (scroll down)
                if fingers == [1, 0, 0, 0, 0]:
                    pyautogui.scroll(-100)
                    time.sleep(0.2)