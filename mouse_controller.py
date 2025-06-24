import cv2
import numpy as np
import time
import autopy
import pyautogui
from hand_detector import HandDetector
from gestures import (
    MoveCursorGesture,
    ClickGesture,
    RightClickGesture,
    ScrollUpGesture,
    ScrollDownGesture
)

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
                    gesture = MoveCursorGesture(
                        landmark_list, fingers,
                        frame_reduction=self.frame_reduction,
                        screen_width=self.screen_width,
                        screen_height=self.screen_height,
                        prev_x=self.previous_location_x,
                        prev_y=self.previous_location_y,
                        smoothing=self.smoothing
                    )
                    self.previous_location_x, self.previous_location_y = gesture.execute(image)

                # Index + thumb (click)
                elif fingers[1] == 1 and fingers.count(1) == 1:
                    ClickGesture(landmark_list, fingers).execute(image)

                # Index + pinky (right click)
                elif fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 1:
                    RightClickGesture(landmark_list, fingers).execute(image)

                # All fingers open (scroll up)
                elif fingers == [0, 1, 1, 1, 1]:
                    ScrollUpGesture(landmark_list, fingers).execute(image)

                # All fingers folded (scroll down)
                elif fingers == [1, 0, 0, 0, 0]:
                    ScrollDownGesture(landmark_list, fingers).execute(image)

            current_time = time.time()
            frames_per_second = 1 / (current_time - self.previous_time) if self.previous_time else 0
            self.previous_time = current_time

            cv2.putText(image, f"FPS: {int(frames_per_second)}", (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
            cv2.imshow("Virtual Mouse", image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.webcam.release()
        cv2.destroyAllWindows()