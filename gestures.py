import cv2
import numpy as np
import autopy
import pyautogui
import time

class BaseGesture:
    def __init__(self, landmark_list, fingers_up):
        self.landmark_list = landmark_list
        self.fingers_up = fingers_up

    def execute(self, image):
        raise NotImplementedError("Subclasses must implement the execute method.")

class MoveCursorGesture(BaseGesture):
    def __init__(self, landmark_list, fingers_up, frame_reduction, screen_width, screen_height, prev_x, prev_y, smoothing):
        super().__init__(landmark_list, fingers_up)
        self.frame_reduction = frame_reduction
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.prev_x = prev_x
        self.prev_y = prev_y
        self.smoothing = smoothing

    def execute(self, image):
        index_x, index_y = self.landmark_list[8][1:]
        mapped_x = np.interp(index_x, (self.frame_reduction, 640 - self.frame_reduction), (0, self.screen_width))
        mapped_y = np.interp(index_y, (self.frame_reduction, 480 - self.frame_reduction), (0, self.screen_height))
        current_x = self.prev_x + (mapped_x - self.prev_x) / self.smoothing
        current_y = self.prev_y + (mapped_y - self.prev_y) / self.smoothing
        autopy.mouse.move(self.screen_width - current_x, current_y)
        cv2.circle(image, (index_x, index_y), 15, (255, 0, 255), cv2.FILLED)
        return current_x, current_y
    
class ClickGesture(BaseGesture):
    def execute(self, image):
        autopy.mouse.click()
        time.sleep(0.3)


class RightClickGesture(BaseGesture):
    def execute(self, image):
        autopy.mouse.click(button=autopy.mouse.Button.RIGHT)
        time.sleep(0.5)


class ScrollUpGesture(BaseGesture):
    def execute(self, image):
        pyautogui.scroll(100)
        time.sleep(0.2)


class ScrollDownGesture(BaseGesture):
    def execute(self, image):
        pyautogui.scroll(-100)
        time.sleep(0.2)
