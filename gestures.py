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
