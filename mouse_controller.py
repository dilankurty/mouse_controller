import cv2
import numpy as np
import time
import autopy
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