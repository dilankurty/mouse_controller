import cv2
import mediapipe as mediapipe_module
import math

class HandDetector:
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mediapipe_hands = mediapipe_module.solutions.hands
        self.hands = self.mediapipe_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )
        self.mediapipe_draw = mediapipe_module.solutions.drawing_utils
        self.finger_tip_ids = [4, 8, 12, 16, 20]
        self.landmark_list = []
        self.results = None

    def find_hands(self, image, draw=True):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(image_rgb)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mediapipe_draw.draw_landmarks(
                        image, hand_landmarks, self.mediapipe_hands.HAND_CONNECTIONS
                    )
        return image
