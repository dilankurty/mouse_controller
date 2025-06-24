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

    def find_positions(self, image, hand_index=0, draw=True):
        x_coordinates = []
        y_coordinates = []
        bounding_box = []
        self.landmark_list = []

        if self.results.multi_hand_landmarks:
            selected_hand = self.results.multi_hand_landmarks[hand_index]
            image_height, image_width, _ = image.shape

            for landmark_index, landmark in enumerate(selected_hand.landmark):
                pixel_x = int(landmark.x * image_width)
                pixel_y = int(landmark.y * image_height)
                x_coordinates.append(pixel_x)
                y_coordinates.append(pixel_y)
                self.landmark_list.append([landmark_index, pixel_x, pixel_y])

                if draw:
                    cv2.circle(image, (pixel_x, pixel_y), 5, (255, 0, 255), cv2.FILLED)

            x_min, x_max = min(x_coordinates), max(x_coordinates)
            y_min, y_max = min(y_coordinates), max(y_coordinates)
            bounding_box = x_min, y_min, x_max, y_max

            if draw:
                cv2.rectangle(
                    image,
                    (x_min - 20, y_min - 20),
                    (x_max + 20, y_max + 20),
                    (0, 255, 0),
                    2
                )

        return self.landmark_list, bounding_box

    def fingers_up(self):
        fingers = []

        if not self.landmark_list:
            return fingers

        if self.landmark_list[self.finger_tip_ids[0]][1] > self.landmark_list[self.finger_tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for tip_index in range(1, 5):
            if self.landmark_list[self.finger_tip_ids[tip_index]][2] < self.landmark_list[self.finger_tip_ids[tip_index] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def find_distance(self, index_1, index_2, image, draw=True, radius=15, thickness=3):
        x_coord_1, y_coord_1 = self.landmark_list[index_1][1:]
        x_coord_2, y_coord_2 = self.landmark_list[index_2][1:]
        center_x, center_y = (x_coord_1 + x_coord_2) // 2, (y_coord_1 + y_coord_2) // 2

        if draw:
            cv2.line(image, (x_coord_1, y_coord_1), (x_coord_2, y_coord_2), (255, 0, 255), thickness)
            cv2.circle(image, (x_coord_1, y_coord_1), radius, (255, 0, 255), cv2.FILLED)
            cv2.circle(image, (x_coord_1, y_coord_2), radius, (255, 0, 255), cv2.FILLED)
            cv2.circle(image, (center_x, center_y), radius, (0, 0, 255), cv2.FILLED)

        distance = math.hypot(x_coord_2 - x_coord_1, y_coord_2 - y_coord_1)
        
        return distance, image, [x_coord_1, y_coord_1, x_coord_2, y_coord_2, center_x, center_y]
