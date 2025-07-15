import cv2
import mediapipe as mp
import numpy as np
from utils import classify_static_gesture

class GestureRecognizer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                         max_num_hands=1,
                                         min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils

    def detect_and_classify(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb_frame)
        gesture = "None"

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                landmark_list = []
                for lm in hand_landmarks.landmark:
                    landmark_list.append([lm.x, lm.y, lm.z])

                landmark_array = np.array(landmark_list)
                gesture = classify_static_gesture(landmark_array)

        return frame, gesture
