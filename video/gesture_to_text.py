import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import json
import time
from threading import Timer

class HandGestureDetector:
    def __init__(self, model_path='mp_hand_gesture', names_path='mp_hand_gesture/gesture.names', threshold=40):
        # Initialize mediapipe
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mpDraw = mp.solutions.drawing_utils

        # Load the gesture recognizer model
        self.model = load_model(model_path)

        # Load class names
        with open(names_path, 'r') as f:
            self.classNames = f.read().split('\n')

        # Initialize the webcam
        self.cap = cv2.VideoCapture(0)

        # Variables for tracking hand movement
        self.prev_x = None
        self.threshold = threshold

    def detect_gesture_movement(self):
        gesture_info_list = []

        def stop():
            self.cap.release()
            cv2.destroyAllWindows()

        timer = Timer(3, stop)
        timer.start()

        while not timer.finished.wait(0.1):
            ret, frame = self.cap.read()

            if not ret:
                continue  # Skip frames where capture was unsuccessful

            x, y, c = frame.shape

            frame = cv2.flip(frame, 1)
            framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.hands.process(framergb)

            if result.multi_hand_landmarks:
                gesture_info = {
                    'gesture': '',
                    'movement': '',
                    'time': time.time()
                }
                landmarks = []
                for handslms in result.multi_hand_landmarks:
                    for lm in handslms.landmark:
                        lmx = int(lm.x * x)
                        lmy = int(lm.y * y)
                        landmarks.append([lmx, lmy])

                # Check for hand movement
                if self.prev_x is not None:
                    delta_x = landmarks[0][0] - self.prev_x
                    if delta_x > self.threshold:
                        gesture_info['movement'] = "Moving from left to right!"
                    elif delta_x < -self.threshold:
                        gesture_info['movement'] = "Moving from right to left!"

                self.prev_x = landmarks[0][0]

                # Predict gesture
                prediction = self.model.predict([landmarks])
                classID = np.argmax(prediction)
                gesture_info['gesture'] = self.classNames[classID]

                gesture_info_list.append(gesture_info)

        return gesture_info_list

