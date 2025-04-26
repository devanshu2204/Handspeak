import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

model = load_model("hand_sign_model.h5")

# Word-based labels (update to include digits/operators if needed in future)
LABELS = {
    0: "HELLO", 1: "YES", 2: "NO", 3: "PLEASE", 4: "THANK YOU",
    5: "BYE", 6: "I", 7: "YOU", 8: "LOVE", 9: "DELETE",
    10: "0", 11: "1", 12: "2", 13: "3", 14: "4", 15: "5",
    16: "6", 17: "7", 18: "8", 19: "9",
    20: "+", 21: "-", 22: "*", 23: "/", 24: "=", 25: ".",
    26: "(", 27: ")"
}


# Allowed calculator symbols
CALCULATOR_SYMBOLS = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                      "+", "-", "*", "/", "=", ".", "(", ")"}

def extract_features(hand_landmarks):
    features = []
    for landmark in hand_landmarks.landmark:
        features.extend([landmark.x, landmark.y, landmark.z])
    return np.array(features).reshape(1, -1)

def recognize_hand_sign(frame, calculator_mode=False):
    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1) as hands:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                features = extract_features(hand_landmarks)
                prediction = model.predict(features)
                predicted_label = np.argmax(prediction)
                detected_sign = LABELS.get(predicted_label, "")

                # Ignore non-calculator gestures in calculator mode
                if calculator_mode and detected_sign not in CALCULATOR_SYMBOLS and detected_sign != "[DEL]":
                    return None, frame

                return detected_sign, frame

    return None, frame
