import cv2
import mediapipe as mp
import numpy as np
import os

# Define word and calculator gestures to collect
GESTURES = ["HELLO", "YES", "NO", "PLEASE", "THANK YOU", "BYE", "I", "YOU", "LOVE", "DELETE", 
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", 
            "+", "-", "*", "/", "=", ".", "(", ")"]
DATA_DIR = "./data"

# Map unsafe gestures to safe folder names
SAFE_GESTURE_MAP = {
    '*': 'multiply',
    '/': 'divide',
    '+': 'plus',
    '-': 'minus',
    '=': 'equals',
    '.': 'dot',
    '(': 'open_paren',
    ')': 'close_paren'
}

# Function to sanitize gesture names for safe folder usage
def sanitize_gesture_name(gesture_name):
    return SAFE_GESTURE_MAP.get(gesture_name, gesture_name)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Create dataset directories
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

for gesture in GESTURES:
    sanitized_gesture = sanitize_gesture_name(gesture)
    os.makedirs(os.path.join(DATA_DIR, sanitized_gesture), exist_ok=True)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("📌 Press 'S' to save | 'N' to switch gesture | 'Q' to quit.")

current_gesture = 0  # Start with the first gesture
sample_count = len(os.listdir(os.path.join(DATA_DIR, sanitize_gesture_name(GESTURES[current_gesture]))))  # Track samples

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

            gesture_display = GESTURES[current_gesture]
            folder_display = sanitize_gesture_name(gesture_display)
            cv2.putText(frame, f"Gesture: {gesture_display} ({folder_display})", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Samples: {sample_count}", (30, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    sidebar = np.zeros((frame.shape[0], 200, 3), dtype=np.uint8)
    cv2.putText(sidebar, "Controls:", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(sidebar, "'S' - Save", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(sidebar, "'N' - Next", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(sidebar, "'Q' - Quit", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    frame = np.hstack((frame, sidebar))
    cv2.imshow("Hand Gesture Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and result.multi_hand_landmarks:
        sanitized_gesture = sanitize_gesture_name(GESTURES[current_gesture])
        sample_count += 1
        filename = os.path.join(DATA_DIR, sanitized_gesture, f"sample_{sample_count}.npy")
        np.save(filename, landmarks)
        print(f"✅ Saved: {filename}")
    elif key == ord('n'):
        current_gesture = (current_gesture + 1) % len(GESTURES)
        sanitized_gesture = sanitize_gesture_name(GESTURES[current_gesture])
        sample_count = len(os.listdir(os.path.join(DATA_DIR, sanitized_gesture)))
        print(f"🔄 Switched to: {GESTURES[current_gesture]} (Samples: {sample_count})")
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("🎉 Word and calculator gesture data collection complete!")
