import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Updated gesture list — word-based
GESTURES = [
    "HELLO", "YES", "NO", "PLEASE", "THANK YOU", "BYE", "I", "YOU", "LOVE", "DELETE",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "+", "-", "*", "/", "=", ".", "(", ")"
]

DATA_DIR = "./dataset_augmented"

X, y = [], []

# Load dataset
for label, gesture in enumerate(GESTURES):
    gesture_path = os.path.join(DATA_DIR, gesture)

    if not os.path.exists(gesture_path):
        print(f"Warning: Folder '{gesture}' not found in {DATA_DIR}. Skipping.")
        continue

    for file in os.listdir(gesture_path):
        file_path = os.path.join(gesture_path, file)

        try:
            landmarks = np.load(file_path)
            X.append(landmarks)
            y.append(label)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

if len(X) == 0:
    print("Error: No valid training data found!")
    exit()

X = X / np.max(X)  # Normalize

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation="relu", input_shape=(X.shape[1],)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(len(GESTURES), activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train
try:
    model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))
    model.save("hand_sign_model.h5")
    print("✅ Model trained and saved as 'hand_sign_model.h5'")
except Exception as e:
    print(f"Error during training: {e}")
