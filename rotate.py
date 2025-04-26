import os
import numpy as np

# Paths
INPUT_DIR = "data/"
OUTPUT_DIR = "dataset_augmented/"

# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Function to rotate hand landmarks
def rotate_landmarks(landmarks, angle=15):
    """Rotates hand landmarks by a given angle (in degrees)."""
    radians = np.radians(angle)
    cos_val, sin_val = np.cos(radians), np.sin(radians)
    
    rotated_landmarks = landmarks.copy()
    for i in range(0, len(landmarks), 3):  # Assuming (x, y, z) format
        x, y = landmarks[i], landmarks[i+1]
        rotated_landmarks[i] = cos_val * x - sin_val * y
        rotated_landmarks[i+1] = sin_val * x + cos_val * y
    
    return rotated_landmarks

# Process each class
for class_name in os.listdir(INPUT_DIR):
    class_path = os.path.join(INPUT_DIR, class_name)
    output_class_path = os.path.join(OUTPUT_DIR, class_name)

    if not os.path.exists(output_class_path):
        os.makedirs(output_class_path)

    for file_name in os.listdir(class_path):
        file_path = os.path.join(class_path, file_name)

        # Load .npy file
        data = np.load(file_path)

        # Flip horizontally
        flipped_data = data.copy()
        for i in range(0, len(data), 3):  # Assuming (x, y, z) format
            flipped_data[i] = -flipped_data[i]  # Flip x-coordinates

        # Rotate
        rotated_data = rotate_landmarks(data)

        # Save augmented data
        np.save(os.path.join(output_class_path, f"original_{file_name}"), data)
        np.save(os.path.join(output_class_path, f"flipped_{file_name}"), flipped_data)
        np.save(os.path.join(output_class_path, f"rotated_{file_name}"), rotated_data)

print("✅ Data augmentation complete!")
