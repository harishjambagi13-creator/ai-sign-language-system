import cv2
import numpy as np
import pandas as pd
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Download model asset if missing
model_asset = 'hand_landmarker.task'
if not os.path.exists(model_asset):
    import urllib.request
    urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", model_asset)

base_options = python.BaseOptions(model_asset_path=model_asset)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1, running_mode=vision.RunningMode.IMAGE)
detector = vision.HandLandmarker.create_from_options(options)

# Define sign classes to collect
SIGNS = ['A', 'B', 'C', 'HELLO', 'THANKS','YES','NO','PLEASE','SORRY','LOVE']
SAMPLES_PER_SIGN = 100

os.makedirs('data', exist_ok=True)
csv_file = 'data/dataset.csv'

dataset = []
cap = cv2.VideoCapture(0)

print("--- Hand Landmark Collector ---")
print("Press 'c' to start capturing data for each label.")

for label in SIGNS:
    print(f"\nGet ready to record sign: '{label}'")
    count = 0
    capturing = False

    while cap.isOpened() and count < SAMPLES_PER_SIGN:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = detector.detect(mp_image)

        if results.hand_landmarks:
            hand_landmarks = results.hand_landmarks[0]
            
            # Draw landmarks on screen
            h, w, _ = frame.shape
            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

            if capturing:
                x_coords = [lm.x for lm in hand_landmarks]
                y_coords = [lm.y for lm in hand_landmarks]
                base_x, base_y = x_coords[0], y_coords[0]
                norm_x = [x - base_x for x in x_coords]
                norm_y = [y - base_y for y in y_coords]

                row = norm_x + norm_y + [label]
                dataset.append(row)
                count += 1

        status_text = f"Recording '{label}': {count}/{SAMPLES_PER_SIGN}" if capturing else f"Press 'c' to capture '{label}'"
        cv2.rectangle(frame, (0, 0), (640, 50), (0, 0, 0), -1)
        cv2.putText(frame, status_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Data Collector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            capturing = True
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

cap.release()
cv2.destroyAllWindows()

# Save collected data to dataset.csv
cols = [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)] + ['label']
df = pd.DataFrame(dataset, columns=cols)
df.to_csv(csv_file, index=False)
print(f"\nDataset saved successfully to {csv_file}!")