import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
import pickle
import pyttsx3
import speech_recognition as sr
import sounddevice as sd
import threading
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Initialize Text-To-Speech engine
engine = pyttsx3.init()
def speak(text):
    def run():
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass
    threading.Thread(target=run, daemon=True).start()

# Load trained sign language model
model_path = 'models/sign_model.pkl'
model = None
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
else:
    print("Warning: Model file not found! Please run 2_train_model.py")

# Initialize MediaPipe Hand Landmarker Tasks API
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=vision.RunningMode.IMAGE
)
detector = vision.HandLandmarker.create_from_options(options)

# Speech Recognition Function
recognized_speech = "Press 's' to Speak"
def listen_speech():
    global recognized_speech
    recognizer = sr.Recognizer()
    recognized_speech = "Listening (4 sec)..."
    
    try:
        sample_rate = 16000
        duration = 4
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()
        
        audio_data = sr.AudioData(recording.tobytes(), sample_rate, 2)
        text = recognizer.recognize_google(audio_data)
        recognized_speech = f"You Said: {text}"
        print(f"Speech Recognized: {text}")
    except Exception:
        recognized_speech = "Could not understand audio"

cap = cv2.VideoCapture(0)
last_prediction = ""
window_name = "AI Two-Way Sign Language System"
cv2.namedWindow(window_name)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert OpenCV image matrix to MediaPipe Image format
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    current_prediction = "None"

    # Process hand landmarks if detected
    if detection_result.hand_landmarks and model is not None:
        for hand_landmarks in detection_result.hand_landmarks:
            x_coords = [lm.x for lm in hand_landmarks]
            y_coords = [lm.y for lm in hand_landmarks]

            # Normalize coordinates relative to wrist origin
            base_x, base_y = x_coords[0], y_coords[0]
            norm_x = [x - base_x for x in x_coords]
            norm_y = [y - base_y for y in y_coords]

            # Construct 42-element feature vector
            features = np.array(norm_x + norm_y).reshape(1, -1)

            try:
                current_prediction = model.predict(features)[0]
            except Exception:
                current_prediction = "None"

            # Draw key points on image frame
            h, w, _ = frame.shape
            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

            # Trigger text-to-speech for newly recognized gesture
            if current_prediction != last_prediction and current_prediction != "None":
                speak(f"Sign detected: {current_prediction}")
                last_prediction = current_prediction

    # Display Overlay Info
    cv2.rectangle(frame, (0, 0), (640, 80), (0, 0, 0), -1)
    cv2.putText(frame, f"Sign Detected: {current_prediction}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Speech Input: {recognized_speech}", (10, 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow(window_name, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break
    elif key == ord('s'):
        threading.Thread(target=listen_speech, daemon=True).start()

cap.release()
cv2.destroyAllWindows()