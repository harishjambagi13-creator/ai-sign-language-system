# AI Sign Language Recognition System

A Python-based computer vision project that recognizes predefined hand gestures through a webcam and converts recognized signs into text and speech. The application also includes speech input functionality for interactive communication.

**Built with Python, OpenCV, MediaPipe, and Machine Learning.**

---

## Overview

Communication can be challenging when people use different communication methods. This project explores how computer vision and machine learning can help bridge that gap by recognizing predefined hand gestures in real time.

The system captures video from a webcam, detects hand landmarks using MediaPipe, and classifies the extracted hand features using a trained Random Forest model. The predicted sign is displayed on the screen and can be announced using text-to-speech.

The application also supports speech input through a keyboard shortcut.

> **Project status:** Prototype / Active Development. Recognition performance depends on the quality and diversity of the training data.

## Features

- **Real-Time Hand Detection:** Uses a webcam and MediaPipe Hand Landmarker.
- **Machine Learning Classification:** Uses a trained Random Forest classifier to recognize predefined signs.
- **Visual Predictions:** Displays the predicted sign on the webcam window.
- **Text-to-Speech:** Announces newly recognized signs using a speech synthesis engine.
- **Speech Input:** Press a keyboard shortcut to record speech and attempt speech recognition.
- **Custom Dataset:** Includes a workflow for collecting hand-landmark data.
- **Model Training:** Trains and evaluates a classifier using the collected dataset.
- **Local Execution:** Runs as a Python desktop application.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| OpenCV | Webcam capture and visual display |
| MediaPipe | Hand landmark detection |
| NumPy | Numerical operations and feature preparation |
| Pandas | Dataset loading and processing |
| Scikit-learn | Random Forest model training and evaluation |
| pyttsx3 | Text-to-speech output |
| SpeechRecognition | Speech recognition |
| SoundDevice | Audio recording |
| Pickle | Saving and loading the trained model |

## How It Works

The project follows a simple machine-learning pipeline:

1. **Data Collection:** Capture hand landmarks and associate them with sign labels.
2. **Feature Extraction:** Convert hand landmark coordinates into numerical features.
3. **Model Training:** Train a Random Forest classifier using the labeled dataset.
4. **Model Evaluation:** Evaluate the trained classifier using a held-out test split.
5. **Real-Time Detection:** Capture frames from the webcam and detect hand landmarks.
6. **Sign Classification:** Pass the extracted features to the trained model.
7. **Output:** Display the predicted sign and announce it using text-to-speech.
8. **Speech Input:** Use the speech-input shortcut to record and recognize spoken words.

## Project Structure

```text
ai-sign-language-system/
│
├── 1_collect_data.py       # Collect labeled hand-landmark data
├── 2_train_model.py        # Train and evaluate the classifier
├── 3_two_way_system.py     # Webcam recognition and audio features
├── create_sample_data.py   # Generate sample training data
│
├── data/
│   └── dataset.csv         # Labeled training dataset
│
├── models/
│   └── sign_model.pkl      # Trained machine-learning model
│
├── hand_landmarker.task    # MediaPipe hand landmark model
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── .gitignore              # Git ignore rules
```

## Requirements

- Python 3.10 or newer is a reasonable starting point, but compatibility depends on the installed MediaPipe and other package versions.
- A working webcam.
- A microphone for speech input.
- Speakers or headphones for speech output.
- An internet connection for installing packages and the online speech-recognition service.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/harishjambagi13-creator/ai-sign-language-system.git
```

### 2. Open the project directory

```bash
cd ai-sign-language-system
```

### 3. Create a virtual environment

**Windows:**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can install packages using `.venv\Scripts\python.exe -m pip` without activating the environment.

### 4. Install dependencies

If `requirements.txt` is present:

```powershell
python -m pip install -r requirements.txt
```

Otherwise, install the packages used by the application:

```powershell
python -m pip install opencv-python numpy pandas scikit-learn mediapipe pyttsx3 SpeechRecognition sounddevice
```

If installation fails, check the Python and package version compatibility before changing versions.

## Running the Application

Run the application from the project's root directory so the relative paths to the model and MediaPipe task file resolve correctly.

```powershell
python 3_two_way_system.py
```

### Controls

- **Show your hand:** Position your hand within the webcam frame.
- **Speech input:** Press `S` to start a short speech recording.
- **Quit:** Press `Q` to close the application.

Make sure the webcam and microphone are available and that the operating system has granted the required permissions.

## Collecting Data and Training the Model

### Step 1: Collect training data

```powershell
python 1_collect_data.py
```

Follow the prompts in the script to collect labeled examples of the supported hand signs.

### Step 2: Train the classifier

```powershell
python 2_train_model.py
```

The training script reads `data/dataset.csv`, separates the features from the labels, trains a Random Forest classifier, evaluates it on a test split, and saves the model to:

```text
models/sign_model.pkl
```

### Step 3: Run the application

```powershell
python 3_two_way_system.py
```

**Important:** The training data and the features generated during real-time recognition must have the same structure, ordering, and normalization. If predictions are incorrect, verify the label mapping, feature extraction, dataset balance, and training examples.

## Dataset and Model

The project includes a CSV dataset and a serialized trained model.

The current training script uses a Random Forest classifier. The real-time application derives features from the detected hand landmarks and passes those features to the classifier.

Recognition quality depends on factors such as:

- The number of examples for each sign.
- Variation in hand position, orientation, and distance.
- Lighting and webcam quality.
- Similarity between gestures.
- Correct feature normalization and label mapping.
- Consistency between training and inference preprocessing.

The reported test accuracy is not a guarantee of performance on unseen users or real-world conditions.

## Limitations

- The system recognizes only the signs represented in its training dataset.
- Recognition accuracy may vary between users and recording environments.
- Similar hand gestures can be confused with one another.
- Continuous sign-language sentences, grammar, facial expressions, and full sign-language interpretation are not currently guaranteed.
- Speech recognition relies on an online recognition service and may require internet access.
- This prototype has not been validated for medical, legal, emergency, or other critical communication.

## Future Improvements

Potential improvements include:

- Expand and balance the training dataset.
- Improve recognition accuracy with systematic testing.
- Add prediction confidence thresholds and temporal smoothing.
- Evaluate performance using independent test users.
- Support additional signs and more robust gesture recognition.
- Add automated tests and reproducible dependency versions.
- Improve the user interface and accessibility.
- Add screenshots, demonstration videos, and measured evaluation results.

## Contributing

Contributions, bug reports, and suggestions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Make your changes and test them.
4. Submit a pull request describing the changes.

## Disclaimer

This project is intended for educational, research, and demonstration purposes. It is an experimental sign-recognition prototype, not a certified sign-language translation system or a replacement for a qualified interpreter.

## Author

**Harish Jambagi**

GitHub: [@harishjambagi13-creator](https://github.com/harishjambagi13-creator)

## License

No license has been specified yet. Unless a license is added to this repository, users should not assume that they have permission to redistribute or modify the project's code.

---
