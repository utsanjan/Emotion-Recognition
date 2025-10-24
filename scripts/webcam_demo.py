#!/usr/bin/env python3
"""
Real-time Facial Emotion Recognition using Webcam.
Detects faces via MTCNN, classifies emotions using a trained Keras model,
and overlays emotion labels (with color & confidence) on the live video feed.
Optionally generates emotion-based MIDI output when the detected emotion changes.
"""

import os
import cv2
import tqdm
import numpy as np
import tensorflow as tf
from mtcnn.mtcnn import MTCNN
from datetime import datetime
import warnings
import absl.logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(0)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings("ignore")
absl.logging.set_verbosity(absl.logging.ERROR)

try:
    tf.keras.utils.disable_interactive_logging()
except Exception:
    pass

# disable tqdm line-spam in notebooks
tqdm.tqdm = lambda *a, **k: a[0] if a else None

# Configuration
MODEL_PATH = "../models/mobilenet_emotion.keras"
INPUT_SIZE = 224
CAMERA_INDEX = 0
GENERATE_MIDI = False
MIDI_OUT_DIR = "../outputs/generated_music"
os.makedirs(MIDI_OUT_DIR, exist_ok=True)

# Emotion Labels & Colors
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emotion_colors = {
    'Angry': (0, 0, 255),        # Red
    'Disgust': (0, 128, 0),      # Dark Green
    'Fear': (128, 0, 128),       # Purple
    'Happy': (0, 255, 0),        # Bright Green
    'Sad': (255, 0, 0),          # Blue
    'Surprise': (0, 255, 255),   # Yellow
    'Neutral': (200, 200, 200)   # Gray
}


# Load Model and Initialize Detector

print("📦 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully.")

detector = MTCNN()
cap = cv2.VideoCapture(CAMERA_INDEX)
prev_label = None

print("🎥 Starting webcam... Press 'q' in the video window to quit.")


# Real-time Detection Loop
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read from camera. Exiting.")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = detector.detect_faces(rgb)

        for det in detections:
            x, y, w, h = det['box']
            x, y = max(0, x), max(0, y)
            face = frame[y:y + h, x:x + w]

            try:
                face_resized = cv2.resize(face, (INPUT_SIZE, INPUT_SIZE))
            except Exception:
                continue

            # Normalize and predict
            face_arr = face_resized.astype("float32") / 255.0
            face_arr = np.expand_dims(face_arr, axis=0)
            preds = model.predict(face_arr, verbose=0)

            idx = int(np.argmax(preds))
            prob = float(np.max(preds))
            label = labels[idx] if idx < len(labels) else f"Class {idx}"

            # Emotion color
            color = emotion_colors.get(label, (255, 255, 255))

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Text overlay (label + confidence)
            text = f"{label}: {prob:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x, y - th - 10), (x + tw + 6, y), color, -1)
            cv2.putText(frame, text, (x + 3, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            # Optional MIDI generation
            if GENERATE_MIDI and label != prev_label and prob > 0.6:
                try:
                    from scripts.emotion_to_midi import generate_melody
                    midi_path = os.path.join(
                        MIDI_OUT_DIR,
                        f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mid"
                    )
                    generate_melody(label, length=16, out_path=midi_path)
                    print(f"🎵 Saved MIDI for {label}: {midi_path}")
                except Exception as e:
                    print("⚠️ MIDI generation failed:", e)
                prev_label = label

        # Display live video
        cv2.imshow("Facial Emotion Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    pass

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Webcam demo ended.")