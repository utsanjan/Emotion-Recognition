#!/usr/bin/env python3
"""
🎥 Real-time Facial Emotion Recognition using webcam.
- Detects faces via MTCNN
- Classifies emotions using a trained TensorFlow model
- Overlays color-coded emotion labels with confidence
- Optionally generates emotion-based MIDI output when emotion changes
"""

import argparse
import cv2
import numpy as np
import tensorflow as tf
from mtcnn.mtcnn import MTCNN
import os
import warnings
import absl.logging
from datetime import datetime

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

# Optional MIDI import
try:
    from scripts.emotion_to_midi import generate_melody
except Exception:
    generate_melody = None


# ---------------------------------------------------------------
# 🧠 Utility Functions
# ---------------------------------------------------------------
def get_labels_from_dir(train_dir):
    """Fetch emotion class names from directory structure."""
    labels = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    return labels


def detect_and_crop(frame, detector, input_size):
    """Detect faces and crop them to the required input size."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(rgb)
    faces, boxes = [], []

    for det in detections:
        x, y, w, h = det['box']
        x, y = max(0, x), max(0, y)
        face = frame[y:y + h, x:x + w]
        try:
            face_resized = cv2.resize(face, (input_size, input_size))
            faces.append(face_resized)
            boxes.append((x, y, w, h))
        except Exception:
            continue

    return faces, boxes


# ---------------------------------------------------------------
# 🎨 Emotion Colors
# ---------------------------------------------------------------
emotion_colors = {
    'Angry': (0, 0, 255),        # Red
    'Disgust': (0, 128, 0),      # Dark Green
    'Fear': (128, 0, 128),       # Purple
    'Happy': (0, 255, 0),        # Bright Green
    'Sad': (255, 0, 0),          # Blue
    'Surprise': (0, 255, 255),   # Yellow
    'Neutral': (200, 200, 200)   # Gray
}


# ---------------------------------------------------------------
# 🚀 Main
# ---------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Facial Emotion Recognition via Webcam")
    parser.add_argument('--model', type=str, default='models/mobilenet_emotion.keras',
                        help='Path to trained Keras model (.keras or .h5)')
    parser.add_argument('--data-dir', type=str, default='data/cropped_faces',
                        help='Dataset path to infer emotion labels')
    parser.add_argument('--input-size', type=int, default=224,
                        help='Model input size (height/width)')
    parser.add_argument('--camera-index', type=int, default=0,
                        help='Webcam index (default: 0)')
    parser.add_argument('--midi', action='store_true',
                        help='Generate emotion-based MIDI when emotion changes')
    args = parser.parse_args()

    print("📦 Loading model...")
    model = tf.keras.models.load_model(args.model)

    # Get emotion labels
    labels = get_labels_from_dir(os.path.join(args.data_dir, 'train'))
    print("✅ Loaded labels:", labels)

    # Initialize MTCNN detector and camera
    detector = MTCNN()
    cap = cv2.VideoCapture(args.camera_index)
    prev_emotion = None

    print("🎥 Starting webcam... Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame. Exiting.")
            break

        faces, boxes = detect_and_crop(frame, detector, args.input_size)
        for (face, (x, y, w, h)) in zip(faces, boxes):
            face_arr = face.astype('float32') / 255.0
            face_arr = np.expand_dims(face_arr, axis=0)
            preds = model.predict(face_arr, verbose=0)

            idx = int(np.argmax(preds))
            prob = float(np.max(preds))
            emotion = labels[idx] if idx < len(labels) else f"Class {idx}"

            # Determine emotion color
            color = emotion_colors.get(emotion, (255, 255, 255))

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Draw filled background for better visibility
            text = f"{emotion}: {prob:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x, y - th - 10), (x + tw + 4, y), color, -1)
            cv2.putText(frame, text, (x + 2, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            # 🎵 Generate MIDI on emotion change
            if args.midi and generate_melody and emotion != prev_emotion and prob > 0.6:
                midi_path = f"outputs/generated_music/{emotion}_{datetime.now().strftime('%H%M%S')}.mid"
                os.makedirs(os.path.dirname(midi_path), exist_ok=True)
                try:
                    generate_melody(emotion, out_path=midi_path)
                    print(f"🎵 Saved MIDI: {midi_path}")
                except Exception as e:
                    print(f"⚠️ MIDI generation failed: {e}")
                prev_emotion = emotion

        # Display live video
        cv2.imshow("Facial Emotion Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Webcam demo ended.")