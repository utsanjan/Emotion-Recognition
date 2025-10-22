#!/usr/bin/env python3
"""
Real-time facial emotion recognition using webcam.
Detects faces via MTCNN, classifies emotions using trained model,
and overlays labels and probabilities on video frames.
Optionally generates emotion-based MIDI output when emotion changes.
"""
import argparse
import cv2
import numpy as np
import tensorflow as tf
from mtcnn.mtcnn import MTCNN
import os
from datetime import datetime

# Optional import for MIDI generation
try:
    from scripts.emotion_to_midi import generate_melody
except Exception:
    generate_melody = None


def get_labels_from_dir(train_dir):
    labels = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    return labels


def detect_and_crop(frame, detector, input_size):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(rgb)
    faces = []
    boxes = []
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/mobilenet_emotion.h5', help='Path to trained model')
    parser.add_argument('--data-dir', type=str, default='data/cropped_faces', help='Dataset path to infer labels')
    parser.add_argument('--input-size', type=int, default=224)
    parser.add_argument('--camera-index', type=int, default=0)
    parser.add_argument('--midi', action='store_true', help='Generate emotion-based MIDI when emotion changes')
    args = parser.parse_args()

    print("Loading model...")
    model = tf.keras.models.load_model(args.model)
    labels = get_labels_from_dir(os.path.join(args.data_dir, 'train'))
    print("Loaded labels:", labels)

    detector = MTCNN()
    cap = cv2.VideoCapture(args.camera_index)
    prev_emotion = None

    print("Starting webcam... Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces, boxes = detect_and_crop(frame, detector, args.input_size)
        for (face, (x, y, w, h)) in zip(faces, boxes):
            face_arr = face.astype('float32') / 255.0
            face_arr = np.expand_dims(face_arr, axis=0)
            preds = model.predict(face_arr, verbose=0)
            idx = int(np.argmax(preds))
            prob = float(np.max(preds))
            emotion = labels[idx]

            color = (0, 255, 0) if prob > 0.6 else (0, 255, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{emotion}: {prob:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Trigger MIDI generation if emotion changes
            if args.midi and generate_melody and emotion != prev_emotion and prob > 0.6:
                midi_path = f"outputs/generated_music/{emotion}_{datetime.now().strftime('%H%M%S')}.mid"
                os.makedirs(os.path.dirname(midi_path), exist_ok=True)
                generate_melody(emotion, out_path=midi_path)
                prev_emotion = emotion

        cv2.imshow("Facial Emotion Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()