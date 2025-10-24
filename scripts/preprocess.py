#!/usr/bin/env python3
"""
Preprocess FER2013 or similar dataset:
- Convert CSV pixel data to cropped face images.
- Split into train/val sets.
- Detect & crop faces using MTCNN (optional).
- Show single-line tqdm progress bar.
"""

import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from mtcnn.mtcnn import MTCNN
from PIL import Image
import argparse

# 🔇 Disable TensorFlow / MTCNN verbose output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    tf.autograph.set_verbosity(0)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except Exception:
    pass


def pixels_to_image(pixels_str):
    """Convert pixel string to a 48x48 grayscale OpenCV image."""
    arr = np.fromstring(pixels_str, dtype=int, sep=' ')
    img = arr.reshape(48, 48).astype('uint8')
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def detect_and_crop(img, target_size=224, use_mtcnn=True):
    """Detect and crop face using MTCNN, fallback to simple resize."""
    if not use_mtcnn:
        return cv2.resize(img, (target_size, target_size))
    global detector
    if 'detector' not in globals():
        detector = MTCNN()
    results = detector.detect_faces(img)
    if not results:
        return cv2.resize(img, (target_size, target_size))
    x, y, w, h = results[0]['box']
    x, y = max(0, x), max(0, y)
    face = img[y:y + h, x:x + w]
    return cv2.resize(face, (target_size, target_size))


def main(args):
    # --- Load dataset ---
    if not args.csv:
        raise ValueError("Please provide --csv path to fer2013.csv")

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.csv)
    df = df[df['pixels'].notnull()]
    print(f"Loaded dataset: {df.shape[0]} samples")

    # --- Convert CSV pixels to temporary raw images ---
    temp_dir = os.path.join(args.out, 'temp_raw')
    os.makedirs(temp_dir, exist_ok=True)
    X_paths, y_labels = [], []

    print("Converting pixel strings to images...")
    for idx, row in tqdm(df.iterrows(), total=len(df), dynamic_ncols=True):
        img = pixels_to_image(row['pixels'])
        label = str(int(row['emotion']))
        out_path = os.path.join(temp_dir, f'{idx}_{label}.jpg')
        Image.fromarray(img).save(out_path)
        X_paths.append(out_path)
        y_labels.append(label)

    # --- Split into train/val ---
    X_train, X_val, y_train, y_val = train_test_split(
        X_paths, y_labels, test_size=0.15, stratify=y_labels, random_state=42
    )

    splits = [('train', X_train, y_train), ('val', X_val, y_val)]

    # --- Process images (crop + resize) with single tqdm bar ---
    print("Cropping and saving processed images...")
    all_images = [
        (img_path, label, split_name)
        for split_name, X_split, y_split in splits
        for img_path, label in zip(X_split, y_split)
    ]

    for img_path, label, split_name in tqdm(
        all_images,
        desc="Processing all images",
        dynamic_ncols=True,
        leave=True
    ):
        img = cv2.imread(img_path)
        if img is None:
            continue
        cropped = detect_and_crop(img, target_size=args.target_size, use_mtcnn=args.use_mtcnn)
        out_folder = os.path.join(args.out, split_name, label)
        os.makedirs(out_folder, exist_ok=True)
        out_file = os.path.join(out_folder, os.path.basename(img_path))
        cv2.imwrite(out_file, cropped)

    print("\n✅ Preprocessing completed successfully!")
    print(f"Processed data saved to: {args.out}")
    print("Folder structure:")
    for split in ['train', 'val']:
        split_path = os.path.join(args.out, split)
        if os.path.exists(split_path):
            classes = os.listdir(split_path)
            print(f"  {split}/ ({len(classes)} classes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess FER2013 or similar dataset")
    parser.add_argument("--csv", type=str, required=True, help="Path to fer2013.csv")
    parser.add_argument("--out", type=str, default="../data/cropped_faces", help="Output directory")
    parser.add_argument("--target-size", type=int, default=224, help="Target image size")
    parser.add_argument("--use-mtcnn", action="store_true", help="Use MTCNN for face detection")
    args = parser.parse_args()
    main(args)