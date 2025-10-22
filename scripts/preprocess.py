#!/usr/bin/env python3
"""
Preprocess script:
- Load FER2013 CSV (or image folder)
- Crop faces using MTCNN (optional)
- Save images into folder structure: out/train/<label>/..., out/val/<label>/...
"""
import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from mtcnn.mtcnn import MTCNN


def pixels_to_image(pixels_str):
    arr = np.fromstring(pixels_str, dtype=int, sep=' ')
    img = arr.reshape(48, 48).astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def save_image(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(img).save(path)


def detect_and_crop(img, target_size=224):
    detector = MTCNN()
    results = detector.detect_faces(img)
    if results:
        x, y, w, h = results[0]['box']
        x, y = max(0, x), max(0, y)
        face = img[y:y + h, x:x + w]
        face = cv2.resize(face, (target_size, target_size))
        return face
    else:
        return cv2.resize(img, (target_size, target_size))


def process_fer_csv(csv_path, out_dir, target_size=224, use_mtcnn=False, test_size=0.15, random_state=42):
    df = pd.read_csv(csv_path)
    df = df[df['pixels'].notnull()]
    X_paths, y_labels = [], []

    temp_dir = os.path.join(out_dir, 'temp_raw')
    os.makedirs(temp_dir, exist_ok=True)

    print('Converting CSV pixels to images...')
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img = pixels_to_image(row['pixels'])
        label = str(int(row['emotion']))
        out_path = os.path.join(temp_dir, f'{idx}_{label}.jpg')
        save_image(img, out_path)
        X_paths.append(out_path)
        y_labels.append(label)

    print('Splitting into train/val...')
    X_train, X_val, y_train, y_val = train_test_split(
        X_paths, y_labels, test_size=test_size, stratify=y_labels, random_state=random_state
    )

    print('Cropping (optional) and resizing...')
    for split_name, X_split, y_split in [('train', X_train, y_train), ('val', X_val, y_val)]:
        for i, (p, label) in enumerate(tqdm(list(zip(X_split, y_split)))):
            img = cv2.imread(p)
            if use_mtcnn:
                cropped_img = detect_and_crop(img, target_size)
            else:
                cropped_img = cv2.resize(img, (target_size, target_size))

            out_folder = os.path.join(out_dir, split_name, label)
            os.makedirs(out_folder, exist_ok=True)
            out_file = os.path.join(out_folder, os.path.basename(p))
            cv2.imwrite(out_file, cropped_img)

    print('Done. Preprocessed data saved to:', out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, help='../data/fer2013.csv')
    parser.add_argument('--out', type=str, default='data/cropped_faces', help='Output folder')
    parser.add_argument('--target-size', type=int, default=224)
    parser.add_argument('--use-mtcnn', action='store_true')
    args = parser.parse_args()

    if not args.csv:
        raise ValueError('../data/fer2013.csv')

    process_fer_csv(args.csv, args.out, target_size=args.target_size, use_mtcnn=args.use_mtcnn)