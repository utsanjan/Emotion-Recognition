#!/usr/bin/env python3
"""
Utility for detecting and cropping faces from images using MTCNN.
Used by preprocess.py and webcam_demo.py.
"""
import cv2
from mtcnn.mtcnn import MTCNN


def detect_and_crop(image, target_size=224):
    """
    Detects the largest face in the image and returns a cropped + resized face.
    Returns None if no face is detected.
    """
    detector = MTCNN()
    results = detector.detect_faces(image)
    if not results:
        return None
    # Choose largest bounding box (in case of multiple faces)
    largest = max(results, key=lambda x: x['box'][2] * x['box'][3])
    x, y, w, h = largest['box']
    x, y = max(0, x), max(0, y)
    face = image[y:y + h, x:x + w]
    face = cv2.resize(face, (target_size, target_size))
    return face


def align_and_crop(image, target_size=224):
    """
    For future use: aligns the face using MTCNN keypoints (eyes, nose, mouth)
    before cropping. Currently returns same as detect_and_crop().
    """
    return detect_and_crop(image, target_size=target_size)


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, required=True, help='Path to input image')
    parser.add_argument('--out', type=str, default='cropped_face.jpg')
    parser.add_argument('--size', type=int, default=224)
    args = parser.parse_args()

    img = cv2.imread(args.img)
    if img is None:
        raise FileNotFoundError(f"Image not found: {args.img}")

    cropped = detect_and_crop(img, target_size=args.size)
    if cropped is not None:
        cv2.imwrite(args.out, cropped)
        print(f"Saved cropped face to {args.out}")
    else:
        print("No face detected in the image.")