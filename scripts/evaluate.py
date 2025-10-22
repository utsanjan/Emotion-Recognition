#!/usr/bin/env python3
"""
Evaluate a trained model on the validation or test set.
Generates classification report and confusion matrix plots.
"""
import argparse
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def plot_confusion_matrix(cm, labels, out_path=None):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/mobilenet_emotion.h5')
    parser.add_argument('--data', type=str, default='data/cropped_faces/val')
    parser.add_argument('--input-size', type=int, default=224)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--out', type=str, default='outputs/figures/confusion_matrix.png')
    args = parser.parse_args()

    print("Loading model...")
    model = tf.keras.models.load_model(args.model)

    datagen = ImageDataGenerator(rescale=1. / 255)
    generator = datagen.flow_from_directory(
        args.data,
        target_size=(args.input_size, args.input_size),
        batch_size=args.batch,
        class_mode='categorical',
        shuffle=False
    )

    print("Evaluating model...")
    preds = model.predict(generator, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = generator.classes
    labels = list(generator.class_indices.keys())

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=labels))

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, labels, out_path=args.out)
    print(f"Confusion matrix saved to {args.out}")

    acc = np.sum(y_pred == y_true) / len(y_true)
    print(f"Overall Accuracy: {acc * 100:.2f}%")