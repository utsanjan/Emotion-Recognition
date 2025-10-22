#!/usr/bin/env python3
"""
Utility for loading FER2013 or custom datasets.
Provides functions to create Keras ImageDataGenerators or load pre-saved numpy arrays.
"""
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_generators(data_dir, img_size=(224, 224), batch_size=32):
    """
    Create train and validation ImageDataGenerators from a directory structure:
    data_dir/train/<class>/...
    data_dir/val/<class>/...
    """
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_gen, val_gen


def load_numpy_dataset(data_path):
    """
    Load numpy arrays previously saved by preprocess.py.
    Expects X_train.npy, y_train.npy, X_val.npy, y_val.npy.
    """
    X_train = np.load(os.path.join(data_path, 'X_train.npy'))
    y_train = np.load(os.path.join(data_path, 'y_train.npy'))
    X_val = np.load(os.path.join(data_path, 'X_val.npy'))
    y_val = np.load(os.path.join(data_path, 'y_val.npy'))
    return X_train, y_train, X_val, y_val


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/cropped_faces', help='Dataset directory or numpy path')
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--use-numpy', action='store_true', help='Load from .npy files instead of directory')
    args = parser.parse_args()

    if args.use_numpy:
        X_train, y_train, X_val, y_val = load_numpy_dataset(args.data)
        print(f"Loaded numpy dataset: X_train={X_train.shape}, X_val={X_val.shape}")
    else:
        train_gen, val_gen = create_generators(args.data, img_size=(args.img_size, args.img_size), batch_size=args.batch)
        print(f"Train classes: {train_gen.num_classes}, Validation classes: {val_gen.num_classes}")