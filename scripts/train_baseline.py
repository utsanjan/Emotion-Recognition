#!/usr/bin/env python3
"""
Train a simple CNN baseline model on FER2013 (48x48 grayscale).
This gives a fast benchmark before using transfer learning.
"""
import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


def build_baseline_cnn(input_shape=(48, 48, 1), num_classes=7):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/cropped_faces', help='Path to preprocessed 48x48 grayscale dataset')
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--out-model', type=str, default='models/baseline_cnn.h5')
    args = parser.parse_args()

    train_dir = os.path.join(args.data, 'train')
    val_dir = os.path.join(args.data, 'val')

    img_size = (48, 48)
    datagen_train = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True)
    datagen_val = ImageDataGenerator(rescale=1. / 255)

    train_gen = datagen_train.flow_from_directory(
        train_dir, target_size=img_size, color_mode='grayscale',
        batch_size=args.batch, class_mode='categorical'
    )
    val_gen = datagen_val.flow_from_directory(
        val_dir, target_size=img_size, color_mode='grayscale',
        batch_size=args.batch, class_mode='categorical'
    )

    num_classes = train_gen.num_classes
    model = build_baseline_cnn(input_shape=(48, 48, 1), num_classes=num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        ModelCheckpoint(args.out_model, save_best_only=True, monitor='val_loss'),
        EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
        ReduceLROnPlateau(factor=0.5, patience=2, monitor='val_loss')
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks
    )

    model.save(args.out_model)
    print(f"Baseline CNN model saved to {args.out_model}")