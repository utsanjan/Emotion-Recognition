#!/usr/bin/env python3
"""
Train a transfer-learning model (MobileNetV2 or ResNet50) on FER2013 images.
"""
import argparse
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def build_model(arch='mobilenet', input_shape=(224, 224, 3), num_classes=7, dropout=0.5):
    if arch == 'mobilenet':
        base = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    elif arch == 'resnet50':
        base = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError("Unsupported architecture. Choose 'mobilenet' or 'resnet50'.")

    base.trainable = False
    inputs = Input(shape=input_shape)
    x = base(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model, base


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/cropped_faces', help='Path to dataset folder with train/val subfolders')
    parser.add_argument('--arch', type=str, default='mobilenet', choices=['mobilenet', 'resnet50'])
    parser.add_argument('--input-size', type=int, default=224)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--unfreeze-layers', type=int, default=30, help='Number of layers to unfreeze for fine-tuning')
    parser.add_argument('--out-model', type=str, default='models/mobilenet_emotion.h5')
    args = parser.parse_args()

    train_dir = os.path.join(args.data, 'train')
    val_dir = os.path.join(args.data, 'val')
    img_size = (args.input_size, args.input_size)

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1
    )
    val_datagen = ImageDataGenerator(rescale=1. / 255)

    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=img_size, batch_size=args.batch, class_mode='categorical'
    )
    val_gen = val_datagen.flow_from_directory(
        val_dir, target_size=img_size, batch_size=args.batch, class_mode='categorical'
    )

    num_classes = train_gen.num_classes
    model, base = build_model(arch=args.arch, input_shape=(args.input_size, args.input_size, 3), num_classes=num_classes)
    model.compile(optimizer=Adam(args.lr), loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(args.out_model, save_best_only=True, monitor='val_loss'),
        tf.keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True, monitor='val_loss'),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, monitor='val_loss')
    ]

    print("Starting initial training (feature extraction)...")
    history = model.fit(train_gen, validation_data=val_gen, epochs=args.epochs, callbacks=callbacks)

    print("Fine-tuning last layers...")
    base.trainable = True
    if args.unfreeze_layers > 0:
        for layer in base.layers[:-args.unfreeze_layers]:
            layer.trainable = False

    model.compile(optimizer=Adam(args.lr / 10), loss='categorical_crossentropy', metrics=['accuracy'])
    ft_history = model.fit(train_gen, validation_data=val_gen, epochs=10, callbacks=callbacks)

    model.save(args.out_model)
    print(f"Transfer learning model saved to {args.out_model}")