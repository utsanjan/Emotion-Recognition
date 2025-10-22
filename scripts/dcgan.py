#!/usr/bin/env python3
"""
DCGAN for generating synthetic facial expression images (48x48 grayscale).
You can optionally train one GAN per emotion or conditionally on labels.
"""
import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm


def make_generator(latent_dim=100):
    model = Sequential([
        layers.Dense(6 * 6 * 128, input_dim=latent_dim),
        layers.LeakyReLU(0.2),
        layers.Reshape((6, 6, 128)),
        layers.Conv2DTranspose(128, 4, strides=2, padding='same'),
        layers.LeakyReLU(0.2),
        layers.Conv2DTranspose(64, 4, strides=2, padding='same'),
        layers.LeakyReLU(0.2),
        layers.Conv2DTranspose(1, 4, strides=2, padding='same', activation='tanh')
    ])
    return model


def make_discriminator(input_shape=(48, 48, 1)):
    model = Sequential([
        layers.Conv2D(64, 4, strides=2, padding='same', input_shape=input_shape),
        layers.LeakyReLU(0.2),
        layers.Conv2D(128, 4, strides=2, padding='same'),
        layers.LeakyReLU(0.2),
        layers.Flatten(),
        layers.Dense(1)
    ])
    return model


def generate_and_save_images(generator, epoch, output_dir, latent_dim=100, n=25):
    noise = np.random.randn(n, latent_dim)
    gen_imgs = generator.predict(noise)
    gen_imgs = 0.5 * gen_imgs + 0.5  # Rescale to [0,1]

    import matplotlib.pyplot as plt
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(5, 5))
    for i in range(n):
        plt.subplot(5, 5, i + 1)
        plt.imshow(gen_imgs[i, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"epoch_{epoch:03d}.png"))
    plt.close()


def train_dcgan(data_dir, output_dir, epochs=20000, batch_size=64, latent_dim=100):
    datagen = ImageDataGenerator(rescale=1. / 255)
    dataset = datagen.flow_from_directory(
        data_dir, target_size=(48, 48), color_mode='grayscale',
        batch_size=batch_size, class_mode=None
    )

    generator = make_generator(latent_dim)
    discriminator = make_discriminator()

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    optimizer_gen = tf.keras.optimizers.Adam(1e-4)
    optimizer_disc = tf.keras.optimizers.Adam(1e-4)

    @tf.function
    def train_step(real_imgs):
        noise = tf.random.normal([batch_size, latent_dim])
        fake_imgs = generator(noise, training=True)

        with tf.GradientTape() as disc_tape:
            real_output = discriminator(real_imgs, training=True)
            fake_output = discriminator(fake_imgs, training=True)
            disc_loss = cross_entropy(tf.ones_like(real_output), real_output) + \
                        cross_entropy(tf.zeros_like(fake_output), fake_output)

        grads_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        optimizer_disc.apply_gradients(zip(grads_disc, discriminator.trainable_variables))

        with tf.GradientTape() as gen_tape:
            noise = tf.random.normal([batch_size, latent_dim])
            fake_imgs = generator(noise, training=True)
            fake_output = discriminator(fake_imgs, training=True)
            gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)

        grads_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
        optimizer_gen.apply_gradients(zip(grads_gen, generator.trainable_variables))

        return disc_loss, gen_loss

    os.makedirs(output_dir, exist_ok=True)
    print("Starting DCGAN training... Press Ctrl+C to stop.")
    step = 0
    for epoch in tqdm(range(epochs)):
        real_imgs = next(dataset)
        if real_imgs.shape[0] != batch_size:
            continue
        d_loss, g_loss = train_step(real_imgs)
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: D_loss={d_loss.numpy():.4f}, G_loss={g_loss.numpy():.4f}")
            generate_and_save_images(generator, epoch, output_dir, latent_dim)

    generator.save(os.path.join(output_dir, "dcgan_generator.h5"))
    discriminator.save(os.path.join(output_dir, "dcgan_discriminator.h5"))
    print("Training complete. Models saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/cropped_faces/train', help='Path to training images (48x48 grayscale)')
    parser.add_argument('--out', type=str, default='outputs/generated_faces', help='Output folder for generated images')
    parser.add_argument('--epochs', type=int, default=20000)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--latent', type=int, default=100)
    args = parser.parse_args()

    train_dcgan(args.data, args.out, epochs=args.epochs, batch_size=args.batch, latent_dim=args.latent)
