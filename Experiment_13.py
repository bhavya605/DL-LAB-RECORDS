import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

DATA_DIR = "D:\\University\\SEM-V\\Deep Learning\\DL LAB\\archive\\flowers"
IMG_SIZE = 64
BATCH_SIZE = 64
LATENT_DIM = 100
EPOCHS = 50

AUTOTUNE = tf.data.AUTOTUNE

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    label_mode=None,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Normalize to [-1, 1]
dataset = dataset.map(lambda x: (tf.cast(x, tf.float32) - 127.5) / 127.5)
dataset = dataset.prefetch(AUTOTUNE)

def build_generator():
    model = keras.Sequential([
        layers.Input(shape=(LATENT_DIM,)),
        layers.Dense(8 * 8 * 256, use_bias=False),
        layers.Reshape((8, 8, 256)),

        layers.Conv2DTranspose(128, 4, strides=2, padding="same", activation="relu"),
        layers.Conv2DTranspose(64, 4, strides=2, padding="same", activation="relu"),
        layers.Conv2DTranspose(32, 4, strides=2, padding="same", activation="relu"),

        layers.Conv2DTranspose(3, 3, activation="tanh", padding="same"),
    ])
    return model

def build_discriminator():
    model = keras.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Conv2D(64, 4, strides=2, padding="same"),
        layers.LeakyReLU(0.2),

        layers.Conv2D(128, 4, strides=2, padding="same"),
        layers.LeakyReLU(0.2),

        layers.Conv2D(256, 4, strides=2, padding="same"),
        layers.LeakyReLU(0.2),

        layers.Flatten(),
        layers.Dense(1)
    ])
    return model

generator = build_generator()
discriminator = build_discriminator()

cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)
gen_optimizer = keras.optimizers.Adam(2e-4, beta_1=0.5)
disc_optimizer = keras.optimizers.Adam(2e-4, beta_1=0.5)

def generator_loss(fake_logits):
    return cross_entropy(tf.ones_like(fake_logits), fake_logits)

def discriminator_loss(real_logits, fake_logits):
    real_loss = cross_entropy(tf.ones_like(real_logits), real_logits)
    fake_loss = cross_entropy(tf.zeros_like(fake_logits), fake_logits)
    return real_loss + fake_loss

@tf.function
def train_step(real_images):
    noise = tf.random.normal([BATCH_SIZE, LATENT_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        fake_images = generator(noise, training=True)

        real_logits = discriminator(real_images, training=True)
        fake_logits = discriminator(fake_images, training=True)

        g_loss = generator_loss(fake_logits)
        d_loss = discriminator_loss(real_logits, fake_logits)

    gradients_gen = gen_tape.gradient(g_loss, generator.trainable_variables)
    gradients_disc = disc_tape.gradient(d_loss, discriminator.trainable_variables)

    gen_optimizer.apply_gradients(zip(gradients_gen, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(gradients_disc, discriminator.trainable_variables))

    return g_loss, d_loss

def train(dataset, epochs):
    for epoch in range(epochs):
        g_losses = []
        d_losses = []

        print(f"Epoch {epoch+1}/{epochs}")
        for real_images in dataset:
            g_loss, d_loss = train_step(real_images)
            g_losses.append(g_loss)
            d_losses.append(d_loss)

        print(f" Generator Loss: {np.mean(g_losses):.4f} | Discriminator Loss: {np.mean(d_losses):.4f}")

        # generate sample after each epoch
        generate_and_show(epoch + 1)

    print("Training finished!")

def generate_and_show(epoch):
    noise = tf.random.normal([16, LATENT_DIM])
    generated = generator(noise, training=False)
    generated = (generated + 1) / 2.0  # back to [0,1]

    plt.figure(figsize=(6, 6))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(generated[i])
        plt.axis("off")
    plt.suptitle(f"Generated Samples - Epoch {epoch}")
    plt.show()

train(dataset, EPOCHS)
