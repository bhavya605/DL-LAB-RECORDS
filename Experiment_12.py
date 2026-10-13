import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

DATA_DIR = "D:\\University\\SEM-V\\Deep Learning\\DL LAB\\archive\\flowers"  
IMG_HEIGHT = 96
IMG_WIDTH  = 96
BATCH_SIZE = 32
ENCODING_SIZE = 8        # small for demo; increase (64/128/256) for better quality
EPOCHS = 20              # tune for your run
AUTOTUNE = tf.data.AUTOTUNE
SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# labeled generator (for visualizations and per-class statistics)
labeled_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    seed=84,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='int'   # returns (images, labels)
)

flower_names = labeled_ds.class_names
print("Classes:", flower_names)

# unlabeled train/val datasets for VAE training (images only)
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    seed=SEED,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode=None,
    validation_split=0.2,
    subset='training'
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    seed=SEED,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode=None,
    validation_split=0.2,
    subset='validation'
)

# Normalize to [0,1] and create (input, target) pairs
def norm_map(x):
    x = tf.cast(x, tf.float32) / 255.0
    return x, x

train_ds = train_ds.map(norm_map, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
val_ds   = val_ds.map(norm_map, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)

# quick sample plot (from labeled generator)
plt.figure(figsize=(10, 12))
for images, labels in labeled_ds.take(1):
    n = min(12, images.shape[0])
    for i in range(n):
        ax = plt.subplot(6, 2, i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(flower_names[int(labels[i].numpy())])
        plt.axis("off")
plt.suptitle("Sample images (with labels)")
plt.show()

class Sampling(layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        eps = tf.random.normal(tf.shape(mean))
        return mean + tf.exp(0.5 * log_var) * eps

# Encoder
encoder_inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = layers.Conv2D(64, 3, padding='same', activation='selu', kernel_initializer='lecun_normal')(encoder_inputs)
x = layers.Conv2D(128, 3, strides=2, padding='same', activation='selu', kernel_initializer='lecun_normal')(x)
x = layers.Conv2D(256, 3, strides=2, padding='same', activation='selu', kernel_initializer='lecun_normal')(x)
x = layers.Conv2D(512, 3, strides=2, padding='same', activation='selu', kernel_initializer='lecun_normal')(x)
x = layers.Flatten()(x)

codings_mean = layers.Dense(ENCODING_SIZE, name='z_mean')(x)
codings_log_var = layers.Dense(ENCODING_SIZE, name='z_log_var')(x)
codings = Sampling()([codings_mean, codings_log_var])

variational_encoder = keras.Model(encoder_inputs, [codings_mean, codings_log_var, codings], name='encoder')
variational_encoder.summary()

# Decoder
decoder_inputs = keras.Input(shape=(ENCODING_SIZE,))
# compute expected spatial dims after three stride-2 layers:
# 96 -> 48 -> 24 -> 12
spatial_h = IMG_HEIGHT // 8
spatial_w = IMG_WIDTH  // 8
x = layers.Dense(512 * spatial_h * spatial_w, activation='selu')(decoder_inputs)
x = layers.Reshape((spatial_h, spatial_w, 512))(x)
x = layers.Conv2DTranspose(512, 3, padding='same', activation='selu')(x)
x = layers.Conv2DTranspose(256, 3, strides=2, padding='same', activation='selu')(x)
x = layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='selu')(x)
x = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='selu')(x)
decoder_outputs = layers.Conv2DTranspose(3, 3, padding='same', activation='sigmoid')(x)

variational_decoder = keras.Model(decoder_inputs, decoder_outputs, name='decoder')
variational_decoder.summary()

# VAE Model: encoder -> sampling -> decoder
_, _, z = variational_encoder(encoder_inputs)
reconstructions = variational_decoder(z)
vae = keras.Model(encoder_inputs, reconstructions, name='vae')

# This Lambda computes the mean KL divergence over the batch and returns a scalar KerasTensor.
kl_loss_layer = layers.Lambda(
    lambda inputs: tf.reduce_mean(
        -0.5 * tf.reduce_sum(1.0 + inputs[1] - tf.square(inputs[0]) - tf.exp(inputs[1]), axis=1)
    ),
    name='kl_loss_mean'
)([codings_mean, codings_log_var])

# add KL loss (symbolic) to model losses
vae.add_loss(kl_loss_layer)

def lr_schedule(epoch):
    return 1e-4 * (0.5 ** (epoch // 10))

lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)
early_stop = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

vae.compile(optimizer=keras.optimizers.Nadam(), loss='mse')

history = vae.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[lr_scheduler, early_stop],
    verbose=2
)

# save model(s)
vae.save('vae_trained.h5')
variational_encoder.save('vae_encoder.h5')
variational_decoder.save('vae_decoder.h5')

for images, _ in val_ds.take(1):
    recon = vae.predict(images)
    n = min(8, images.shape[0])
    plt.figure(figsize=(12, 6))
    for i in range(n):
        ax = plt.subplot(2, n, i+1)
        plt.imshow(images[i].numpy().astype('uint8') if images.dtype == tf.uint8 else images[i].numpy())
        plt.title("original"); plt.axis('off')
        ax = plt.subplot(2, n, n+i+1)
        plt.imshow(recon[i])
        plt.title("reconstructed"); plt.axis('off')
    plt.suptitle("Original vs Reconstructed (validation samples)")
    plt.show()
    break

random_codings = tf.random.normal(shape=(20, ENCODING_SIZE))
generated = variational_decoder.predict(random_codings, batch_size=20)

plt.figure(figsize=(12, 8))
for i, img in enumerate(generated):
    ax = plt.subplot(4, 5, i+1)
    plt.imshow(img)
    plt.axis('off')
plt.suptitle("Random samples from VAE decoder")
plt.show()

flower_means = {name: {'sum': np.zeros(ENCODING_SIZE), 'count': 0} for name in flower_names}
flower_examples = {name: [] for name in flower_names}

# iterate labeled_ds (images are uint8 in [0,255]); convert to [0,1]
for images, labels in labeled_ds:
    imgs = images.numpy().astype('float32') / 255.0
    means, logvars, _ = variational_encoder.predict(imgs, batch_size=imgs.shape[0])
    for idx in range(imgs.shape[0]):
        cname = flower_names[int(labels[idx].numpy())]
        flower_means[cname]['sum'] += means[idx]
        flower_means[cname]['count'] += 1
        # keep a few example images per class for display
        if len(flower_examples[cname]) < 3:
            flower_examples[cname].append(imgs[idx])

# compute mean vectors and decode to get "average" flower per class
for name in flower_names:
    cnt = flower_means[name]['count']
    if cnt > 0:
        mean_vec = flower_means[name]['sum'] / cnt
    else:
        mean_vec = np.zeros(ENCODING_SIZE)
    flower_means[name]['mean_vec'] = mean_vec
    flower_means[name]['avg_image'] = variational_decoder.predict(mean_vec[None, :])[0]

n_classes = len(flower_names)
cols = 5
rows = n_classes
plt.figure(figsize=(cols*3, rows*2.5))
for i, name in enumerate(flower_names):
    exs = flower_examples[name]
    # show up to 3 examples
    for j in range(3):
        ax = plt.subplot(rows, cols, i*cols + (j+1))
        if j < len(exs):
            plt.imshow(exs[j])
        plt.title(f"{name} ex{j+1}")
        plt.axis('off')
    # show reconstruction of a sample (reconstruct ex 0 if exists)
    ax = plt.subplot(rows, cols, i*cols + 4)
    if len(exs) > 0:
        rec = vae.predict(exs[0][None, ...])[0]
        plt.imshow(rec)
    plt.title(f"{name} recon")
    plt.axis('off')
    # show average
    ax = plt.subplot(rows, cols, i*cols + 5)
    plt.imshow(flower_means[name]['avg_image'])
    plt.title(f"avg {name}")
    plt.axis('off')
plt.suptitle("Per-class: examples | reconstruction | average (decoded mean)")
plt.show()
