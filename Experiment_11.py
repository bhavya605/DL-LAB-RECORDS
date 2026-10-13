import os
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle as pkl
from collections import defaultdict
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import umap
import scipy.spatial

DATA_DIR    = "D:\\University\\SEM-V\\Deep Learning\\DL LAB\\archive\\flowers"   # change this to your dataset folder
IMG_HEIGHT  = 96
IMG_WIDTH   = 96
BATCH_SIZE  = 64
ENCODING_SIZE= 256        # smaller -> faster; original used 1024
EPOCHS      = 10          # set low for demo; increase for better results
AUTOTUNE    = tf.data.AUTOTUNE

# For UMAP and embedding visualization (limit to keep it fast)
EMBED_LIMIT = 2000        # set None to use all images (may be slow)
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.1

print("TensorFlow version:", tf.__version__)
print("Using dataset dir:", DATA_DIR)

# Full generator with inferred labels (for visualization & embeddings)
flower_generator = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    seed=84,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='int'   # returns (images, labels)
)

flower_class_names = flower_generator.class_names
print("Found classes:", flower_class_names)

# Train/validation generators (no labels, for autoencoder input)
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    seed=42,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode=None,           # we only need images for autoencoder (input==output)
    validation_split=0.2,
    subset='training'
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    seed=42,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode=None,
    validation_split=0.2,
    subset='validation'
)

# Normalization mapping function (images -> images/255, images/255)
def replicate_inputs_to_outputs(images):
    images = tf.cast(images, tf.float32) / 255.0
    return images, images

# Apply map, cache and prefetch for performance
train_ds = train_ds.map(replicate_inputs_to_outputs, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
val_ds   = val_ds.map(replicate_inputs_to_outputs, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)

# Quick sanity: show a few images from labeled generator
plt.figure(figsize=(10, 14))
for images, labels in flower_generator.take(1):
    for i in range(min(12, images.shape[0])):
        ax = plt.subplot(6, 2, i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(flower_class_names[labels[i].numpy()])
        plt.axis("off")
plt.suptitle("Sample images (with labels)"); plt.show()

# Encoder
encoder = keras.models.Sequential([
    layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.Flatten(),
    layers.Dense(ENCODING_SIZE, name='bottleneck')
], name='encoder')

encoder.summary()

# Decoder
decoder = keras.models.Sequential([
    layers.Input(shape=(ENCODING_SIZE,)),
    layers.Dense(128 * (IMG_HEIGHT//4) * (IMG_WIDTH//4), activation='relu'),
    layers.Reshape((IMG_HEIGHT//4, IMG_WIDTH//4, 128)),
    layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu'),
    layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu'),
    layers.Conv2D(3, 3, padding='same', activation='sigmoid')   # outputs in [0,1]
], name='decoder')

decoder.summary()

# Autoencoder: encoder -> decoder
ae = keras.models.Sequential([encoder, decoder], name='autoencoder')

def exponential_decay_fn(epoch):
    return 1e-3 * 0.1 ** (epoch / 10)

lr_scheduler = keras.callbacks.LearningRateScheduler(exponential_decay_fn)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

ae.compile(optimizer=keras.optimizers.Nadam(), loss='mse')
tf.random.set_seed(42)

history = ae.fit(train_ds,
                 validation_data=val_ds,
                 epochs=EPOCHS,
                 callbacks=[lr_scheduler, early_stopping_cb],
                 verbose=2)

# Save model (optional)
ae.save('trained_ae.h5')

# Use validation dataset to visualize some reconstructions
for images, _ in val_ds.take(1):
    recon = ae.predict(images)
    n = min(8, images.shape[0])
    plt.figure(figsize=(12, 6))
    for i in range(n):
        ax = plt.subplot(2, n, i+1)
        plt.imshow(images[i].numpy().astype('uint8') if images.dtype==tf.uint8 else (images[i].numpy()/255.0))
        plt.title("original"); plt.axis('off')

        ax = plt.subplot(2, n, n + i + 1)
        plt.imshow(recon[i])
        plt.title("reconstructed"); plt.axis('off')
    plt.suptitle("Original vs Reconstructed (validation samples)")
    plt.show()
    break

# Extract image arrays and labels (up to EMBED_LIMIT)
images_list = []
labels_list = []
count = 0
for images, labels in flower_generator:
    for i in range(images.shape[0]):
        images_list.append(images[i].numpy().astype('float32') / 255.0)
        labels_list.append(int(labels[i].numpy()))
        count += 1
        if EMBED_LIMIT and count >= EMBED_LIMIT:
            break
    if EMBED_LIMIT and count >= EMBED_LIMIT:
        break

images_arr = np.stack(images_list, axis=0)
labels_arr = np.array(labels_list)
print("Embeddings build: images:", images_arr.shape, "labels:", labels_arr.shape)

# Get encoder outputs (embeddings)
embeddings = encoder.predict(images_arr, batch_size=128)
print("Embeddings shape:", embeddings.shape)

# Standardize embeddings for UMAP
scaler = StandardScaler()
scaled_embeddings = scaler.fit_transform(embeddings)

# Run UMAP (fast-ish settings)
umap_reducer = umap.UMAP(n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST, metric='cosine', random_state=42)
umap_embedding = umap_reducer.fit_transform(scaled_embeddings)
print("UMAP output shape:", umap_embedding.shape)

colors = plt.cm.tab20(np.linspace(0, 1, max(16, len(flower_class_names))))
plt.figure(figsize=(10,10))
for i, cname in enumerate(flower_class_names):
    idxs = np.where(labels_arr == i)[0]
    if len(idxs) == 0:
        continue
    plt.scatter(umap_embedding[idxs,0], umap_embedding[idxs,1], c=[colors[i]], label=cname, s=8)
plt.legend(markerscale=2)
plt.title("UMAP projection of encoder embeddings (sampled)")
plt.show()

# choose some grid points and show images near them (fast; uses a few grid points)
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# build quick mapping from indices to image tensors for thumbnails
# Keep images scaled to uint8 for display
thumbs = (images_arr * 255).astype('uint8')

fig, ax = plt.subplots(figsize=(10,10))
for i, cname in enumerate(flower_class_names):
    idxs = np.where(labels_arr == i)[0]
    if len(idxs) == 0: continue
    ax.scatter(umap_embedding[idxs,0], umap_embedding[idxs,1], c=[colors[i]], s=8, label=cname)
ax.legend(markerscale=2)

# select a few grid points (coarse) and annotate with 3 thumbnails each if available
sample_points = []
# pick random points near cluster centers
for pt in [(0,0), (2,2), (-2,-2), (1,-1), (-1,1)]:
    # find k nearest samples to this grid point
    dists = scipy.spatial.distance.cdist(np.array([pt]), umap_embedding)
    order = np.argsort(dists[0])[:4]
    sample_points.append((pt, order.tolist()))

for pt, order in sample_points:
    for j, idx in enumerate(order):
        img_thumb = thumbs[idx]
        imagebox = OffsetImage(img_thumb, zoom=0.5)
        ab = AnnotationBbox(imagebox, (umap_embedding[idx,0], umap_embedding[idx,1]), frameon=True)
        ax.add_artist(ab)

plt.title("UMAP with sample thumbnails (few points)")
plt.show()

# pick two random embeddings from the same class or different classes
def latent_interpolation_demo(idx_a, idx_b, steps=10):
    emb_a = embeddings[idx_a]
    emb_b = embeddings[idx_b]
    steps_vecs = [emb_a + (emb_b - emb_a) * (i / (steps-1)) for i in range(steps)]
    steps_arr = np.stack(steps_vecs, axis=0)
    reconstructions = decoder.predict(steps_arr)
    plt.figure(figsize=(16,4))
    for i, recon in enumerate(reconstructions):
        ax = plt.subplot(1, steps, i+1)
        plt.imshow(recon)
        plt.axis('off')
        plt.title(f"step {i}")
    plt.suptitle(f"Latent interpolation [{idx_a}] → [{idx_b}]")
    plt.show()

# pick two random indices (within the sampled embeddings)
if embeddings.shape[0] >= 2:
    idx_a, idx_b = 0, min(1, embeddings.shape[0]-1)
    latent_interpolation_demo(idx_a, idx_b, steps=8)

with open('embeddings_etc.pkl', 'wb') as fout:
    pkl.dump([embeddings, scaled_embeddings, umap_embedding, scaler, umap_reducer, labels_arr, flower_class_names], fout)

print("Done.")
