import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.datasets import mnist
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, auc
)

# ============================================================
# 1. DATASET LOADING
# ============================================================
from tensorflow.keras.preprocessing.image import ImageDataGenerator

dataset_dir = r"C:\DataSets\archive (12-4)\flowers"
img_size = (64, 64)

datagen = ImageDataGenerator(rescale=1./255)
train_gen = datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=32,
    class_mode=None,
    shuffle=True
)

# Load 1000 images for training
print("Loading training images...")
x_train = []
for i, batch in enumerate(train_gen):
    x_train.append(batch)
    if len(x_train) * 32 >= 1000:
        break
x_train = np.concatenate(x_train, axis=0)[:1000]

latent_dim = 100

# ============================================================
# 2. EDA + PREPROCESSING
# ============================================================
print("Training Data Shape:", x_train.shape)

plt.figure(figsize=(10,3))
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(x_train[i])
    plt.title("Real")
    plt.axis("off")
plt.savefig("real_samples.png")
print("Real samples saved as real_samples.png")
plt.close()

# ============================================================
# 3. BUILDING GAN (Generator + Discriminator)
# ============================================================

# Generator
def build_generator():
    inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(8*8*256, activation="relu")(inputs)
    x = layers.Reshape((8,8,256))(x)
    x = layers.Conv2DTranspose(128, 4, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(64, 4, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(3, 4, strides=2, padding="same", activation="sigmoid")(x)
    return Model(inputs, x)

# Discriminator
def build_discriminator():
    inputs = layers.Input(shape=(64,64,3))
    x = layers.Conv2D(64, 4, strides=2, activation="relu")(inputs)
    x = layers.Conv2D(128, 4, strides=2, activation="relu")(x)
    x = layers.Conv2D(256, 4, strides=2, activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1, activation="sigmoid")(x)
    return Model(inputs, x)

generator = build_generator()
discriminator = build_discriminator()

discriminator.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Combined GAN
discriminator.trainable = False
gan_input = layers.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = Model(gan_input, gan_output)
gan.compile(optimizer="adam", loss="binary_crossentropy")

# ============================================================
# 4. TRAINING THE MODEL
# ============================================================
epochs = 1000  # Reduced for faster training
batch_size = 32

d_loss_hist = []
g_loss_hist = []

for step in range(epochs):

    # Real samples
    idx = np.random.randint(0, x_train.shape[0], batch_size)
    real_imgs = x_train[idx]
    real_labels = np.ones((batch_size,1))

    # Fake samples
    noise = np.random.normal(0,1,(batch_size,latent_dim))
    fake_imgs = generator.predict(noise)
    fake_labels = np.zeros((batch_size,1))

    # Train discriminator
    d_loss_real = discriminator.train_on_batch(real_imgs, real_labels)
    d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_labels)
    d_loss = 0.5 * (d_loss_real[0] + d_loss_fake[0])

    # Train generator
    noise = np.random.normal(0,1,(batch_size,latent_dim))
    g_loss = gan.train_on_batch(noise, np.ones((batch_size,1)))

    d_loss_hist.append(d_loss)
    g_loss_hist.append(g_loss)

    if step % 200 == 0:
        print(f"Step {step} | D Loss: {d_loss:.4f} | G Loss: {g_loss:.4f}")

# ============================================================
# 5. VISUALIZING MODEL ARCHITECTURE
# ============================================================
tf.keras.utils.plot_model(generator, to_file="generator_arch.png", show_shapes=True)
tf.keras.utils.plot_model(discriminator, to_file="discriminator_arch.png", show_shapes=True)

# ============================================================
# 6. ACCURACY & LOSS CURVES
# ============================================================
plt.plot(d_loss_hist, label="Discriminator Loss")
plt.plot(g_loss_hist, label="Generator Loss")
plt.title("GAN Loss Curves")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss_curves.png")
print("Loss curves saved as loss_curves.png")
plt.close()

# ============================================================
# 7. CONFUSION MATRIX (Discriminator classification real vs fake)
# ============================================================
noise = np.random.normal(0,1,(500,latent_dim))
fake_samples = generator.predict(noise)

real_preds = discriminator.predict(x_train[:500])
fake_preds = discriminator.predict(fake_samples)

y_true = np.array([1]*500 + [0]*500)
y_pred = np.array((real_preds > 0.5).astype(int).tolist() + 
                  (fake_preds > 0.5).astype(int).tolist())

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.savefig("confusion_matrix.png")
print("Confusion matrix saved as confusion_matrix.png")
plt.close()

# ============================================================
# 8. METRICS (ACC, PRECISION, RECALL, F1, ROC, AUC)
# ============================================================
acc = accuracy_score(y_true, y_pred)
pre = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

fpr, tpr, _ = roc_curve(y_true, y_pred)
auc_val = auc(fpr, tpr)

print("Accuracy:", acc)
print("Precision:", pre)
print("Sensitivity:", rec)
print("Specificity:", cm[0,0] / (cm[0,0] + cm[0,1]))
print("F1 Score:", f1)
print("AUC:", auc_val)

plt.plot(fpr, tpr)
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.savefig("roc_curve.png")
print("ROC curve saved as roc_curve.png")
plt.close()

# ============================================================
# IMAGE GENERATION OUTPUT
# ============================================================
noise = np.random.normal(0,1,(25,latent_dim))
gen_imgs = generator.predict(noise)

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(gen_imgs[i])
    plt.axis("off")
plt.suptitle("Generated Images (GAN)")
plt.savefig("generated_images.png")
print("Generated images saved as generated_images.png")
plt.close()

print("\nGAN training complete!")
