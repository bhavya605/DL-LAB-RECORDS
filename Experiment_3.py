import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Data Preprocessing
df = pd.read_csv("D:\\University\\SEM-V\\Deep Learning\\DL LAB\\Music-Genre-Classification\\Data\\features_3_sec.csv")   # replace with your actual file name
print("Shape of dataset:", df.shape)
print(df.head())

print("\nMissing values:\n", df.isnull().sum()) 

df = df.drop_duplicates() 

for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].fillna(df[col].median())          

X = df.drop(["filename", "length", "label"], axis=1)
y = df["label"]

le = LabelEncoder()            
y_encoded = le.fit_transform(y)

scaler = StandardScaler()  
X_scaled = scaler.fit_transform(X)

print("\nFeature matrix shape:", X_scaled.shape)
print("Target vector shape:", y_encoded.shape)
print("Classes:", le.classes_)


# EDA (Exploratory Data Analysis)
plt.figure(figsize=(8,4))                               # checking the class distribution
sns.countplot(x=y, order=y.value_counts().index)
plt.title("Class Distribution of Genres")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
print("\nSummary statistics of features:\n", pd.DataFrame(X).describe())   # checking basic stats for the numeric data
sample_features = list(X.columns[:5])                   # outlier detection
plt.figure(figsize=(12,6))
X[sample_features].boxplot()
plt.title("Boxplots for first 5 features (outlier check)")
plt.xticks(rotation=45)
plt.show()


# NN Model 
input_dim = X_scaled.shape[1]
num_classes = len(le.classes_)
model = models.Sequential([
    layers.Input(shape=(input_dim,)),                   # I/P layer
    layers.Dense(64, activation='relu'),                # Hidden layer
    layers.Dense(num_classes, activation='softmax')])   #O/P layer
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',             # labels are integers
    metrics=['accuracy'])
model.summary()

# Training & Testing the model
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
print("Training set:", X_train.shape, y_train.shape)
print("Testing set:", X_test.shape, y_test.shape)
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),  
    epochs=20,   
    batch_size=32,
    verbose=1)


# Model Performance Analysis
y_pred_probs = model.predict(X_test)          
y_pred = y_pred_probs.argmax(axis=1)          

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro')
rec = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
print("\nPerformance Metrics:")
print("Accuracy :", acc)
print("Precision:", prec)
print("Recall   :", rec)
print("F1-score :", f1)

print("\nClassification Report:")                       # detailed classification report per genre
print(classification_report(y_test, y_pred, target_names=le.classes_))

cm = confusion_matrix(y_test, y_pred)    
print("\nConfusion Matrix:\n", cm)

# Visualising training/validation accuracy and loss
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title("Accuracy over epochs")
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Loss over epochs")
plt.show()

# Confusion Matrix Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()