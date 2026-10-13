import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model

# Activation Functions with Plots
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

x = np.linspace(-5, 5, 200)
plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
plt.plot(x, sigmoid(x))
plt.title("Sigmoid")
plt.subplot(2,2,2)
plt.plot(x, relu(x))
plt.title("ReLU")
plt.subplot(2,2,3)
plt.plot(x, tanh(x))
plt.title("Tanh")
plt.subplot(2,2,4)
x_soft = np.linspace(-2, 2, 5)
plt.bar(range(len(x_soft)), softmax(x_soft))
plt.title("Softmax Example")
plt.show()


# Loss Functions with Plots
def mse(y_true, y_pred):                                    #mean squared error
    return np.mean((y_true - y_pred)**2)

def cross_entropy(y_true, y_pred):                          #cross-entropy
    eps = 1e-10
    return -np.sum(y_true * np.log(y_pred + eps)) / y_true.shape[0]

y_true = np.array([1])                                      #plot for mse & cross-entropy for binary classification
y_pred = np.linspace(0, 1, 100)
mse_values = [(1 - yp)**2 for yp in y_pred]

y_true_binary = np.array([[1,0]])                           #class 0
y_pred_vals = np.linspace(0.01, 0.99, 100)
ce_values = [-np.log(y_pred_vals)]                          # when true class = 1

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(y_pred, mse_values)
plt.title("MSE Loss (y=1)")
plt.subplot(1,2,2)
plt.plot(y_pred_vals, ce_values[0])
plt.title("Cross-Entropy Loss (class=1)")
plt.show()


# Backpropagation on a small NN
X = np.array([[0,0],[0,1],[1,0],[1,1]])                                
y = np.array([[0],[1],[1],[0]])

def sigmoid(x): 
    return 1/(1+np.exp(-x))
def sigmoid_deriv(x): 
    return x*(1-x)

np.random.seed(42)                                          #initializing weights
W1 = np.random.randn(2,2)
b1 = np.zeros((1,2))
W2 = np.random.randn(2,1)
b2 = np.zeros((1,1))
lr = 0.1
epochs = 10000
losses = []

for epoch in range(epochs):                                   #training
    h_in = np.dot(X, W1) + b1                                 #forward pass
    h_out = sigmoid(h_in)
    out_in = np.dot(h_out, W2) + b2
    out = sigmoid(out_in)

    loss = np.mean((y - out)**2)                              #loss(mse)
    losses.append(loss)

    d_out = (y - out) * sigmoid_deriv(out)                    #backpropagation
    dW2 = np.dot(h_out.T, d_out)
    db2 = np.sum(d_out, axis=0, keepdims=True)
    d_hidden = d_out.dot(W2.T) * sigmoid_deriv(h_out)
    dW1 = np.dot(X.T, d_hidden)
    db1 = np.sum(d_hidden, axis=0, keepdims=True)

    W1 += lr*dW1                                               #update weights
    b1 += lr*db1
    W2 += lr*dW2
    b2 += lr*db2
print("Final predictions:")
print(out.round())
plt.plot(losses)
plt.title("Loss Curve (Backprop Training)")
plt.show()


# Optimizer Comparison on Employee Dataset
df = pd.read_csv("D:\\University\\SEM-V\\Deep Learning\\DL LAB\\employee_data.csv")  
print("Dataset Shape:", df.shape)
print(df.head())
print(df.info())
print(df["Department"].value_counts())
print(df["Remote_Work"].value_counts())

df = df.dropna(subset=["Remote_Work"])                          #dropping target missing values 

df["Age"] = df["Age"].fillna(df["Age"].median())                #dropping missing values in features
df["Salary"] = df["Salary"].fillna(df["Salary"].median())
df["Experience_Years"] = df["Experience_Years"].fillna(df["Experience_Years"].median())
df["Bonus"] = df["Bonus"].fillna(df["Bonus"].median())

df["Department"] = df["Department"].fillna(df["Department"].mode()[0])

le = LabelEncoder()
df["Department"] = le.fit_transform(df["Department"])
df["Remote_Work"] = df["Remote_Work"].map({"Yes":1,"No":0})  

df = df.drop(columns=["ID","Name","Join_Date"])

X = df.drop("Remote_Work", axis=1)                              #features
y = df["Remote_Work"]                                           #target variables

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

#Sequential NN Function
def build_model(optimizer):
    model = Sequential([
        Dense(16, activation="relu", input_shape=(X_train.shape[1],)),
        Dense(8, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return model

#training the model with different Optimizers
optimizers = {
    "SGD": tf.keras.optimizers.SGD(learning_rate=0.01),
    "Momentum": tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    "Adam": tf.keras.optimizers.Adam(learning_rate=0.01)
}

histories = {}
for opt_name, opt in optimizers.items():
    print(f"\nTraining with {opt_name}...")
    model = build_model(opt)
    history = model.fit(
        X_train, y_train,
        epochs=50, batch_size=16,
        validation_data=(X_test, y_test),
        verbose=0
    )
    histories[opt_name] = (model, history)


# Visualizing Model Architecture
plot_model(build_model("adam"), show_shapes=True, show_layer_names=True, to_file="employee_model.png")

# Accuracy & Loss Curves for Optimizers
plt.figure(figsize=(14,6))
for i, (opt_name, (model, history)) in enumerate(histories.items()):
    plt.subplot(1,2,1)
    plt.plot(history.history["accuracy"], label=f"{opt_name} Train")
    plt.plot(history.history["val_accuracy"], linestyle="--", label=f"{opt_name} Val")
    plt.title("Accuracy Comparison"); plt.xlabel("Epochs"); plt.ylabel("Accuracy")

    plt.subplot(1,2,2)
    plt.plot(history.history["loss"], label=f"{opt_name} Train")
    plt.plot(history.history["val_loss"], linestyle="--", label=f"{opt_name} Val")
    plt.title("Loss Comparison"); plt.xlabel("Epochs"); plt.ylabel("Loss")
plt.legend()
plt.show()

# Performance Metrics & Confusion Matrix
for opt_name, (model, history) in histories.items():
    print(f"\n=== Performance Report for {opt_name} ===")
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)                          #Sensitivity
    specificity = tn / (tn+fp)
    f1 = f1_score(y_test, y_pred)

    # ROC & AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    print(f"Accuracy:     {acc:.4f}")
    print(f"Precision:    {pre:.4f}")
    print(f"Recall (SEN): {rec:.4f}")
    print(f"Specificity:  {specificity:.4f}")
    print(f"F1-Score:     {f1:.4f}")
    print(f"AUC-ROC:      {roc_auc:.4f}")

    plt.figure(figsize=(5,4))                                   #Plot for the confusion matrix
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Remote","Remote"], yticklabels=["No Remote","Remote"])
    plt.title(f"Confusion Matrix - {opt_name}")
    plt.ylabel("True Label"); plt.xlabel("Predicted Label")
    plt.show()

    plt.figure(figsize=(6,6))                                    #ROC curves
    plt.plot(fpr, tpr, label=f"{opt_name} (AUC={roc_auc:.2f})")
    plt.plot([0,1], [0,1], "k--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {opt_name}")
    plt.legend()
    plt.show()
