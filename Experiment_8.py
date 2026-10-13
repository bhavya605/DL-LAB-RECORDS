import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, accuracy_score

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, padding=2)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x, return_activations=False):
        a1 = torch.relu(self.conv1(x))
        p1 = self.pool(a1)
        a2 = torch.relu(self.conv2(p1))
        p2 = self.pool(a2)
        flat = p2.view(p2.size(0), -1)
        h = torch.relu(self.fc1(flat))
        logits = self.fc2(h)
        if return_activations:
            return logits, [a1, a2]
        return logits

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    data_dir = "./data"
    train_full = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_set   = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    train_size = int(0.9 * len(train_full))
    val_size = len(train_full) - train_size
    train_set, val_set = random_split(train_full, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))

    batch_size = 256
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_set, batch_size=512, shuffle=False, num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=0, pin_memory=False)

    print(f"Train: {len(train_set)}  Val: {len(val_set)}  Test: {len(test_set)}")

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    n_epochs = 10
    train_losses, val_losses, val_accs = [], [], []

    for epoch in range(1, n_epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        model.eval()
        val_running_loss = 0.0
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_running_loss += loss.item() * xb.size(0)
                preds.append(torch.argmax(logits, dim=1).cpu().numpy())
                trues.append(yb.cpu().numpy())

        val_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        acc = accuracy_score(trues, preds)
        val_accs.append(acc)

        print(f"Epoch {epoch}/{n_epochs}  TrainLoss: {epoch_loss:.4f}  ValLoss: {val_loss:.4f}  ValAcc: {acc:.4f}")

    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(range(1, n_epochs+1), train_losses, label="train loss")
    plt.plot(range(1, n_epochs+1), val_losses, label="val loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(range(1, n_epochs+1), val_accs, label="val acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
    plt.show()

    all_preds, all_trues = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            logits = model(xb)
            all_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            all_trues.append(yb.numpy())
    all_preds = np.concatenate(all_preds); all_trues = np.concatenate(all_trues)
    cm = confusion_matrix(all_trues, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
    plt.show()

    sample_img, sample_label = test_set[0]
    x = sample_img.unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits, activations = model(x, return_activations=True)
        a1, a2 = activations

    plt.figure(figsize=(3,3))
    plt.imshow(sample_img.squeeze(0).cpu().numpy(), cmap='gray')
    plt.title(f"Label: {sample_label}"); plt.axis('off'); plt.show()

    n1 = a1.shape[1]
    plt.figure(figsize=(12,2))
    for i in range(n1):
        plt.subplot(1, n1, i+1)
        plt.imshow(a1[0, i].cpu().numpy(), cmap='gray'); plt.axis('off')
    plt.suptitle("Conv1 feature maps"); plt.show()

    n2 = a2.shape[1]; cols = 4; rows = int(np.ceil(n2/cols))
    plt.figure(figsize=(cols*2, rows*2))
    for i in range(n2):
        plt.subplot(rows, cols, i+1)
        plt.imshow(a2[0, i].cpu().numpy(), cmap='gray'); plt.axis('off')
    plt.suptitle("Conv2 feature maps"); plt.show()

    # test evaluation
    test_preds, test_trues = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            test_preds.append(torch.argmax(logits, dim=1).cpu().numpy()); test_trues.append(yb.numpy())
    test_preds = np.concatenate(test_preds); test_trues = np.concatenate(test_trues)
    test_acc = accuracy_score(test_trues, test_preds)
    print(f"Test accuracy: {test_acc*100:.2f}%")

if __name__ == "__main__":
    main()