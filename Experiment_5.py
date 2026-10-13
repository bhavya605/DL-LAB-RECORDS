import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, accuracy_score

DATA_ROOT = "D:\\University\\SEM-V\\Deep Learning\\DL LAB\\leapGestRecog\\leapGestRecog"
BATCH_SIZE = 32       # reduce to 16 or 8 if CPU is slow
NUM_EPOCHS = 5        # increase later for better accuracy
LR = 1e-4
VAL_SPLIT = 0.2       # fraction of data to use as validation
NUM_WORKERS = 0       # use 0 on Windows / CPU, >0 if you have a stable Linux/GPU setup
SAVE_MODEL_PATH = "resnet18_finetuned.pth"
RANDOM_SEED = 42

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def imshow_batch(batch_tensor, classes, imgs_per_row=6):
    # batch_tensor: (B,C,H,W) in [0,1] (we un-normalize inside)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    batch = batch_tensor.cpu().permute(0,2,3,1).numpy()
    batch = (batch * std) + mean
    batch = np.clip(batch, 0, 1)
    n = batch.shape[0]
    cols = min(imgs_per_row, n)
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(cols*2, rows*2))
    for i in range(n):
        plt.subplot(rows, cols, i+1)
        plt.imshow(batch[i])
        plt.title(classes[i])
        plt.axis('off')
    plt.show()

def build_dataloaders(root, batch_size=32, val_split=0.2, num_workers=0):
    # transforms tuned for pretrained models: resize -> center crop -> to tensor -> normalize
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    # validation should NOT have random flips
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    full_dataset = datasets.ImageFolder(root, transform=train_transform)

    # split into train/val
    total = len(full_dataset)
    val_size = int(val_split * total)
    train_size = total - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(RANDOM_SEED))
    # ensure val uses val_transform (override)
    val_ds.dataset.transform = val_transform

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    class_names = full_dataset.classes
    print(f"Found classes: {class_names}")
    print(f"Total images: {total} -> train: {train_size}, val: {val_size}")
    return train_loader, val_loader, class_names

def build_model(num_classes, device):
    model = models.resnet18(pretrained=True)
    # replace final fc
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model = model.to(device)
    return model

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_trues = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_preds.append(preds)
        all_trues.append(yb.detach().cpu().numpy())
    avg_loss = running_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_trues = np.concatenate(all_trues)
    acc = accuracy_score(all_trues, all_preds)
    return avg_loss, acc

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_trues = []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            running_loss += loss.item() * xb.size(0)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.append(preds)
            all_trues.append(yb.cpu().numpy())
    avg_loss = running_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_trues = np.concatenate(all_trues)
    acc = accuracy_score(all_trues, all_preds)
    return avg_loss, acc, all_trues, all_preds

def plot_curves(train_losses, val_losses, train_accs, val_accs):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='train loss')
    plt.plot(val_losses, label='val loss')
    plt.xlabel('epoch'); plt.ylabel('loss'); plt.legend(); plt.title('Loss')

    plt.subplot(1,2,2)
    plt.plot(train_accs, label='train acc')
    plt.plot(val_accs, label='val acc')
    plt.xlabel('epoch'); plt.ylabel('accuracy'); plt.legend(); plt.title('Accuracy')
    plt.show()

def main():
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if not os.path.isdir(DATA_ROOT):
        raise SystemExit(f"Dataset folder not found at: {DATA_ROOT}\nMake sure the folder contains class subfolders (00, 01_palm, ...).")

    train_loader, val_loader, class_names = build_dataloaders(DATA_ROOT, batch_size=BATCH_SIZE, val_split=VAL_SPLIT, num_workers=NUM_WORKERS)
    # show a small sample batch
    xb, yb = next(iter(train_loader))
    sample_labels = [class_names[int(i)] for i in yb[:8]]
    imshow_batch(xb[:8], sample_labels)

    model = build_model(len(class_names), device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(1, NUM_EPOCHS+1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        train_losses.append(train_loss); val_losses.append(val_loss)
        train_accs.append(train_acc); val_accs.append(val_acc)
        print(f"Epoch {epoch}/{NUM_EPOCHS}  TrainLoss: {train_loss:.4f}  TrainAcc: {train_acc:.3f}  ValLoss: {val_loss:.4f}  ValAcc: {val_acc:.3f}")

    # plots
    plot_curves(train_losses, val_losses, train_accs, val_accs)

    # confusion matrix on validation
    _, _, y_true, y_pred = evaluate(model, val_loader, criterion, device)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns = None
    try:
        import seaborn as sns
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    except Exception:
        plt.imshow(cm, cmap='viridis')
        plt.title('Confusion matrix')
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.show()

    # save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names
    }, SAVE_MODEL_PATH)
    print("Model saved to", SAVE_MODEL_PATH)

if __name__ == "__main__":
    main()
