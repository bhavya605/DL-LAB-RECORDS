import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import albumentations as A

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_PATH = "D:\\University\\SEM-V\\Deep Learning\\DL LAB\\archive (2)\\img_dir\\train"    # change if local
MASK_PATH  = "D:\\University\\SEM-V\\Deep Learning\\DL LAB\\ann_dir\\train"   # change if local

SAMPLE_LIMIT = 1000    # set to None to use all; lower => faster demo
INPUT_SIZE = 256       # 256 is faster than 512; change to 128 to go faster
BATCH_SIZE = 4
NUM_WORKERS = 0        # keep 0 on Windows / Kaggle CPU; set >0 on Linux GPU
NUM_EPOCHS = 8         # small for demo; increase for real training
LR = 1e-4
WEIGHT_DECAY = 1e-6
N_CLASSES = 7

print("Device:", DEVICE)
random.seed(42); np.random.seed(42); torch.manual_seed(42)

def create_df(image_path):
    ids = []
    for root, _, files in os.walk(image_path):
        for f in files:
            name = os.path.splitext(f)[0]   # e.g. 'area123'
            # keep suffix digits (original code stripped letters)
            # extract digits inside name
            digits = ''.join([c for c in name if c.isdigit()])
            if digits == '':
                continue
            ids.append(digits)
    df = pd.DataFrame({'id': ids})
    df = df.drop_duplicates().reset_index(drop=True)
    return df

df = create_df(IMAGE_PATH)
if SAMPLE_LIMIT:
    df = df.sample(min(SAMPLE_LIMIT, len(df)), random_state=42).reset_index(drop=True)

print("Total images used:", len(df))

from sklearn.model_selection import train_test_split
ids = df['id'].values
X_trainval, X_test = train_test_split(ids, test_size=0.10, random_state=19)
X_train, X_val = train_test_split(X_trainval, test_size=0.15, random_state=19)

print('Train:', len(X_train), 'Val:', len(X_val), 'Test:', len(X_test))

train_transform = A.Compose([
    A.Resize(INPUT_SIZE, INPUT_SIZE, interpolation=cv2.INTER_NEAREST),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.GridDistortion(p=0.2),
    A.RandomBrightnessContrast(p=0.2),
    A.GaussNoise(p=0.2)
])

val_transform = A.Compose([
    A.Resize(INPUT_SIZE, INPUT_SIZE, interpolation=cv2.INTER_NEAREST)
])

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
to_tensor_norm = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

class DroneDataset(Dataset):
    def __init__(self, img_path, mask_path, ids, transform=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        idx_name = self.ids[idx]
        img = cv2.imread(os.path.join(self.img_path, f'area{idx_name}.png'), cv2.IMREAD_COLOR)
        mask = cv2.imread(os.path.join(self.mask_path, f'area{idx_name}.png'), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            # fallback: return a zero sample (shouldn't happen with clean dataset)
            img = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
            mask = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        # normalize and convert
        img = Image.fromarray(img)
        img = to_tensor_norm(img)
        mask = torch.from_numpy(mask).long()
        return img, mask

train_set = DroneDataset(IMAGE_PATH, MASK_PATH, X_train, transform=train_transform)
val_set   = DroneDataset(IMAGE_PATH, MASK_PATH, X_val, transform=val_transform)
test_set  = DroneDataset(IMAGE_PATH, MASK_PATH, X_test, transform=val_transform)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=(DEVICE.type=='cuda'))
val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(DEVICE.type=='cuda'))
test_loader  = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(DEVICE.type=='cuda'))

print("Batches (train/val/test):", len(train_loader), len(val_loader), len(test_loader))

def double_conv(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )

def crop_img(tensor, target_tensor):
    # center-crop tensor to match target in H/W
    tH = tensor.size(2); tW = tensor.size(3)
    gH = target_tensor.size(2); gW = target_tensor.size(3)
    deltaH = (tH - gH) // 2
    deltaW = (tW - gW) // 2
    return tensor[:, :, deltaH:tH-deltaH, deltaW:tW-deltaW]

class UNet(nn.Module):
    def __init__(self, in_ch=3, n_classes=7, retain_dim=True):
        super().__init__()
        self.pool = nn.MaxPool2d(2,2)
        self.down1 = double_conv(in_ch, 64)
        self.down2 = double_conv(64,128)
        self.down3 = double_conv(128,256)
        self.down4 = double_conv(256,512)
        self.down5 = double_conv(512,1024)

        self.uptran1 = nn.ConvTranspose2d(1024,512,2,2)
        self.upconv1 = double_conv(1024,512)
        self.uptran2 = nn.ConvTranspose2d(512,256,2,2)
        self.upconv2 = double_conv(512,256)
        self.uptran3 = nn.ConvTranspose2d(256,128,2,2)
        self.upconv3 = double_conv(256,128)
        self.uptran4 = nn.ConvTranspose2d(128,64,2,2)
        self.upconv4 = double_conv(128,64)

        self.out = nn.Conv2d(64, n_classes, kernel_size=1)
        self.retain_dim = retain_dim

    def forward(self, x, out_size=(INPUT_SIZE, INPUT_SIZE)):
        x1 = self.down1(x)
        x2 = self.pool(x1)
        x3 = self.down2(x2)
        x4 = self.pool(x3)
        x5 = self.down3(x4)
        x6 = self.pool(x5)
        x7 = self.down4(x6)
        x8 = self.pool(x7)
        x9 = self.down5(x8)

        x = self.uptran1(x9)
        y = crop_img(x7, x)
        x = self.upconv1(torch.cat([x, y], dim=1))

        x = self.uptran2(x)
        y = crop_img(x5, x)
        x = self.upconv2(torch.cat([x, y], dim=1))

        x = self.uptran3(x)
        y = crop_img(x3, x)
        x = self.upconv3(torch.cat([x, y], dim=1))

        x = self.uptran4(x)
        y = crop_img(x1, x)
        x = self.upconv4(torch.cat([x, y], dim=1))

        x = self.out(x)
        if self.retain_dim:
            x = F.interpolate(x, size=out_size, mode='bilinear', align_corners=False)
        x = F.softmax(x, dim=1)
        return x

model = UNet(in_ch=3, n_classes=N_CLASSES).to(DEVICE)
print("Model params:", sum(p.numel() for p in model.parameters()) )

def pixel_accuracy(output, mask):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        correct = (pred == mask).float().sum()
        total = mask.numel()
        return (correct/total).item()

def mIoU(output, mask, n_classes=N_CLASSES, eps=1e-10):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1).view(-1)
        mask = mask.view(-1)
        ious = []
        for c in range(n_classes):
            pred_c = pred == c
            mask_c = mask == c
            if mask_c.sum().item() == 0:
                continue
            inter = (pred_c & mask_c).sum().float().item()
            union = (pred_c | mask_c).sum().float().item()
            ious.append((inter + eps) / (union + eps))
        if len(ious) == 0:
            return 0.0
        return float(np.mean(ious))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
steps_per_epoch = max(1, len(train_loader))
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, epochs=NUM_EPOCHS, steps_per_epoch=steps_per_epoch)

def get_lr(opt):
    return opt.param_groups[0]['lr']

def fit(epochs, model, train_loader, val_loader, criterion, optimizer, scheduler):
    history = {'train_loss':[], 'val_loss':[], 'train_miou':[], 'val_miou':[], 'train_acc':[], 'val_acc':[], 'lrs':[]}
    model.to(DEVICE)
    t0 = time.time()
    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0; running_iou = 0.0; running_acc = 0.0; n_batches = 0
        for imgs, masks in tqdm(train_loader, desc=f"Train E{epoch}/{epochs}"):
            imgs = imgs.to(DEVICE); masks = masks.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            scheduler.step()
            history['lrs'].append(get_lr(optimizer))

            running_loss += loss.item()
            running_iou += mIoU(outputs, masks)
            running_acc += pixel_accuracy(outputs, masks)
            n_batches += 1

        train_loss = running_loss / max(1, n_batches)
        train_miou = running_iou / max(1, n_batches)
        train_acc = running_acc / max(1, n_batches)

        # validation
        model.eval()
        val_loss = 0.0; val_iou = 0.0; val_acc = 0.0; v_batches = 0
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Val E{epoch}/{epochs}"):
                imgs = imgs.to(DEVICE); masks = masks.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                val_iou += mIoU(outputs, masks)
                val_acc += pixel_accuracy(outputs, masks)
                v_batches += 1

        val_loss = val_loss / max(1, v_batches)
        val_miou = val_iou / max(1, v_batches)
        val_acc = val_acc / max(1, v_batches)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_miou'].append(train_miou)
        history['val_miou'].append(val_miou)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch}/{epochs}  TrainLoss:{train_loss:.4f} ValLoss:{val_loss:.4f} Train mIoU:{train_miou:.4f} Val mIoU:{val_miou:.4f} TrainAcc:{train_acc:.4f} ValAcc:{val_acc:.4f} Time:{(time.time()-t0)/60:.2f}m")

    return history

history = fit(NUM_EPOCHS, model, train_loader, val_loader, criterion, optimizer, scheduler)

# save model (optional)
torch.save(model.state_dict(), "unet_demo.pt")

def plot_loss(h):
    plt.figure(figsize=(8,4))
    plt.plot(h['train_loss'], '-o', label='train')
    plt.plot(h['val_loss'], '-o', label='val')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss'); plt.legend(); plt.grid()
    plt.show()

def plot_miou(h):
    plt.figure(figsize=(8,4))
    plt.plot(h['train_miou'], '-o', label='train_mIoU')
    plt.plot(h['val_miou'], '-o', label='val_mIoU')
    plt.xlabel('Epoch'); plt.ylabel('mIoU'); plt.title('mIoU'); plt.legend(); plt.grid()
    plt.show()

def plot_acc(h):
    plt.figure(figsize=(8,4))
    plt.plot(h['train_acc'], '-o', label='train_acc')
    plt.plot(h['val_acc'], '-o', label='val_acc')
    plt.xlabel('Epoch'); plt.ylabel('Pixel Accuracy'); plt.title('Pixel Accuracy'); plt.legend(); plt.grid()
    plt.show()

plot_loss(history); plot_miou(history); plot_acc(history)

model.eval()
def predict_and_show(model, dataset, idx):
    img, mask = dataset[idx]
    img_t = to_tensor_norm(Image.fromarray(np.array(img))) if not isinstance(img, torch.Tensor) else img
    # if dataset returns tensor already normalized, use it directly
    if isinstance(img, torch.Tensor):
        inp = img.unsqueeze(0).to(DEVICE)
    else:
        inp = img_t.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(inp)
        pred = torch.argmax(out, dim=1).cpu().squeeze(0).numpy()
    # visualize (convert original img to numpy for plotting)
    if isinstance(img, torch.Tensor):
        # reverse normalization for display
        t = img.cpu().permute(1,2,0).numpy()
        t = (t * np.array(std) + np.array(mean))
        t = np.clip(t, 0, 1)
        img_disp = (t * 255).astype(np.uint8)
    else:
        img_disp = np.array(img)

    fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(15,5))
    ax1.imshow(img_disp); ax1.set_title("Image"); ax1.axis('off')
    ax2.imshow(mask.cpu().numpy()); ax2.set_title("Ground truth"); ax2.axis('off')
    ax3.imshow(pred); ax3.set_title("Predicted"); ax3.axis('off')
    plt.show()

# show 3 random test samples (or fewer if small test set)
n_show = min(3, len(test_set))
for i in np.random.choice(len(test_set), n_show, replace=False):
    predict_and_show(model, test_set, i)

def evaluate_set(model, ds):
    model.eval()
    miou_scores=[]; acc_scores=[]
    with torch.no_grad():
        for i in tqdm(range(len(ds)), desc="Evaluating"):
            img, mask = ds[i]
            if isinstance(img, torch.Tensor):
                inp = img.unsqueeze(0).to(DEVICE)
            else:
                inp = to_tensor_norm(Image.fromarray(np.array(img))).unsqueeze(0).to(DEVICE)
            out = model(inp)
            miou_scores.append(mIoU(out, mask.to(DEVICE)))
            acc_scores.append(pixel_accuracy(out, mask.to(DEVICE)))
    return np.mean(miou_scores), np.mean(acc_scores)

test_miou, test_acc = evaluate_set(model, test_set)
print("Test mIoU:", test_miou, "Test Pixel Accuracy:", test_acc)
