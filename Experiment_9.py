import os
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.ops import nms
import selectivesearch

SEED = 42
IMAGE_ROOT = "D:\\University\\SEM-V\\Deep Learning\\DL LAB\\archive (1)\\images\\images"
CSV_PATH   = "D:\\University\\SEM-V\\Deep Learning\\DL LAB\\archive (1)\\df.csv"

MAX_IMAGES = 100
MAX_PROPOSALS_PER_IMAGE = 10
INPUT_SIZE = 128
IOU_POSITIVE_THRESHOLD = 0.3

NUM_EPOCHS = 2
LR = 1e-3
NUM_WORKERS = 0     # keep 0 for Windows
BATCH_SIZE = 1      # 1 image per batch → many ROI crops internally
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def show(img, bbs=None, texts=None, figsize=(6,6), ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img[..., ::-1])
    if bbs is not None:
        for i, bb in enumerate(bbs):
            x1,y1,x2,y2 = map(int, bb)
            rect = plt.Rectangle((x1,y1), x2-x1, y2-y1,
                                 fill=False, color='red', linewidth=1.2)
            ax.add_patch(rect)
            if texts is not None:
                ax.text(x1, y1-4, texts[i], fontsize=8, color='yellow')
    ax.axis('off')
    if ax is None:
        plt.show()

def extract_iou(a, b, eps=1e-8):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    areaA = max(0,(a[2]-a[0])*(a[3]-a[1]))
    areaB = max(0,(b[2]-b[0])*(b[3]-b[1]))
    union = areaA + areaB - interArea
    return interArea/(union+eps) if union>0 else 0.0

def extract_candidates(img, scale=200, min_size=50):
    _, regions = selectivesearch.selective_search(img, scale=scale, min_size=min_size)
    h,w,_ = img.shape
    img_area = h*w
    cands = []
    seen = set()
    for r in regions:
        rect = r['rect']  # x,y,w,h
        if rect in seen: continue
        seen.add(rect)
        x,y,w2,h2 = rect
        if w2<=0 or h2<=0: continue
        area = w2*h2
        if area < 0.0005*img_area: continue
        if area > 0.95*img_area: continue
        cands.append([x,y,w2,h2])
    return cands

if not os.path.exists(CSV_PATH):
    raise SystemExit("CSV not found.")
if not os.path.isdir(IMAGE_ROOT):
    raise SystemExit("Images folder not found.")

df = pd.read_csv(CSV_PATH)
uids = df['ImageID'].unique().tolist()

print(f"Found {len(uids)} unique images, using {MAX_IMAGES}.")

FPATHS = []
GTBBS  = []
ROIS   = []
CLSS   = []
DELTAS = []

for img_id in tqdm(uids[:MAX_IMAGES]):
    fpath = os.path.join(IMAGE_ROOT, f"{img_id}.jpg")
    if not os.path.exists(fpath):
        continue

    img = cv2.imread(fpath, cv2.IMREAD_COLOR)[..., ::-1]
    if img is None: continue
    H, W, _ = img.shape

    df_i = df[df['ImageID']==img_id]
    if df_i.shape[0] == 0: continue

    gtbbs = (df_i[['XMin','YMin','XMax','YMax']].values * np.array([W,H,W,H])).astype(int).tolist()
    labels = df_i['LabelName'].tolist()

    cands = extract_candidates(img)
    if len(cands)==0: continue
    cands = cands[:MAX_PROPOSALS_PER_IMAGE*4]  # aggressive trimming

    c_xyxy = np.array([[x,y,x+w,y+h] for x,y,w,h in cands])
    ious = np.array([[extract_iou(c, g) for g in gtbbs] for c in c_xyxy])

    rois_i = []
    labels_i = []
    deltas_i = []

    for ci,cand in enumerate(c_xyxy):
        best = int(np.argmax(ious[ci]))
        best_iou = ious[ci,best]
        gt = gtbbs[best]

        if best_iou > IOU_POSITIVE_THRESHOLD:
            labels_i.append(labels[best])
        else:
            labels_i.append("background")

        dx1 = (gt[0]-cand[0]) / W
        dy1 = (gt[1]-cand[1]) / H
        dx2 = (gt[2]-cand[2]) / W
        dy2 = (gt[3]-cand[3]) / H
        deltas_i.append([dx1,dy1,dx2,dy2])

        rois_i.append(cand/np.array([W,H,W,H]))

    # sample fixed # proposals per image
    if len(rois_i) > MAX_PROPOSALS_PER_IMAGE:
        idxs = random.sample(range(len(rois_i)), MAX_PROPOSALS_PER_IMAGE)
        rois_i   = [rois_i[i] for i in idxs]
        labels_i = [labels_i[i] for i in idxs]
        deltas_i = [deltas_i[i] for i in idxs]

    FPATHS.append(fpath)
    GTBBS.append(gtbbs)
    ROIS.append(rois_i)
    CLSS.append(labels_i)
    DELTAS.append(deltas_i)

print("ROI dataset built:", len(FPATHS), "images")

flat = [l for arr in CLSS for l in arr]
unique = sorted(set(flat))
label2target = {lab:i for i,lab in enumerate(unique)}
target2label = {i:lab for lab,i in label2target.items()}
background_class = label2target['background']

print("Label map:", label2target)

normalize = transforms.Normalize([0.485,0.456,0.406],
                                 [0.229,0.224,0.225])

def preprocess(crop):
    t = torch.tensor(crop).permute(2,0,1).float() / 255.
    return normalize(t)

class RCNNDataset:
    def __init__(self, fpaths, rois, labels, deltas):
        self.fpaths=fpaths; self.rois=rois; self.labels=labels; self.deltas=deltas

    def __len__(self): return len(self.fpaths)

    def __getitem__(self, ix):
        f = self.fpaths[ix]
        img = cv2.imread(f, cv2.IMREAD_COLOR)[..., ::-1]
        H,W,_ = img.shape
        bbs = (np.array(self.rois[ix])*np.array([W,H,W,H])).astype(int).tolist()
        labs = self.labels[ix]
        dels = self.deltas[ix]
        return img,bbs,labs,dels

    def collate_fn(self, batch):
        X=[]; Y=[]; D=[]
        for (img,bbs,labs,dels) in batch:
            for i,(x1,y1,x2,y2) in enumerate(bbs):
                if x2<=x1 or y2<=y1: continue
                crop = img[y1:y2, x1:x2]
                if crop.size==0: continue
                crop = cv2.resize(crop,(INPUT_SIZE,INPUT_SIZE))
                X.append(preprocess(crop).unsqueeze(0))
                Y.append(label2target[labs[i]])
                D.append(dels[i])
        if len(X)==0:
            dummy = torch.zeros((1,3,INPUT_SIZE,INPUT_SIZE),device=device)
            return dummy, torch.tensor([background_class],device=device), torch.zeros((1,4),device=device)
        return torch.cat(X).to(device), torch.LongTensor(Y).to(device), torch.FloatTensor(D).to(device)

idxs=list(range(len(FPATHS)))
random.shuffle(idxs)
cut=int(0.9*len(idxs))
train_idx, val_idx = idxs[:cut], idxs[cut:]

train_ds = RCNNDataset([FPATHS[i] for i in train_idx],
                       [ROIS[i] for i in train_idx],
                       [CLSS[i] for i in train_idx],
                       [DELTAS[i] for i in train_idx])
val_ds   = RCNNDataset([FPATHS[i] for i in val_idx],
                       [ROIS[i] for i in val_idx],
                       [CLSS[i] for i in val_idx],
                       [DELTAS[i] for i in val_idx])

train_loader = DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True,
                          collate_fn=train_ds.collate_fn,num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_ds,batch_size=BATCH_SIZE,shuffle=False,
                          collate_fn=val_ds.collate_fn,num_workers=NUM_WORKERS)

print("Train batches:", len(train_loader), "Val batches:", len(val_loader))

backbone = models.resnet18(pretrained=True)
backbone.fc = nn.Identity()
for p in backbone.parameters(): p.requires_grad=False
backbone.to(device).eval()

class RCNNHead(nn.Module):
    def __init__(self, n_classes=len(label2target)):
        super().__init__()
        self.backbone = backbone
        self.cls = nn.Linear(512, n_classes)
        self.bbox = nn.Sequential(nn.Linear(512,256), nn.ReLU(),
                                  nn.Linear(256,4), nn.Tanh())
        self.ce = nn.CrossEntropyLoss()

    def forward(self,x):
        feat = self.backbone(x)
        return self.cls(feat), self.bbox(feat)

    def loss_fn(self,cls_pred,bbox_pred,labels,deltas):
        cls_loss = self.ce(cls_pred, labels)
        pos = labels!=background_class
        if pos.sum()>0:
            reg_loss = nn.L1Loss()(bbox_pred[pos], deltas[pos])
        else:
            reg_loss = torch.tensor(0.0,device=labels.device)
        return cls_loss + 10*reg_loss, cls_loss.detach(), reg_loss.detach()

rcnn = RCNNHead().to(device)
optimizer = optim.SGD(rcnn.parameters(), lr=LR, momentum=0.9)

train_losses=[]; val_losses=[]
train_accs=[]; val_accs=[]

for ep in range(1, NUM_EPOCHS+1):
    rcnn.train()
    tloss=0; tcorrect=0; tsamp=0

    for X,Y,D in train_loader:
        optimizer.zero_grad()
        c_pred, b_pred = rcnn(X)
        loss,_,_ = rcnn.loss_fn(c_pred,b_pred,Y,D)
        loss.backward()
        optimizer.step()
        tloss += loss.item()*X.size(0)
        pred = torch.argmax(c_pred,dim=1)
        tcorrect += (pred==Y).sum().item()
        tsamp += X.size(0)

    train_losses.append(tloss/tsamp)
    train_accs.append(tcorrect/tsamp)

    # validation
    rcnn.eval()
    vloss=0; vcorrect=0; vsamp=0
    with torch.no_grad():
        for Xv,Yv,Dv in val_loader:
            cp,bp = rcnn(Xv)
            l,_,_ = rcnn.loss_fn(cp,bp,Yv,Dv)
            vloss += l.item()*Xv.size(0)
            predv = torch.argmax(cp,dim=1)
            vcorrect += (predv==Yv).sum().item()
            vsamp += Xv.size(0)

    val_losses.append(vloss/vsamp)
    val_accs.append(vcorrect/vsamp)

    print(f"Epoch {ep}/{NUM_EPOCHS} | TrainLoss: {train_losses[-1]:.4f} Acc: {train_accs[-1]:.3f} | ValLoss: {val_losses[-1]:.4f} Acc: {val_accs[-1]:.3f}")

epochs = range(1, NUM_EPOCHS+1)
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(epochs, train_losses, "-o", label="Train Loss")
plt.plot(epochs, val_losses, "-o", label="Val Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.title("Loss Curve")
plt.subplot(1,2,2)
plt.plot(epochs, train_accs, "-o", label="Train Acc")
plt.plot(epochs, val_accs, "-o", label="Val Acc")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.title("Accuracy Curve")
plt.show()

all_preds=[]; all_true=[]
rcnn.eval()
with torch.no_grad():
    for Xv,Yv,Dv in val_loader:
        cp,bp = rcnn(Xv)
        pv = torch.argmax(cp,dim=1).cpu().numpy()
        tv = Yv.cpu().numpy()
        all_preds.extend(pv.tolist())
        all_true.extend(tv.tolist())

if len(all_true)>0:
    cm = confusion_matrix(all_true, all_preds, labels=list(range(len(target2label))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=[target2label[i] for i in range(len(target2label))])
    plt.figure(figsize=(8,6))
    disp.plot(cmap="Blues", xticks_rotation='vertical', ax=plt.gca())
    plt.title("Confusion Matrix (ROI-level)")
    plt.show()

def test_predictions(fpath):
    img = cv2.imread(fpath,cv2.IMREAD_COLOR)[..., ::-1]
    H,W,_=img.shape
    cands = extract_candidates(img)[:MAX_PROPOSALS_PER_IMAGE*4]
    if len(cands)==0:
        print("No candidates.")
        return

    xyxy = np.array([[x,y,x+w,y+h] for x,y,w,h in cands])
    crops=[]
    for (x1,y1,x2,y2) in xyxy:
        if x2<=x1 or y2<=y1: continue
        crop=img[y1:y2,x1:x2]
        if crop.size==0: continue
        crop=cv2.resize(crop,(INPUT_SIZE,INPUT_SIZE))
        crops.append(preprocess(crop).unsqueeze(0))

    if len(crops)==0:
        print("No valid crops.")
        return

    X = torch.cat(crops).to(device)
    rcnn.eval()
    with torch.no_grad():
        cls,bbox = rcnn(X)
        probs = torch.softmax(cls,dim=1)
        confs, preds = torch.max(probs,dim=1)
        confs=confs.cpu().numpy()
        preds=preds.cpu().numpy()
        bbox=bbox.cpu().numpy()

    keep = preds!=background_class
    if keep.sum()==0:
        print("No detections.")
        show(img)
        return

    xyxy = xyxy[keep]
    preds=preds[keep]
    confs=confs[keep]
    deltas=bbox[keep]

    # denorm bbox
    bbs = xyxy.astype(float) + np.stack([
        deltas[:,0]*W, deltas[:,1]*H, deltas[:,2]*W, deltas[:,3]*H
    ],axis=1)

    boxes = torch.tensor(bbs, dtype=torch.float32)
    scores = torch.tensor(confs, dtype=torch.float32)
    keep_ix = nms(boxes, scores, 0.3).numpy()

    final_boxes=bbs[keep_ix]
    final_scores=confs[keep_ix]
    final_preds=preds[keep_ix]

    texts = [target2label[int(p)] + f" ({s:.2f})"
             for p,s in zip(final_preds, final_scores)]

    show(img, final_boxes.tolist(), texts)

#demo
print("Demo on first validation image:")
test_predictions(val_ds.fpaths[0])
