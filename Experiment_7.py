import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),          # convert to tensor
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=64, shuffle=False)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)   # output: 8 feature maps
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)  # output: 16 feature maps

        self.fc1 = nn.Linear(16 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))     # conv1 feature maps
        x = self.pool(x)
        x = torch.relu(self.conv2(x))     # conv2 feature maps
        x = self.pool(x)
        x = x.view(-1, 16*5*5)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_losses = []
train_acc = []

for epoch in range(5):   # 5 epochs (enough for MNIST)
    total, correct = 0, 0
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_losses.append(running_loss / len(train_loader))
    train_acc.append(100 * correct / total)

    print(f"Epoch {epoch+1} | Loss: {train_losses[-1]:.4f} | Acc: {train_acc[-1]:.2f}%")

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(train_losses)
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(1,2,2)
plt.plot(train_acc)
plt.title("Training Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")

plt.show()

# get one image
images, labels = next(iter(test_loader))
img = images[0].unsqueeze(0).to(device)

model.eval()
with torch.no_grad():
    fmap1 = torch.relu(model.conv1(img))
    fmap1_pool = model.pool(fmap1)

    fmap2 = torch.relu(model.conv2(fmap1_pool))

plt.figure(figsize=(12, 4))
for i in range(8):  # conv1 has 8 feature maps
    plt.subplot(1, 8, i + 1)
    plt.imshow(fmap1[0, i].cpu(), cmap='gray')
    plt.axis('off')
plt.suptitle("Feature Maps from Convolution Layer 1")
plt.show()

plt.figure(figsize=(12, 6))
for i in range(16):  
    plt.subplot(4, 4, i + 1)
    plt.imshow(fmap2[0, i].cpu(), cmap='gray')
    plt.axis('off')
plt.suptitle("Feature Maps from Convolution Layer 2")
plt.show()
