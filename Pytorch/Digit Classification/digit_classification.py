import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics  # Import torchmetrics for accuracy calculation
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

EPOCH = 1

# --------------------------
# 1. Data Preprocessing
# --------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean/std
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=1000, shuffle=False)

# --------------------------
# 2. Device Setup
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# 3. CNN Model (for MNIST 1x28x28)
# --------------------------
model = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, padding=1),  # [B, 32, 28, 28]
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),                            # [B, 32, 14, 14]
    
    nn.Conv2d(32, 64, kernel_size=3, padding=1), # [B, 64, 14, 14]
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),                            # [B, 64, 7, 7]
    
    nn.Flatten(),                               # [B, 64*7*7]=[B, 3136]
    nn.Linear(64 * 7 * 7, 128),
    nn.ReLU(inplace=True),
    nn.Linear(128, 10)                          # logits
).to(device)

# --------------------------
# 4. Loss and Optimizer
# --------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --------------------------
# 5. Accuracy Metric (using torchmetrics)
# --------------------------
accuracy_metric = torchmetrics.classification.Accuracy(num_classes=10, task='multiclass').to(device)

# --------------------------
# 6. Training Loop
# --------------------------
for epoch in range(EPOCH):
    model.train()
    running_loss = 0.0
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Loss calculation
        running_loss += loss.item()

        # Update accuracy metric
        accuracy_metric.update(outputs, targets)

        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch+1}/5], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(train_loader)
    accuracy = accuracy_metric.compute()  # Get accuracy from torchmetrics
    print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}, Accuracy: {accuracy.item() * 100:.2f}%")

    # Reset accuracy after each epoch
    accuracy_metric.reset()

# --------------------------
# 7. Testing Loop (with torchmetrics)
# --------------------------
model.eval()
accuracy_metric.reset()  # Reset accuracy metric before testing
correct, total = 0, 0
with torch.no_grad():
    for data, targets in test_loader:
        data, targets = data.to(device), targets.to(device)
        outputs = model(data)
        
        # Update accuracy metric
        accuracy_metric.update(outputs, targets)

accuracy = accuracy_metric.compute()  # Get accuracy
print(f"Test Accuracy: {accuracy.item() * 100:.2f}%")

# --------------------------
# 8. Prediction Visualization
# --------------------------
model.eval()
images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)

with torch.no_grad():
    outputs = model(images)
    _, preds = torch.max(outputs, 1)

images = images.cpu()
labels = labels.cpu()
preds = preds.cpu()

n = 6
plt.figure(figsize=(2*n, 3))
for i in range(n):
    plt.subplot(1, n, i+1)
    plt.imshow(images[i].squeeze(), cmap='gray')
    plt.title(f"True: {labels[i].item()}\nPred: {preds[i].item()}")
    plt.axis('off')
plt.tight_layout()
plt.show()
