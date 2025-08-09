
```python
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# --------------------------
# 1. Data Preprocessing
# --------------------------
# Normalize MNIST dataset (mean and std)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Download and load datasets
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# --------------------------
# 2. Neural Network Model
# --------------------------
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)  # 10 classes for digits 0–9

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x  # raw logits (CrossEntropyLoss will handle softmax)

# --------------------------
# 3. Device Setup
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# 4. Model, Loss, Optimizer
# --------------------------
model = DigitClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --------------------------
# 5. Training Function
# --------------------------
def train(model, device, train_loader, optimizer, criterion, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")

# --------------------------
# 6. Testing Function
# --------------------------
def test(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

# --------------------------
# 7. Prediction Visualization
# --------------------------
def visualize_predictions(model, device, test_loader, n=6):
    model.eval()
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)

    outputs = model(images)
    _, preds = torch.max(outputs, 1)

    images = images.cpu()
    labels = labels.cpu()
    preds = preds.cpu()

    plt.figure(figsize=(12, 4))
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f"True: {labels[i]}\nPred: {preds[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# --------------------------
# 8. Run the Training, Test, and Visualization
# --------------------------
train(model, device, train_loader, optimizer, criterion, epochs=5)
test(model, device, test_loader)
visualize_predictions(model, device, test_loader)
```







## 1) Imports (কী কেন)

1. `matplotlib.pyplot as plt` — প্লট/ভিজুয়ালাইজেশন
2. `torch` — টেনসর, অটো-গ্র্যাড, ডিভাইস
3. `torch.nn as nn` — লেয়ার/মডেল/লস ক্লাস
4. `torch.optim as optim` — অপ্টিমাইজার (Adam/SGD/…)
5. `DataLoader` — ব্যাচে ডেটা লোড
6. `datasets, transforms` — রেডিমেড ডেটাসেট ও প্রি-প্রসেসিং

---

## 2) Data Processing

1. **Transforms (MNIST)**

   * `ToTensor()` → ইমেজ → টেনসর \[0,1]
   * `Normalize((0.1307,), (0.3081,))` → স্ট্যান্ডার্ডাইজ
2. **Dataset**

   * `datasets.MNIST(..., train=True/False, download=True, transform=transform)`
3. **DataLoader**

   * Train: `batch_size=64`, `shuffle=True`
   * Test: `batch_size=1000`, `shuffle=False`
   * (ঐচ্ছিক) `num_workers` (Linux 2–8), `pin_memory=True` (CUDA)

---

## 3) Model Architecture (MLP for MNIST)

1. ইনপুট: \[N, 1, 28, 28] → `Flatten()` → \[N, **784**]
2. লেয়ার: `Linear(784,128)` → `ReLU` → `Linear(128,64)` → `ReLU` → `Linear(64,10)`
3. আউটপুট: **logits** \[N,10] (Softmax লস ফাংশনই হ্যান্ডেল করে)

---

## 4) Device Setup

1. `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
2. **রুল:** model এবং data—**একই device** এ থাকতে হবে

---

## 5) Model, Loss, Optimizer

1. `model = DigitClassifier().to(device)`
2. `criterion = nn.CrossEntropyLoss()` — মাল্টি-ক্লাস; logits ইনপুট
3. `optimizer = optim.Adam(model.parameters(), lr=1e-3)`

   * (প্রো) AdamW + `weight_decay=1e-4` ব্যবহার্য

---

## 6) Training Loop — ধাপে ধাপে (সবসময় এভাবেই যাবে)

1. **Mode set:** `model.train()`
2. **Batch loop:** `for data, targets in train_loader:`
3. **Device move:** `data, targets = data.to(device), targets.to(device)`
4. **Zero grads:** `optimizer.zero_grad()`
5. **Forward:** `outputs = model(data)`
6. **Loss:** `loss = criterion(outputs, targets)`
7. **Backward:** `loss.backward()`
8. **(Optional) Clip:** `torch.nn.utils.clip_grad_norm_(...)`
9. **Update:** `optimizer.step()`
10. **Log:** প্রতি `log_interval` ব্যাচে লস প্রিন্ট
11. **Epoch end:** `avg_train_loss = running_loss / len(train_loader)`
12. **(Recommended) Validation:** eval+no\_grad, val loss/acc
13. **(Optional) Scheduler:** `scheduler.step(val_loss)` / `scheduler.step()`
14. \*\*(Optional) Early stopping / Checkpoint save\`

---

## 7) Evaluation (Validation/Test) Loop — ধাপে ধাপে

1. **Mode set:** `model.eval()`
2. **No grad:** `with torch.no_grad():`
3. **Batch loop:** `for data, targets in data_loader:`
4. **Device move:** `data, targets = data.to(device), targets.to(device)`
5. **Forward:** `outputs = model(data)`
6. **Loss (optional but good):** `loss = criterion(outputs, targets)` → `total_loss += loss.item()`
7. **Predictions:** `preds = outputs.argmax(dim=1)`
8. **Metrics:** `correct += (preds == targets).sum().item()`; `total += targets.size(0)`
9. **Aggregate:** `avg_loss = total_loss / len(data_loader)`; `accuracy = correct / total`
10. **Return/Log:** `avg_loss, accuracy`

---

## 8) Prediction Visualization — ধাপে ধাপে

1. **Mode set:** `model.eval()`
2. **Get a batch:** `images, labels = next(iter(test_loader))`
3. **Device move:** `images, labels = images.to(device), labels.to(device)`
4. **Forward:** `outputs = model(images)`
5. **Predictions:** `preds = outputs.argmax(dim=1)`
6. **CPU for plotting:** `images, labels, preds = images.cpu(), labels.cpu(), preds.cpu()`
7. **Figure:** `plt.figure(figsize=(2*n, 3))`
8. **Loop 0..n-1:** `subplot → imshow(images[i].squeeze(), cmap='gray') → title(True/Pred) → axis off`
9. **Neat layout:** `plt.tight_layout(); plt.show()`

   * (টিপ) `n = min(n, images.size(0))` দিয়ে আউট-অফ-রেঞ্জ এড়াও
   * RGB হলে `permute(1,2,0)`, `cmap` বাদ

---

## 9) End-to-End রান অর্ডার

1. **Transforms সেট** (train/test)
2. **Dataset লোড** (`download=True`)
3. **DataLoader** (train: shuffle=True; test: shuffle=False)
4. **Device সিলেক্ট**
5. **Model → device**, **Loss**, **Optimizer**
6. **Train (N epochs)** → avg train loss
7. **Test/Validation** → avg loss, accuracy
8. **Visualization** → কিছু প্রেডিকশন প্লট
9. **(Optional)** Scheduler / Early stopping / Checkpoint

---

## 10) টিউনিং চিটশিট

* **batch\_size** ↑ → দ্রুত/স্ট্যাবল, RAM ↑; ↓ → ধীর, কখনো জেনারালাইজ ↑
* **lr** ↑ → ফাস্ট/অস্থির; ↓ → স্লো/প্লেটো → Scheduler ইউজফুল
* **epochs** ↑ → শেখা ↑/ওভারফিট ↑ → Early stopping সহায়ক
* **optimizer** → AdamW (স্টার্ট), বড় ভিশনে SGD+momentum + cosine/step LR
* **regularization** → weight decay, dropout, data augmentation
* **perf** → Linux এ `num_workers` বাড়াও; CUDA হলে `pin_memory=True`; AMP ব্যবহার করো

---

এটাই **সিরিয়াল-ওয়াইজ ফুল সামারি**—Imports → Data → Model → Device → Loss/Opt → Train → Eval → Viz → Run → Tuning।


