
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
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean/std
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=1000, shuffle=False)

# --------------------------
# 2. CNN Model (for MNIST 1x28x28)
# --------------------------
class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # [B,32,28,28]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             # [B,32,14,14]

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # [B,64,14,14]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             # [B,64,7,7]
        )
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),                                # [B, 64*7*7]=[B,3136]
            nn.Linear(64*7*7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)                           # logits
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x  # logits (CrossEntropyLoss will handle softmax)

# --------------------------
# 3. Device Setup
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# 4. Model, Loss, Optimizer
# --------------------------
model = DigitCNN().to(device)
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
                print(f"Epoch [{epoch+1}/{epochs}], "
                      f"Batch [{batch_idx}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}")
        
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")

# --------------------------
# 6. Testing Function
# --------------------------
def test(model, device, test_loader):
    model.eval()
    correct, total = 0, 0
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
    images, labels = next(iter(test_loader))
    n = min(n, images.size(0))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

    images = images.cpu()
    labels = labels.cpu()
    preds = preds.cpu()

    plt.figure(figsize=(2*n, 3))
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f"True: {labels[i].item()}\nPred: {preds[i].item()}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# --------------------------
# 8. Run
# --------------------------
train(model, device, train_loader, optimizer, criterion, epochs=5)
test(model, device, test_loader)
visualize_predictions(model, device, test_loader, n=6)
```
চল ধাপে ধাপে পুরো কোডটা বুঝি—**বিশেষ করে `DigitCNN`** ক্লাসটা গভীরভাবে। আমি প্রতিটি লেয়ার কী করছে, আউটপুট শেপ কী হচ্ছে, কেন এমন ডিজাইন, কী বদলালে কী হবে—সব বুঝিয়ে দিচ্ছি।

---

# বড় ছবি: এই কোড কী করছে

1. **MNIST ডেটা** (গ্রে-স্কেল 1×28×28) লোড করে `ToTensor + Normalize`
2. **CNN মডেল (`DigitCNN`)** দিয়ে ট্রেনিং
3. টেস্ট অ্যাকুরেসি মাপে
4. কয়েকটা প্রেডিকশন প্লট করে

---

# `DigitCNN`—লাইন ধরে ব্যাখ্যা (গভীরভাবে)

```python
class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # [B,32,28,28]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             # [B,32,14,14]

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # [B,64,14,14]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             # [B,64,7,7]
        )
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),                                # [B, 64*7*7]=[B,3136]
            nn.Linear(64*7*7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)                           # logits
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x  # logits (CrossEntropyLoss will handle softmax)
```

## ইনপুট শেপ

* **MNIST ইমেজ:** `[batch, channels, height, width] = [B, 1, 28, 28]`
* এখানে `channels=1` কারণ গ্রে-স্কেল। RGB হলে 3।

---

## Feature extractor ব্লক

### 1) `nn.Conv2d(1, 32, kernel_size=3, padding=1)`

* **কাজ:** ইনপুট ইমেজ থেকে 3×3 কনভলিউশন ফিল্টার দিয়ে **৩২টি ফিচার ম্যাপ** বের করা।
* **ইনপুট:** `[B, 1, 28, 28]`
* **প্যারাম:**

  * `in_channels=1` (গ্রে-স্কেল)
  * `out_channels=32` → 32টি কনভ ফিল্টার (শেখা যাবে)
  * `kernel_size=3` → 3×3 উইন্ডো
  * `padding=1` → আউটপুট spatial size **সমান** রাখতে (28 → 28)
* **আউটপুট শেপ:** `[B, 32, 28, 28]`
* **কেন 3×3 + padding=1?**

  * 3×3 ছোট কণা (edges, corners, strokes) ধরতে ভালো
  * padding 1 দিলে বর্ডার ইনফো হারায় না; স্ট্রাইড 1 হলে same size থাকে

### 2) `nn.ReLU(inplace=True)`

* **কাজ:** নন-লিনিয়ারিটি এনে জটিল প্যাটার্ন শিখতে সাহায্য করে; নেগেটিভ→0, পজিটিভ→যেমন আছে।
* `inplace=True` মেমরি সাশ্রয় করে (ফরওয়ার্ডে সামান্য অপটিমাইজেশন)।

### 3) `nn.MaxPool2d(2)`

* **কাজ:** 2×2 উইন্ডোতে **ম্যাক্স পুল**—স্পেশাল ডাউনস্যাম্পলিং করে।
* **আউটপুট শেপ:** `[B, 32, 14, 14]` (28/2 = 14)
* **কেন দরকার?**

  * কম্পিউটেশন কমে, ট্রান্সলেশন ইনভ্যারিয়ান্স কিছুটা আসে, ওভারফিটিং কমে

### 4) `nn.Conv2d(32, 64, kernel_size=3, padding=1)`

* **কাজ:** আগের 32 চ্যানেল থেকে আরো **ডিপ** ফিচার 64 চ্যানেলে এক্সট্রাক্ট করা।
* **ইনপুট:** `[B, 32, 14, 14]`
* **আউটপুট:** `[B, 64, 14, 14]` (padding=1 → same size)
* **ইন্টুইশন:** প্রথম ব্লক লোকাল স্ট্রোক/এজ; দ্বিতীয় ব্লক কম্বাইন্ড প্যাটার্ন (কাঁটা, বাঁক, স্ট্রোকের সংমিশ্রণ)

### 5) `nn.ReLU(inplace=True)`

* আগের মতোই নন-লিনিয়ারিটি।

### 6) `nn.MaxPool2d(2)`

* **আউটপুট:** `[B, 64, 7, 7]` (14/2 = 7)

---

## Classifier ব্লক

### 7) `nn.Flatten()`

* **কাজ:** টেনসর ফ্ল্যাট করে FC-তে দেওয়ার মতো 2D বানায়।
* **ইনপুট:** `[B, 64, 7, 7]`
* **আউটপুট ফিচার সাইজ:** `64 * 7 * 7 = 3136` → শেপ `[B, 3136]`

  * **গণনার নিয়ম:** `channels × height × width`

### 8) `nn.Linear(64*7*7, 128)`

* **কাজ:** কনভ ফিচার থেকে 128-ডাইমেনশন এলাকা—উচ্চ লেভেল কম্প্রেশন।
* **কেন 128?**

  * MNIST ছোট ডেটা; খুব বড় ডেন্স লেয়ার দিলে ওভারফিটিং হতে পারে। 128 যথেষ্ট শক্তিশালী ও লাইট।

### 9) `nn.ReLU(inplace=True)`

* নন-লিনিয়ারিটি।

### 10) `nn.Linear(128, 10)`

* **কাজ:** 10টি ডিজিটের জন্য **logits** আউটপুট।
* CrossEntropyLoss softmax ভিতরেই করবে; এখানে logits-ই যথেষ্ট।

---

## Forward pass

```python
def forward(self, x):
    x = self.features(x)   # কনভ ব্লক → [B,64,7,7]
    x = self.classifier(x) # ফ্ল্যাট+ডেন্স → [B,10]
    return x               # logits
```

---

# শেপ ট্র্যাকিং (এক লাইনে)

`[B,1,28,28]`
→ Conv(3×3,pad1,32) → `[B,32,28,28]`
→ ReLU → (same)
→ Pool(2×2) → `[B,32,14,14]`
→ Conv(3×3,pad1,64) → `[B,64,14,14]`
→ ReLU → (same)
→ Pool(2×2) → `[B,64,7,7]`
→ Flatten → `[B,3136]`
→ Linear(3136→128) → `[B,128]`
→ ReLU → `[B,128]`
→ Linear(128→10) → `[B,10]` (logits)

---

# কেন এই ডিজাইন MNIST-এ কাজ করে?

* **ছোট, স্পার্স স্ট্রোক** (ডিজিটের রেখা) ধরতে **3×3 Conv** খুব কার্যকর
* **২ বার Pooling** করে size 28→14→7 : ছোট, কিন্তু ফিচার-সমৃদ্ধ
* শেষের ডেন্স লেয়ারগুলো **পার-ইমেজ** লেভেলে ডিসিশন নেয়
* **প্যারামিটার সংখ্যা কম** → ওভারফিটিং কম, ট্রেনিং দ্রুত

---

# কী বদলালে কী হবে?

### 1) `kernel_size` 3→5

* Receptive field বাড়ে; বেশি কনটেক্সট দেখবে
* প্যারামিটার ও কম্পিউট খরচ বাড়ে
* MNIST-এ 3 সাধারণত যথেষ্ট

### 2) `padding` বাদ দিলে (`padding=0`)

* স্প্যাটিয়াল সাইজ কমে যাবে (28→26; 14→12 ইত্যাদি)
* শেষের ফ্ল্যাটেন সাইজ বদলে যাবে, `Linear`–এর ইনপুট ডাইমেনশন **আপডেট** করতে হবে

### 3) `MaxPool2d(2)` বাদ দিলে

* স্প্যাটিয়াল সাইজ বড় থাকবে → কম্পিউট/প্যারাম বেশি → overfit ঝুঁকি
* বড় সাইজ থাকলে `GlobalAveragePooling`–ও ব্যবহার করা যায় (CNN আধুনিকে জনপ্রিয়)

### 4) চ্যানেল বাড়ানো (32→64→128)

* ক্ষমতা বাড়ে (জটিল প্যাটার্ন শিখবে)
* ওভারফিটিং ও ট্রেন সময় বাড়ে
* MNIST-এ সাধারণত 32/64 যথেষ্ট

### 5) Dropout/BatchNorm যোগ করা

* **Dropout (0.3–0.5)** → ওভারফিটিং কমে
* **BatchNorm2d** (Conv এর পরে) → ট্রেনিং স্থির/দ্রুত

  * উদাহরণ: `Conv2d → BatchNorm2d → ReLU`

### 6) RGB/অন্যান্য সাইজে অ্যাডাপ্ট করা

* RGB হলে **প্রথম Conv `in_channels=3`**
* ইনপুট সাইজ 32×32 হলে Pooling শেষে সাইজ বদলাবে (32→16→8), ফলে `Linear`–এর ইনপুট **64*8*8=4096** হবে—লাইনারে সেই অনুযায়ী আপডেট করো।

---

# মোট প্যারামিটার (আইডিয়া)

* Conv2d: `out_ch * (in_ch * k*k + bias)`

  * 1st conv: `32*(1*3*3 + 1)` = 32\*(9+1)=320
  * 2nd conv: `64*(32*3*3 + 1)` = 64\*(288+1)=64\*289=18496
* FC1: `3136*128 + 128` = 401,536 + 128 = 401,664
* FC2: `128*10 + 10` = 1,280 + 10 = 1,290
* **মোট \~421k** (মোটামুটি MNIST-এর জন্য লাইটওয়েট)

---

# ট্রেনিং টিপস (এই মডেলের সাথে)

* **AdamW(lr=1e-3, wd=1e-4)** দিয়ে শুরু করো
* **LR scheduler** (ReduceLROnPlateau বা CosineAnnealingLR) যোগ করলে টিউনিং সহজ
* **Early stopping**: val loss না কমলে থামো
* CUDA থাকলে **AMP** (`autocast + GradScaler`) ইউজ করে স্পিড/মেমরি বাঁচাও

---

# পুরো স্ক্রিপ্টের বাকি অংশ—সংক্ষিপ্ত কী করছে

* **Data Preprocessing:** `ToTensor + Normalize`
* **DataLoader:** train(batch=64, shuffle=True), test(batch=1000, shuffle=False)
* **Device:** `"cuda"` যদি থাকে, না হলে `"cpu"`
* **Loss:** `CrossEntropyLoss` (logits ইনপুট, softmax ভিতরে)
* **Train loop:** প্রতি ব্যাচে zero\_grad → forward → loss → backward → step, ইপক শেষে avg loss
* **Test loop:** eval + no\_grad, accuracy গণনা
* **Visualization:** টেস্ট থেকে `n` ইমেজ তুলে True vs Pred প্লট

---

# চাইলে কিছু “প্রো” টাচ

* Conv ব্লকে: `Conv → BatchNorm2d → ReLU → MaxPool`
* Classifier-এ: `Dropout(0.3)`
* Data augmentation (MNIST–এ হালকা): `RandomRotation(10)`


