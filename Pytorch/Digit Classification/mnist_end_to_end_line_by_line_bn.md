বুঝেছি 🙂 — তুমি যেমন **training loop**–এর জন্য লাইনে–লাইনে, খুঁটিনাটি ব্যাখ্যা চেয়েছিলে, **একই স্টাইলে পুরো স্ক্রিপ্টের সব ব্লক** (১–৮) সাজিয়ে দিলাম।
এটা কপি করে GitHub-এ রাখতে পারো।

> **GitHub ফাইল নাম (সাজেস্টেড):** `docs/mnist_end_to_end_line_by_line_bn.md`

---

# 1) Imports — লাইনে–লাইনে

```
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
```

* `matplotlib.pyplot as plt` → রেজাল্ট/ইমেজ প্লট করতে।
* `torch` → টেনসর, অটো-গ্রাড, ডিভাইস ম্যানেজমেন্ট।
* `torch.nn as nn` → লেয়ার/মডেল/লস ক্লাস।
* `torch.optim as optim` → Adam/SGD ইত্যাদি অপ্টিমাইজার।
* `DataLoader` → ব্যাচে ডেটা লোড।
* `datasets`, `transforms` → রেডিমেড ডেটাসেট + প্রি-প্রসেসিং।

---

# 2) Data Preprocessing — লাইনে–লাইনে

```
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
```

* `Compose([...])` → কয়েকটা ট্রান্সফর্ম ক্রমানুসারে চালায়।
* `ToTensor()` → PIL→Tensor, পিক্সেল `[0,1]` স্কেলে।
* `Normalize(mean,std)` → `(x-mean)/std`; **MNIST mean=0.1307, std=0.3081**।

```
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
```

* `root='./data'` → এখানে ডেটা থাকবে।
* `train=True/False` → ট্রেন/টেস্ট স্প্লিট।
* `download=True` → না থাকলে নামায়।
* `transform` → উপরের প্রি-প্রসেসিং অ্যাপ্লাই।

```
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=1000, shuffle=False)
```

* `batch_size=64` (train) → স্থির গ্রেডিয়েন্ট + ভালো স্পিড।
* `shuffle=True` (train) → জেনারালাইজেশন ভালো।
* `batch_size=1000` (test) → ইভ্যালুয়েশন দ্রুত; `shuffle=False` রাখে।

> **টিউনিং টিপস:** বড় GPU হলে train `batch_size` বাড়াতে পারো; Windows/Notebook এ `num_workers=0/2`, Linux/Colab এ `num_workers=2–8`, CUDA হলে `pin_memory=True`।

---

# 3) Model — লাইনে–লাইনে

```
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)  # 10 classes for digits 0–9
```

* `nn.Module` সাবক্লাস → কাস্টম মডেল।
* `Flatten()` → `[N,1,28,28]` → `[N,784]`।
* `Linear(784,128)` → প্রথম FC; 28×28=784 **হার্ড রুল: Flatten করা ফিচার = `in_features`**।
* `ReLU()` → নন-লিনিয়ারিটি; দ্রুত/স্টেবল।
* `Linear(128,64)`, `Linear(64,10)` → ধাপে ধাপে ফিচার কম্প্রেস → ১০ ক্লাস logits।

```
    def forward(self, x):
        x = self.flatten(x)        # [N,784]
        x = self.relu(self.fc1(x)) # [N,128]
        x = self.relu(self.fc2(x)) # [N,64]
        x = self.fc3(x)            # [N,10] (logits)
        return x  # CrossEntropyLoss নিজেই softmax নেবে
```

* **logits** রিটার্ন; আলাদা Softmax দেবে না (CrossEntropyLoss নিজেরা নেয়)।

> **ভ্যারিয়েশন:** ইনপুট সাইজ বদলালে `784` বদলাতে হবে, নইলে `LazyLinear`/CNN+GlobalPool নাও।

---

# 4) Device Setup — লাইনে–লাইনে

```
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

* GPU থাকলে `"cuda"`, নইলে `"cpu"`।
* **Rule:** model, data—একই device–এ থাকতে হবে (না হলে RuntimeError)।

---

# 5) Model, Loss, Optimizer — লাইনে–লাইনে

```
model = DigitClassifier().to(device)
```

* মডেল ডিভাইসে (GPU/CPU) পাঠালাম।

```
criterion = nn.CrossEntropyLoss()
```

* মাল্টি-ক্লাস ক্লাসিফিকেশন; **input=logits `[N,C]`**, **target=int `[N]`**।
* আলাদা softmax **দেবে না**।

```
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

* Adam: দ্রুত কনভার্জ; `lr=1e-3` ভালো স্টার্ট।
* বড় `lr` → অস্থির/NaN; ছোট `lr` → স্লো।

> **প্রো টিপস:** `AdamW` + `weight_decay=1e-4` আধুনিক বেস্ট-প্র্যাকটিস; LR scheduler (Cosine/ReduceLROnPlateau) ব্যবহার করলে টিউনিং সহজ হয়।

---

# 6) Training Function — লাইনে–লাইনে

```
def train(model, device, train_loader, optimizer, criterion, epochs=5):
    model.train()
```

* **train mode**: Dropout on, BN ব্যাচ-স্ট্যাটস আপডেট।

```
    for epoch in range(epochs):
        running_loss = 0.0
```

* ইপক লুপ + লস কাউন্টার।

```
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
```

* ব্যাচে ডেটা/লেবেল; device–এ পাঠালাম। (মিসম্যাচ=RuntimeError)

```
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
```

* **ট্রেনিং ৩ ধাপ**: zero\_grad → backward → step
* `outputs` logits `[N,10]`; `targets` int `[N]`।

```
            running_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
```

* প্রতি ১০০ ব্যাচে প্রগ্রেস লগ (ডেটা ছোট হলে 20/50 করা যায়)।

```
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")
```

* ইপকের গড় ট্রেন লস—কমতে থাকা ভালো সংকেত।

> **রিকমেন্ডেড অ্যাড-অনস:** AMP (mixed precision), gradient clipping, LR scheduler, validation loop, early stopping।

---

# 7) Testing Function — লাইনে–লাইনে

```
def test(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
```

* eval mode: Dropout off, BN স্থির।
* accuracy কাউন্টার।

```
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
```

* `no_grad` → গ্রেডিয়েন্ট ক্যালকুলেশন বন্ধ (মেমরি/সময় সাশ্রয়)।
* `argmax(dim=1)` → predicted class।
* `total/correct` আপডেট।

```
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy
```

* % অ্যাকুরেসি রিটার্ন/প্রিন্ট।

---

# 8) Prediction Visualization — লাইনে–লাইনে

```
def visualize_predictions(model, device, test_loader, n=6):
    model.eval()
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)
```

* টেস্ট থেকে প্রথম ব্যাচ; device–এ পাঠানো।
* `n` ছবির ভিজুয়াল (ব্যাচে `n`–এর কম হলে আগে `n=min(n, images.size(0))` করো)।

```
    outputs = model(images)
    _, preds = torch.max(outputs, 1)
```

* logits→preds (ক্লাস ইনডেক্স)।

```
    images = images.cpu()
    labels = labels.cpu()
    preds = preds.cpu()
```

* matplotlib CPU টেনসর লাগে, তাই CPU-তে আনা।

```
    plt.figure(figsize=(12, 4))
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f"True: {labels[i]}\nPred: {preds[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
```

* 1×n গ্রিডে ছবি; `squeeze()` → \[1,28,28]→\[28,28]।
* `cmap='gray'` → গ্রেস্কেল; RGB হলে বাদ দাও।
* শিরোনামে সত্য লেবেল ও প্রেডিকশন।

---

# 9) End-to-End রান—লাইনে–লাইনে

```
train(model, device, train_loader, optimizer, criterion, epochs=5)
```

* ৫ ইপক ট্রেনিং; **epochs বাড়ালে** বেশি শেখে (ওভারফিটিং হলে ভ্যালিডেশন/আর্লি-স্টপিং দরকার)।

```
test(model, device, test_loader)
```

* টেস্ট সেটে ফাইনাল একিউরেসি।

```
visualize_predictions(model, device, test_loader)
```

* কিছু স্যাম্পলের প্রেডিকশন vs সত্য লেবেল দেখায় (ডিবাগ/বোঝার জন্য দারুণ)।

---    


---

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


