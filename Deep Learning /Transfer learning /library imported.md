

```python
import os, math, time, copy, random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
```

এর **প্রত্যেকটা লাইনের কাজ** বুঝিয়ে দিই।
চলো ধাপে ধাপে খুব সহজ ভাষায় দেখি 👇

---

## 🧩 ১️⃣ Python Built-in Modules

### 🔹 `import os`

➡️ অপারেটিং সিস্টেম (Operating System)-এর সাথে কাজ করার জন্য।
**ব্যবহার:**

* ফাইল বা ফোল্ডার তৈরি/মুছে ফেলা (`os.mkdir`, `os.remove`)
* পাথ হ্যান্ডেল করা (`os.path.join`)
* পরিবেশ ভ্যারিয়েবল (environment variables) নেওয়া

🧠 উদাহরণ:

```python
os.makedirs("models", exist_ok=True)
```

👉 “models” নামের ফোল্ডার না থাকলে বানিয়ে দেয়।

---

### 🔹 `import math`

➡️ গাণিতিক (mathematical) ফাংশন ব্যবহারের জন্য।
**ব্যবহার:** `math.sqrt()`, `math.pi`, `math.exp()`, `math.log()` ইত্যাদি।

🧠 উদাহরণ:

```python
val = math.sqrt(16)  # 4.0
```

---

### 🔹 `import time`

➡️ টাইম সম্পর্কিত কাজের জন্য (training duration, sleep, timing ইত্যাদি)।
**ব্যবহার:**

* কোড কত সময় নিচ্ছে সেটা মাপা
* epoch time record করা

🧠 উদাহরণ:

```python
start = time.time()
# training loop
print("Time:", time.time() - start)
```

---

### 🔹 `import copy`

➡️ ডিপ কপি করার জন্য (model weight বা object copy করার সময়)।
**ব্যবহার:** যখন তুমি একটি model state dictionary কপি করো যাতে পরে আবার best model restore করা যায়।

🧠 উদাহরণ:

```python
best_model_wts = copy.deepcopy(model.state_dict())
```

👉 এটি model-এর “best weights” কপি করে রাখে, যাতে পরবর্তীতে সেই ভালো model পুনরুদ্ধার করা যায়।

---

### 🔹 `import random`

➡️ র‍্যান্ডম সংখ্যা তৈরির জন্য, reproducibility বজায় রাখতে `random.seed()` ব্যবহার করা হয়।
**ব্যবহার:** Data shuffle, augmentation randomness, reproducible result।

🧠 উদাহরণ:

```python
random.seed(42)
```

---

## ⚙️ ২️⃣ PyTorch Core Modules

### 🔹 `import torch`

➡️ মূল PyTorch লাইব্রেরি — tensor তৈরি, GPU computation, deep learning সবকিছু এখান থেকেই শুরু।
**ব্যবহার:**

* `torch.Tensor()` তৈরি করা
* GPU ট্রান্সফার (`tensor.to(device)`)
* `torch.cuda` দিয়ে GPU ব্যবস্থাপনা

🧠 উদাহরণ:

```python
x = torch.rand(2, 3)
```

---

### 🔹 `import torch.nn as nn`

➡️ **`nn` = Neural Network** মডিউল
এখানে neural network layer (যেমন Conv2d, Linear, ReLU, Dropout) ও model architecture define করা হয়।

🧠 উদাহরণ:

```python
layer = nn.Conv2d(3, 64, kernel_size=3, padding=1)
```

---

### 🔹 `import torch.optim as optim`

➡️ Optimizer (SGD, Adam, AdamW ইত্যাদি) ব্যবহার করার জন্য।
এগুলো weight আপডেট করে gradient অনুযায়ী।

🧠 উদাহরণ:

```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

---

### 🔹 `from torch.optim.lr_scheduler import CosineAnnealingLR`

➡️ **Learning rate scheduler** — ট্রেনিং চলার সময় learning rate ধীরে ধীরে পরিবর্তন করে দেয় (cosine curve অনুযায়ী)।
এইভাবে model আরও ভালোভাবে converge করে।

🧠 উদাহরণ:

```python
scheduler = CosineAnnealingLR(optimizer, T_max=20)
```

👉 প্রতিটি epoch শেষে learning rate ধীরে ধীরে কমে যাবে।

---

### 🔹 `from torch.utils.data import DataLoader`

➡️ Dataset থেকে batch তৈরি করে model-এ feed করার জন্য DataLoader ব্যবহার হয়।
এটি batch, shuffle, এবং multi-threaded loading হ্যান্ডেল করে।

🧠 উদাহরণ:

```python
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
```

---

## 🖼️ ৩️⃣ TorchVision: Image Tools

### 🔹 `from torchvision import datasets, transforms, models`

এটা PyTorch-এর sub-library — **computer vision-এর জন্য বিশেষভাবে তৈরি।**

**ব্যবহার:**

| Submodule    | কাজ                                                                  |
| ------------ | -------------------------------------------------------------------- |
| `datasets`   | জনপ্রিয় image dataset লোড করা যেমন CIFAR10, ImageNet                |
| `transforms` | Image preprocessing ও augmentation (resize, flip, normalize ইত্যাদি) |
| `models`     | Pretrained model architecture (VGG, ResNet, MobileNet, ইত্যাদি)      |

🧠 উদাহরণ:

```python
train_data = datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor())
model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
```

---

## 🧠 সারসংক্ষেপ

| Module                   | কাজ                                             |
| ------------------------ | ----------------------------------------------- |
| `os`                     | ফাইল ও ফোল্ডার পরিচালনা                         |
| `math`                   | গাণিতিক হিসাব                                   |
| `time`                   | সময় মাপা                                       |
| `copy`                   | ডিপ কপি (model weights সংরক্ষণ)                 |
| `random`                 | র‍্যান্ডম সংখ্যা, reproducibility               |
| `torch`                  | PyTorch মূল লাইব্রেরি (Tensor ops, GPU support) |
| `torch.nn`               | Neural Network লেয়ার ও মডেল আর্কিটেকচার        |
| `torch.optim`            | Optimizer (SGD, Adam ইত্যাদি)                   |
| `CosineAnnealingLR`      | Learning rate schedule নিয়ন্ত্রণ               |
| `DataLoader`             | Dataset থেকে batch তৈরি ও shuffle               |
| `torchvision.datasets`   | CIFAR10, ImageNet ইত্যাদি dataset               |
| `torchvision.transforms` | Image transform ও augmentation                  |
| `torchvision.models`     | Pretrained CNN model যেমন VGG, ResNet           |

---

---

# 🧭 end-to-end training pipeline (ascii flow)

```
Disk (CIFAR-10)
      │
      ▼
 torchvision.datasets (CIFAR10)
      │
      ▼
 torchvision.transforms  ──► (augment + normalize + resize)
      │
      ▼
 torch.utils.data.DataLoader ──► (batch, shuffle, workers, pin_memory)
      │
      ▼
  Model (torchvision.models.vgg16 or your nn.Module)
      │
      ├──► Criterion (nn.CrossEntropyLoss / label smoothing)
      │
      ├──► Optimizer (optim.AdamW / SGD)
      │
      └──► LR Scheduler (CosineAnnealingLR / StepLR)
      │
      ▼
  Train loop (forward → loss → backward → step → scheduler.step)
      │
      ▼
  Validation + Checkpoint (copy.deepcopy best weights)
```

---

# 🔩 python built-ins

## `os` — filesystem utilities

```python
import os
os.makedirs("checkpoints", exist_ok=True)
p = os.path.join("logs", "run1.txt")
os.environ.get("CUDA_VISIBLE_DEVICES", "0")
```

* `makedirs(path, exist_ok=True)` create folders safely
* `path.join(a,b,...)` portable paths
* `environ` read env vars

## `math` — math helpers

```python
import math
r = math.sqrt(49)     # 7.0
pi = math.pi          # 3.14159...
```

## `time` — timing

```python
import time
t0 = time.time()
# ... code ...
print(f"elapsed: {time.time()-t0:.2f}s")
time.sleep(0.1)
```

## `copy` — deep copies (for best-model snapshots)

```python
import copy
best_wts = copy.deepcopy(model.state_dict())
# later
model.load_state_dict(best_wts)
```

## `random` — reproducibility

```python
import random
random.seed(42)
```

---

# ⚙️ pytorch core

## `torch` — tensors, device

```python
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(16, 3, 224, 224, device=device)
```

* `torch.randn/zeros/ones` tensor creation
* `.to(device)` move to GPU/CPU

## `torch.nn as nn` — layers & models

```python
import torch.nn as nn

class TinyNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*112*112, num_classes)
        )
    def forward(self, x): return self.classifier(self.features(x))
```

**common layers**

* `nn.Conv2d(in_ch, out_ch, kernel_size, stride=1, padding=0, bias=True)`
* `nn.ReLU(inplace=True)` / `nn.LeakyReLU(0.01)`
* `nn.MaxPool2d(kernel_size, stride=None, padding=0)`
* `nn.BatchNorm2d(num_features)`
* `nn.Dropout(p=0.5)`
* `nn.Linear(in_features, out_features, bias=True)`
* `nn.Flatten(start_dim=1)`

## `torch.optim as optim` — optimizers

```python
import torch.optim as optim
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-4)
# others: SGD(..., momentum=0.9, nesterov=True), Adam(...), RMSprop(...)
```

**key params**

* `lr`: learning rate
* `weight_decay`: L2 regularization
* `momentum` (SGD), `betas` (Adam/AdamW)

## `CosineAnnealingLR` — LR scheduler

```python
from torch.optim.lr_scheduler import CosineAnnealingLR
scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
```

* `T_max`: epochs (or steps) to finish one cosine cycle
* `eta_min`: minimum LR at the end

## `DataLoader` — batching

```python
from torch.utils.data import DataLoader
train_loader = DataLoader(
    train_set, batch_size=128, shuffle=True,
    num_workers=4, pin_memory=True, drop_last=False
)
```

* `batch_size`: samples per batch
* `shuffle`: randomize order (train=True)
* `num_workers`: loader subprocesses (try 2–8)
* `pin_memory`: faster host→GPU copies

---

# 🖼️ torchvision (vision toolbox)

## `datasets` — ready datasets (CIFAR-10)

```python
from torchvision import datasets
train_set = datasets.CIFAR10(root="./data", train=True,  download=True, transform=train_tfms)
test_set  = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tfms)
```

**common args**

* `root`: local folder
* `train`: train/test split
* `download`: auto-download
* `transform`: preprocessing pipeline

## `transforms` — preprocessing & augmentation

```python
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

train_tfms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

test_tfms = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
```

**popular transforms**

* `Resize((h,w) | short_side)`
* `CenterCrop(size)` / `RandomCrop(size, padding=...)`
* `RandomHorizontalFlip(p=0.5)`
* `ColorJitter(brightness, contrast, saturation, hue)`
* `RandomRotation(degrees)`
* `ToTensor()` → HWC[0..255] → CHW[0..1]
* `Normalize(mean, std)`

## `models` — pretrained CNNs (VGG16)

```python
from torchvision import models

vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
# replace final FC for CIFAR-10
in_feats = vgg.classifier[6].in_features
vgg.classifier[6] = nn.Linear(in_feats, 10)
vgg = vgg.to(device)
```

**common bits**

* `weights=None` (random) or a specific enum for pretrained
* swap classifier head for your `num_classes`
* freeze / unfreeze layers via `p.requires_grad`

---

# 🔁 minimal train/val loop (drop-in)

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = optim.AdamW(vgg.parameters(), lr=3e-4, weight_decay=5e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=20)
scaler = torch.cuda.amp.GradScaler()

def run_epoch(loader, train=True):
    model = vgg.train() if train else vgg.eval()
    tot_loss = tot_acc = n = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if train: optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(), torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)
        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        pred = logits.argmax(1)
        bs = x.size(0)
        tot_loss += loss.item()*bs
        tot_acc  += (pred==y).float().sum().item()
        n += bs
    return tot_loss/n, tot_acc/n

for epoch in range(1, 21):
    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    va_loss, va_acc = run_epoch(test_loader,  train=False)
    scheduler.step()
    print(f"[{epoch:02d}] train {tr_loss:.4f}/{tr_acc:.3f} | val {va_loss:.4f}/{va_acc:.3f}")
```

---

# 🧪 quick tips (for strong accuracy on CIFAR-10 with VGG16)

* **Resize→224 & ImageNet Normalize** for pretrained VGG.
* Start with **head-only training**, then unfreeze **Block-5** to fine-tune.
* Use **AdamW (lr 3e-4, wd 5e-4)** + **Cosine LR**.
* **Augment** (RandomCrop + Flip), add **label smoothing**.
* Enable **AMP** (`torch.cuda.amp`) and **pin_memory=True** for speed.



