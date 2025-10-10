

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

