
---

# PyTorch Dataset + DataLoader + Transforms — ফুল বাংলা গাইড

> ফাইল নাম (GitHub সাজেস্টেড): **`docs/pytorch_dataset_dataloader_full_bn.md`**

---

## 0) এক নজরে

* **Dataset**: ডেটা কোথায়/কীভাবে আছে (ফাইল/ফোল্ডার/CSV/ওয়েব) — কীভাবে পড়বে তা সংজ্ঞা দেয়
* **Transform**: প্রতিটি স্যাম্পলে প্রি-প্রসেসিং/অগমেন্টেশন
* **DataLoader**: Dataset → batch, shuffle, multi-worker দিয়ে ইটারেট
* **Sampler**: কোন স্যাম্পল কোন ক্রমে/ওজনে আসবে
* **Best practice**: Train → `shuffle=True`, Val/Test → `shuffle=False`, CUDA হলে `pin_memory=True`, বড় ডেটায় `num_workers>0`

---

## 1) Quickstart (MNIST)

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),                       # [0,255] -> [0,1]
    transforms.Normalize((0.1307,), (0.3081,))   # MNIST mean/std
])

train_ds = datasets.MNIST('./data', train=True,  download=True, transform=transform)
test_ds  = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=64,  shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=1000, shuffle=False)
```

**প্যারামিটার কী করলো**

* `root`: ডাউনলোড/রিড লোকেশন
* `train`: True→train split, False→test
* `download`: True হলে না থাকলে ডেটা নামাবে
* `transform`: ইনপুটে প্রি-প্রসেস/অগমেন্টেশন

---

## 2) Transforms — ব্যাখ্যা + কখন ব্যবহার

> সাধারণত অর্ডার: **Resize/Crop → Geo/Color Aug (train only) → ToTensor → Normalize**

```python
from torchvision import transforms
tf_train = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(3/4, 4/3)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),  # ImageNet norm
])
tf_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
])
```

### জিওমেট্রি/সাইজ

* **`Resize(size)`**: সব ইমেজ এক সাইজে। `(H,W)` বা `shorter_side:int`।
* **`CenterCrop(size)`**: কেন্দ্র থেকে নির্দিষ্ট সাইজ; Val/Test-এ কনসিস্টেন্ট।
* **`RandomCrop(size, padding=4)`**: ট্রেনে বৈচিত্র্য; CIFAR ক্লাসিক।
* **`RandomResizedCrop(size, scale, ratio)`**: স্কেল/অ্যাসপেক্ট রেশিও বদলে ক্রপ → ট্রেনে মজবুত।
* **`RandomRotation(degrees)`**: ±ডিগ্রি রোটেশন; লিখিত/সাইন বোর্ডে কম ব্যবহার।
* **`RandomAffine(deg, translate, scale, shear)`**: রোটেট+শিফট+স্কেল+শিয়ার একসাথে।
* **`RandomPerspective(distortion_scale, p)`**: ক্যামেরা অ্যাঙ্গেল বদল সিমুলেট।

### ফ্লিপ

* **`RandomHorizontalFlip(p=0.5)`**: বাম↔ডান। **কাজ**: অরিয়েন্টেশন-রোবাস্ট; **ব্যবহার কোরো না** টেক্সট/সাইন/মেডিক্যালে।
* **`RandomVerticalFlip(p=0.5)`**: উপর↔নিচ; স্যাটেলাইট/মাইক্রোস্কোপে যুক্তিসংগত।

### কালার/লাইট

* **`ColorJitter(b, c, s, h)`**: Brightness/Contrast/Saturation/Hue র‍্যান্ডম বদলায়।

  * উদাহরণ: `ColorJitter(0.2,0.2,0.2,0.1)` → ±20%, hue ±0.1
  * **ব্যবহার কোরো না** যখন রঙই ক্লাস নির্ধারণ করে (মেডিক্যাল স্টেইনিং, নির্দিষ্ট ফলের রঙ)।
* **`RandomGrayscale(p)`**: p সম্ভাবনায় গ্রেস্কেল; রঙ-নির্ভরত কমে।
* **`GaussianBlur(ksize, sigma)`**: ব্লার; সেলফ-সুপারভাইজড/রবাস্টনেসে কাজে দেয়।

### কনভার্সন/নরমালাইজ

* **`ToTensor()`**: PIL→Tensor, `[0,1]`, shape `[C,H,W]`।
* **`Normalize(mean, std)`**: `(x-mean)/std`; Train/Val/Test—একই নরমালাইজ রাখো।

  * **ImageNet** mean/std: `(0.485,0.456,0.406)` / `(0.229,0.224,0.225)`
  * **MNIST**: `(0.1307,)` / `(0.3081,)`

### কন্ট্রোল ফ্লো

* **`RandomApply([t1, t2], p)`**: p সম্ভাবনায় ব্লক অ্যাপ্লাই
* **`RandomChoice([t1, t2, ...])`**: লিস্ট থেকে একটাকে নেয়
* **AutoAugment ফ্যামিলি**: `AutoAugment`, `RandAugment`, `TrivialAugmentWide` — দ্রুত ভালো বেসলাইন

**সতর্কতা**

* অতিরিক্ত Augmentation → কনভার্জ স্লো/পারফ কমতে পারে; ধীরে ধীরে টিউন করো
* Order ভুল কোরো না: **ToTensor → Normalize** (এটা উল্টো করলে ভুল)

---

## 3) DataLoader — সব প্যারামিটার (কি দিলে কী হয়)

```python
from torch.utils.data import DataLoader
loader = DataLoader(
    dataset,
    batch_size=64,          # ব্যাচে 64 স্যাম্পল
    shuffle=True,           # Train=True, Val/Test=False
    sampler=None,           # দিলে shuffle=False রাখতে হয়
    batch_sampler=None,     # দিলে batch_size/shuffle/sampler ব্যবহার হয় না
    num_workers=4,          # CPU কোর→ 2/4/8
    pin_memory=True,        # CUDA হলে GPU কপি দ্রুত
    drop_last=False,        # True হলে শেষ ছোট ব্যাচ বাদ
    timeout=0,              # ডেটা-পড়া টাইমআউট (সাধারণত 0)
    worker_init_fn=None,    # প্রতিটি worker-এ কাস্টম init
    prefetch_factor=2,      # প্রতি worker আগাম ব্যাচ প্রস্তুত রাখে
    persistent_workers=True,# epoch শেষে worker জীবিত (PyTorch>=1.7)
    collate_fn=None         # কাস্টম ব্যাচ মার্জ (ভ্যারিয়েবল সাইজে দরকার)
)
```

**কখন কোন ভ্যালু**

* **Train**: `shuffle=True`, `num_workers=2–8 (Linux/Colab)`, `pin_memory=True (CUDA)`
* **Val/Test**: `shuffle=False`, বড় `batch_size` (512–2048) দ্রুত ইভ্যালুয়েশন
* **Windows/Jupyter**: `num_workers=0/2` (বেশি দিলে মাঝে মাঝে ইস্যু)
* **BatchNorm/DP/DDP**: `drop_last=True` উপকারী

---

## 4) Dataset ভ্যারাইটি

### (a) FashionMNIST

```python
train_ds = datasets.FashionMNIST('./data', train=True,  download=True, transform=tf_train)
test_ds  = datasets.FashionMNIST('./data', train=False, download=True, transform=tf_val)
```

### (b) CIFAR-10/100

```python
cifar_train = transforms.Compose([
  transforms.RandomCrop(32, padding=4),
  transforms.RandomHorizontalFlip(),
  transforms.ToTensor(),
  transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
])
cifar_test = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
])
train_ds = datasets.CIFAR10('./data', train=True,  download=True, transform=cifar_train)
test_ds  = datasets.CIFAR10('./data', train=False, download=True, transform=cifar_test)
```

### (c) ImageFolder (নিজস্ব ফোল্ডার)

```
data/
  cats/
    c1.jpg ...
  dogs/
    d1.jpg ...
```

```python
from torchvision.datasets import ImageFolder
ds = ImageFolder('data', transform=tf_train)
```

### (d) ImageNet (ম্যানুয়াল ডাউনলোড)

```python
train = datasets.ImageFolder('/path/imagenet/train', transform=tf_train)
val   = datasets.ImageFolder('/path/imagenet/val',   transform=tf_val)
```

### (e) STL10, SVHN, Caltech101/256 (দ্রুত)

```python
stl_train = datasets.STL10('./data', split='train', download=True, transform=tf_train)
svhn_train = datasets.SVHN('./data', split='train', download=True, transform=tf_train)
caltech = datasets.ImageFolder('/path/to/caltech101', transform=tf_train)
```

---

## 5) Train/Val/Test split

```python
from torch.utils.data import random_split
n = len(train_ds); tr = int(0.8*n); va = n - tr
train_sub, val_sub = random_split(train_ds, [tr, va], generator=torch.Generator().manual_seed(42))
```

---

## 6) Class imbalance → WeightedRandomSampler

```python
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

# targets: ক্লাস লেবেলের 1D array/tensor
targets = np.array([y for _, y in train_ds])  # ImageFolder হলে ds.samples থেকেও নেওয়া যায়
class_count = np.bincount(targets)
weights_per_class = 1.0 / class_count
weights = weights_per_class[targets]
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

train_loader = DataLoader(train_ds, batch_size=64, sampler=sampler, num_workers=4, pin_memory=True)
```

**কখন দরকার**: এক ক্লাস অনেক বেশি/কম হলে balanced batch পেতে।

---

## 7) Custom Dataset (CSV/JSON/নিজস্ব লজিক)

```python
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd, os

class MyDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None, target_transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.img_dir, row['filename'])).convert('RGB')
        y = row['label']
        if self.transform:        img = self.transform(img)
        if self.target_transform: y   = self.target_transform(y)
        return img, y
```

---

## 8) Custom `collate_fn` (ভ্যারিয়েবল সাইজ/বিশেষ মার্জ)

```python
import torch
def my_collate(batch):
    imgs, labels = zip(*batch)      # tuple of lists
    # উদাহরণ: সাইজ এক হলে সরাসরি stack
    imgs = torch.stack(imgs, dim=0)
    labels = torch.tensor(labels)
    return imgs, labels

loader = DataLoader(ds, batch_size=16, collate_fn=my_collate)
```

---

## 9) Performance টিপস

* SSD ব্যবহার করো, `num_workers` বাড়াও (Linux/Colab: 2–8)
* CUDA হলে: `pin_memory=True`, টেনসর `.to(device, non_blocking=True)`
* বড় ইমেজে OOM → ছোট `batch_size`, AMP (`torch.cuda.amp.autocast`), Grad Accumulation
* Deterministic (রিপ্রোডিউসিবল):

```python
import torch, random, numpy as np
seed=42
torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
```

---

## 10) ট্রেনিং লুপ (মিনিমাল ডেমো)

```python
import torch, torch.nn as nn, torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 128), nn.ReLU(),
    nn.Linear(128, 10)
).to(device)

opt  = optim.Adam(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()

for epoch in range(5):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad()
        out = model(x)
        loss = crit(out, y)
        loss.backward()
        opt.step()

    # eval
    model.eval(); correct=total=0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            pred = model(x).argmax(1)
            correct += (pred==y).sum().item()
            total   += y.size(0)
    print(f"Epoch {epoch+1}: Test Acc = {correct/total:.4f}")
```

---

## 11) Cheat-sheet (কখন কী নেব)

* **Train Loader**: `shuffle=True`, `num_workers=2–8`, `pin_memory=True`
* **Val/Test Loader**: `shuffle=False`, বড় batch (512–2048)
* **Transforms (Train)**: `RandomCrop/ResizedCrop + RandomHorizontalFlip + (ColorJitter) + ToTensor + Normalize`
* **Transforms (Val/Test)**: `Resize/CenterCrop + ToTensor + Normalize`
* **Imbalance**: `WeightedRandomSampler`
* **Variable-size**: `collate_fn` + pad/stack
* **ImageNet-style**: Train→`RandomResizedCrop+Flip+Jitter`, Val→`Resize+CenterCrop`

---

## 12) Common errors (দ্রুত সমাধান)

* **Transforms order ভুল** → সবসময় `ToTensor` → `Normalize`
* **Train/Val নরমালাইজ mismatch** → একি mean/std দাও
* **Win/Notebook crash** → `num_workers=0/2`, `persistent_workers=False`
* **Accuracy ওঠানামা** → Augmentation বেশি/কম → ধীরে টিউন
* **Slow I/O** → SSD + `num_workers` + `prefetch_factor`

---

## 13) Ready-made Recipes

**CIFAR-10**

```python
train_tf = transforms.Compose([
  transforms.RandomCrop(32, padding=4),
  transforms.RandomHorizontalFlip(),
  transforms.ToTensor(),
  transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
])
val_tf = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
])
```

**ImageNet (224)**

```python
train_tf = transforms.Compose([
  transforms.RandomResizedCrop(224),
  transforms.RandomHorizontalFlip(),
  transforms.ColorJitter(0.2,0.2,0.2,0.1),
  transforms.ToTensor(),
  transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
])
val_tf = transforms.Compose([
  transforms.Resize(256),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
])
```

**Documents/Medical (সংযত)**

```python
train_tf = transforms.Compose([
  transforms.Resize((512,512)),
  transforms.RandomRotation(5),
  transforms.ToTensor(),
  transforms.Normalize(mean, std),
])
val_tf = transforms.Compose([
  transforms.Resize((512,512)),
  transforms.ToTensor(),
  transforms.Normalize(mean, std),
])
```

---

