
---

# A) `DigitCNN` — ডিটেইলস ডকুমেন্টেশন

## ইনপুট ধারণা

* ডেটা: **MNIST**, গ্রে-স্কেল `1×28×28`
* টেনসর শেপ: `[Batch, Channel, Height, Width] = [B, 1, 28, 28]`

## ফিচার এক্সট্র্যাক্টর

```python
self.features = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, padding=1),  # [B,32,28,28]
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),                             # [B,32,14,14]

    nn.Conv2d(32, 64, kernel_size=3, padding=1), # [B,64,14,14]
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),                             # [B,64,7,7]
)
```

### 1) `Conv2d(1, 32, 3, padding=1)`

* `in_channels=1`: MNIST গ্রে-স্কেল।
* `out_channels=32`: ৩২টা ফিল্টার → ৩২ ফিচারম্যাপ।
* `kernel_size=3`: 3×3 লোকাল প্যাটার্ন।
* `padding=1`: আউটপুট সাইজ same: 28→28।
* **আউট শেপ:** `[B, 32, 28, 28]`

### 2) `ReLU(inplace=True)`

* নন-লিনিয়ারিটি যোগ করে; নেতিবাচক → 0।
* `inplace=True`: কম মেমরি ব্যবহার।

### 3) `MaxPool2d(2)`

* 2×2 উইন্ডোতে ম্যাক্স; হাইট-উইডথ অর্ধেক।
* 28→14; **আউট শেপ:** `[B, 32, 14, 14]`

### 4) `Conv2d(32, 64, 3, padding=1)`

* আগের আউটপুট চ্যানেল (32) = এই লেয়ারের `in_channels`।
* `out_channels=64`: ক্ষমতা (capacity) বাড়ে।
* **আউট শেপ:** `[B, 64, 14, 14]`

### 5) `ReLU(inplace=True)` → আগের মতোই।

### 6) `MaxPool2d(2)`

* 14→7; **আউট শেপ:** `[B, 64, 7, 7]`

> সারাংশ: `28×28 → 14×14 → 7×7`, চ্যানেল: `1 → 32 → 64`

## ক্লাসিফায়ার

```python
self.classifier = nn.Sequential(
    nn.Flatten(),                                # [B, 64*7*7]=[B,3136]
    nn.Linear(64*7*7, 128),
    nn.ReLU(inplace=True),
    nn.Linear(128, 10)                           # logits
)
```

### 7) `Flatten()`

* `[B, 64, 7, 7] → [B, 64*7*7] = [B, 3136]`

### 8) `Linear(3136, 128)`

* ফিচার কম্প্রেশন/এগ্রিগেশন; 128-D latent।

### 9) `ReLU(inplace=True)`

### 10) `Linear(128, 10)`

* 10 logits (ডিজিট 0–9)।
* `CrossEntropyLoss` softmax ভিতরেই করে, আলাদা softmax দরকার নেই।

## ফরওয়ার্ড

```python
def forward(self, x):
    x = self.features(x)     # [B,64,7,7]
    x = self.classifier(x)   # [B,10]
    return x                 # logits
```

## প্যারামিটার গণনা (আনুমানিক)

* Conv1: `32*(1*3*3 + 1) = 320`
* Conv2: `64*(32*3*3 + 1) = 18,496`
* FC1: `3136*128 + 128 = 401,664`
* FC2: `128*10 + 10 = 1,290`
  **মোট \~ 421K** (MNIST-এর জন্য লাইট/ফাস্ট/যথেষ্ট)

## কখন কী বদলাবে?

* ইনপুট সাইজ বা পুলিং বদলালে `3136` বদলাবে → `Linear`-এর `in_features` আপডেট করতে হবে।
* RGB হলে প্রথম কনভে `in_channels=3` এবং normalize mean/std বদলাবে।

---

# B) CNN আর্কিটেকচার ডিজাইনার — ধাপে–ধাপে

## Step 1 — ইনপুট বুঝো

* সাইজ: ছোট (28×28/32×32) vs বড় (224×224)
* চ্যানেল: 1 (গ্রে) vs 3 (RGB)
* টাস্ক: ক্লাসিফিকেশন / সেগমেন্টেশন / ডিটেকশন (টাস্কভেদে ডাউনস্যাম্পলিং স্ট্র্যাটেজি আলাদা)

## Step 2 — ডাউনস্যাম্পলিং প্ল্যান করো (resolution schedule)

* ক্লাসিফিকেশনে সাধারণত শেষের দিকে **/16 বা /32** করা হয়।
* Thumb rules:

  * MNIST (28×28) → **২ বার** /2 ⇒ 28→14→7
  * CIFAR-10 (32×32) → **৩ বার** /2 ⇒ 32→16→8→4
  * 224×224 → **৪–৫ বার** /2 ⇒ 224→112→56→28→14→7
* ডাউনস্যাম্পলিং দাও **MaxPool(2)** বা **Conv(stride=2)** দিয়ে।

  * Pool = সহজ/ননলার্নেবল
  * Strided Conv = লার্নেবল (ResNet স্টাইল)

## Step 3 — প্রতিটি স্টেজের ভিতরে ব্লক ডিজাইন

* নরমাল টেমপ্লেট: `Conv(3×3, C) → ReLU → Conv(3×3, C) → ReLU → (Pool/Stride)`
* Stable টেমপ্লেট: `Conv → BatchNorm2d → ReLU` (VGG/ResNet প্র্যাকটিস)
* চ্যানেল প্ল্যান: `C` ধাপে ধাপে বাড়াও (32→64→128→256…)

## Step 4 — হেড/ক্লাসিফায়ার

* ছোট ইনপুটে **Flatten + FC** (যেমন এখানে 3136→128→#classes)
* বড় ইনপুটে **GlobalAveragePooling + Linear(#C\_last → #classes)** → প্যারাম কম, জেনারালাইজ ভালো

## Step 5 — রেগুলারাইজেশন/স্ট্যাবিলিটি

* **BatchNorm2d**: প্রতিটি Conv-এর পরে রাখলে ট্রেনিং স্থিতিশীল।
* **Dropout (0.2–0.5)**: FC-এর আগে/Classifier-এ—ওভারফিট কমে।
* **Weight decay (AdamW)**: সাধারণত `1e-4` ভাল স্টার্ট।
* **Data augmentation**: (MNIST-এ হালকা) `RandomRotation(10)` ইত্যাদি।

## Step 6 — লস/অপ্টিমাইজার/স্কেজুলার

* **Loss**: ক্লাসিফিকেশনে `CrossEntropyLoss`
* **Optimizer**: `AdamW(lr=1e-3, weight_decay=1e-4)` বা `SGD(momentum=0.9, lr=0.1)` (বড় ভিশনে)
* **Scheduler**: `CosineAnnealingLR` / `ReduceLROnPlateau (monitor val_loss)`

## Step 7 — ডিবাগ/শেপ–চেক

* প্রতিটি ডাউনস্যাম্পলিংয়ের পরে H×W কেমন হল, লিখে রাখো।
* `print(x.shape)` টেম্পোরারি দিয়ে ফরওয়ার্ড রান করে মিলিয়ে নাও।

---

# C) রেডিমেড ভ্যারিয়েন্টস

## MNIST (28×28, 1ch) — কম্প্যাক্ট

* Downsample: **২ বার** (28→14→7)
* চ্যানেল: `32→64`
* Head: Flatten(64*7*7)→128→10

## CIFAR-10 (32×32, 3ch) — স্টার্টার

* ইনপুট: 3×32×32
* স্টেজ-১: `Conv(3,32)×2 → Pool` ⇒ 32→16
* স্টেজ-২: `Conv(3,64)×2 → Pool` ⇒ 16→8
* স্টেজ-৩: `Conv(3,128)×2 → Pool` ⇒ 8→4
* Head: `GlobalAvgPool` (4×4→1×1) → `Linear(128→10)`
* BN যোগ করলে: `Conv→BN→ReLU`

## 224×224 (ImageNet স্টাইল) — স্কেলডাউন

* Stem: `Conv(7×7,64,stride=2) → Pool(3,stride=2)` ⇒ 224→112→56
* স্টেজ-১: `C=64` (কয়েকটা ব্লক) → 56
* স্টেজ-২: `C=128, stride=2` → 28
* স্টেজ-৩: `C=256, stride=2` → 14
* স্টেজ-৪: `C=512, stride=2` → 7
* Head: `GlobalAvgPool → Linear(512→#classes)`

---

# D) কখন MaxPool দেবো / দেবো না

* **প্রতি Conv-এর পরে নয়**; সাধারণত **প্রতি স্টেজ শেষে ১বার**।
* ছোট ইনপুটে (28×28/32×32) **২–৩ বারের বেশি** পুলিং নয়।
* বিকল্প: **Conv(stride=2)**—লার্নেবল ডাউনস্যাম্পলিং।

---

# E) সাধারণ সমস্যা ও সমাধান

* **`Linear` in\_features mismatch**: ইনপুট সাইজ/পুলিং বদলালে `Flatten` সাইজ বদলায় → `Linear` আপডেট করো বা `nn.LazyLinear(128)` নাও।
* **ডিভাইস মিসম্যাচ**: model/data একই ডিভাইসে রাখো।
* **Softmax দু’বার**: logits-এ আলাদা softmax দিও না—`CrossEntropyLoss` নিজেই করে।
* **ওভারফিটিং**: Dropout, Weight decay, Augmentation, Early stopping।
* **ট্রেনিং স্লো/মেমরি কম**: AMP (mixed precision), ছোট batch, কম চ্যানেল।

---

# F) পরিচ্ছন্ন ভ্যারিয়েন্ট (BatchNorm + Dropout + LazyLinear)

```python
class DigitCNN_BN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28->14
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14->7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(128),             # 64*7*7 অটো ডিটেক্ট করবে
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
```

---

# G) এক লাইনের চিটশিট

* ছোট ইনপুট: **২× MaxPool** (28→14→7), চ্যানেল 32→64 যথেষ্ট
* `Conv→BN→ReLU` ব্লক স্থিতিশীল
* শেষের দিকে **GlobalAvgPool** দিলে FC প্যারামিটার কমে
* আকার/পুলিং বদলালে `Linear` in\_features আপডেট করো বা `LazyLinear` নাও
* ক্লাসিফিকেশনে **logits** রিটার্ন; `CrossEntropyLoss` use

---


---

## **Simple Steps for CNN Design (MNIST Example)**

1. **ইনপুট চ্যানেল ঠিক করো**

   * গ্রেস্কেল → `in_channels=1`
   * RGB → `in_channels=3`

2. **প্রথম Conv লেয়ার**

   * ছোট kernel (৩×৩) + padding=1
   * আউটপুট ফিল্টার = 32 বা 64 (ছোট ডেটাসেট হলে 32 দিয়ে শুরু)

3. **অ্যাক্টিভেশন**

   * `ReLU(inplace=True)` → নন-লিনিয়ারিটি যোগ

4. **ডাউনস্যাম্পলিং**

   * `MaxPool2d(2)` → সাইজ অর্ধেক (28→14)

5. **দ্বিতীয় Conv লেয়ার**

   * `in_channels` = আগের আউটপুট ফিল্টার
   * ফিল্টার সংখ্যা দ্বিগুণ করা (যেমন 32→64)

6. **আবার ReLU + Pooling**

   * সাইজ আরও অর্ধেক (14→7)

7. **Flatten**

   * Conv আউটপুটকে 1D ভেক্টরে পরিণত করা

8. **Fully Connected (FC) লেয়ার**

   * `Linear(Flatten_size, 128)`
   * আবার `ReLU()`

9. **Output Layer**

   * `Linear(128, num_classes)` → logits

10. **লস ফাংশন + অপ্টিমাইজার**

    * লস: `CrossEntropyLoss()`
    * অপ্টিমাইজার: `Adam` বা `AdamW` (lr=0.001)

11. **ট্রেনিং**

    * `model.train()` → batch loop → forward → loss → backward → step

12. **ইভ্যালুয়েশন**

    * `model.eval()` + `torch.no_grad()` → accuracy চেক

---

---

## **1️⃣ Feature Extractor (ফিচার এক্সট্রাক্টর)**

**উদ্দেশ্য:** ছবির কাঁচা পিক্সেল থেকে গুরুত্বপূর্ণ প্যাটার্ন (যেমন edge, texture) বের করা।

**স্টেপ-বাই-স্টেপ:**

1. **Conv2D Layer 1**

   * `nn.Conv2d(1, 32, kernel_size=3, padding=1)`
   * ইনপুট: 1 চ্যানেল (MNIST গ্রেস্কেল)
   * আউটপুট: 32 ফিল্টার → 32টি ফিচার ম্যাপ
   * কের্নেল: ৩×৩ → লোকাল প্যাটার্ন খোঁজা
   * প্যাডিং=1 → সাইজ একই রাখা (28×28)

2. **Activation (ReLU)**

   * `nn.ReLU(inplace=True)`
   * নন-লিনিয়ারিটি যোগ করে জটিল ফিচার শেখা সম্ভব করে।

3. **Pooling 1**

   * `nn.MaxPool2d(2)` → সাইজ অর্ধেক (28×28 → 14×14)
   * ডেটা কমিয়ে computational cost হ্রাস, ইম্পর্ট্যান্ট ফিচার রেখে দেওয়া।

4. **Conv2D Layer 2**

   * `nn.Conv2d(32, 64, kernel_size=3, padding=1)`
   * ইনপুট: আগের লেয়ারের 32 ফিচার ম্যাপ
   * আউটপুট: 64 ফিচার ম্যাপ (আরও বেশি প্যাটার্ন শেখা)

5. **Activation (ReLU)**

   * আবার নন-লিনিয়ারিটি।

6. **Pooling 2**

   * আবার `nn.MaxPool2d(2)` → সাইজ 14×14 → 7×7
   * এখন প্রতিটি ফিচার ম্যাপ ছোট কিন্তু বেশি "meaningful"।

---

## **2️⃣ Classifier (ক্লাসিফায়ার)**

**উদ্দেশ্য:** ফিচার ম্যাপগুলো থেকে ফাইনাল ক্লাস নির্ধারণ করা।

**স্টেপ-বাই-স্টেপ:**

1. **Flatten**

   * `nn.Flatten()` → \[B, 64, 7, 7] → \[B, 3136]
   * সব পিক্সেল ফিচার এক লাইনে এনে Fully Connected Layer-এ পাঠানো।

2. **Fully Connected Layer 1**

   * `nn.Linear(64*7*7, 128)` → 3136 ফিচার → 128 ফিচার
   * উচ্চমাত্রার তথ্য কম্প্রেস করে আরও useful ফিচার তৈরি করা।

3. **Activation (ReLU)**

   * জটিল ডিসিশন বাউন্ডারি তৈরি।

4. **Fully Connected Layer 2 (Output Layer)**

   * `nn.Linear(128, 10)` → 128 ফিচার → 10 logits
   * প্রতিটি digit (0–9) এর জন্য একটি স্কোর।

---

## **3️⃣ Forward Pass (ফরওয়ার্ড পাস)**

**উদ্দেশ্য:** ইনপুট ডেটা sequential ভাবে ফিচার এক্সট্রাকশন ও ক্লাসিফিকেশনের মধ্য দিয়ে চালানো।

**স্টেপ-বাই-স্টেপ:**

1. ইনপুট ইমেজ `self.features` ব্লকে যাবে → Conv + ReLU + Pool ধারাবাহিকভাবে চলবে।
2. ফিচার ম্যাপ আউটপুট `self.classifier` ব্লকে যাবে → Flatten + Fully Connected Layers চলবে।
3. ফাইনাল আউটপুট হবে logits (probability নয়, raw score)।
4. ট্রেনিং এর সময় `CrossEntropyLoss` এই logits থেকে probability এবং loss হিসাব করবে।

---

## **সারসংক্ষেপে CNN আর্কিটেকচার ডিজাইন স্টেপ:**

1. **ইনপুট শেপ চেক**
2. **Feature Extractor**

   * Conv → Activation → Pool
   * ডেপথ (filters) ধাপে ধাপে বাড়ানো
   * Spatial সাইজ Pooling দিয়ে কমানো
3. **Flatten**
4. **Classifier**

   * FC layers → Activation → Output layer
5. **লস ফাংশন ও অপ্টিমাইজার সেট**
6. **Training Loop & Evaluation Loop**
7. **Visualization (Optional)**

---

---

## 1) মূল কাঠামো (যেকোনো CNN)

1. **Feature extractor (স্টেজভিত্তিক)**

   * প্রতিটি স্টেজে: `Conv → (BN) → ReLU → Conv → (BN) → ReLU → (Downsample)`
   * Downsample দাও `MaxPool(2)` **অথবা** `Conv(stride=2)` দিয়ে
   * চ্যানেল ধাপে ধাপে বাড়াও: 32 → 64 → 128 → 256 …
2. **Classifier**

   * ছোট ইনপুটে: `Flatten → Linear → ReLU → Linear(num_classes)`
   * বড় ইনপুটে: `GlobalAvgPool → Linear(num_classes)` (FC প্যারাম কমে)
3. **Forward**

   * ইনপুট → স্টেজগুলো → ক্লাসিফায়ার → **logits** (CE Loss নিজেই softmax করবে)

---

## 2) সাইজ/ডাউনস্যাম্পলিং (রুল অফ থাম্ব)

* **MNIST (28×28)**: 2× ডাউনস্যাম্পল (28→14→7)
* **CIFAR-10 (32×32)**: 3× ডাউনস্যাম্পল (32→16→8→4)
* **224×224 (ImageNet-like)**: 4–5× ডাউনস্যাম্পল (224→112→56→28→14→7)

> শেষ ফিচারম্যাপ \~ **7×7 বা 4×4** হলে GlobalAvgPool সহজে বসে যায়।

---

## 3) কখন কোন ব্লক/টেকনিক

* **BatchNorm2d**: প্রায় সব Conv-এর পর দিন → ট্রেনিং স্থির, দ্রুত কনভার্জ।
* **Dropout (0.2–0.5)**: FC/ক্লাসিফায়ার অংশে; ওভারফিট হলে দিন।
* **Stride vs MaxPool**: দুটোই ঠিক; ResNet স্টাইলে stride conv জনপ্রিয়।
* **Residual connection**: নেট গভীর হলে (২০+ লেয়ার) add/skip helpful।
* **Depthwise separable conv** (MobileNet): মোবাইল/লো-compute টার্গেট।
* **Kernel size**: ডিফল্ট 3×3; বড় কনটেক্সটে প্রথমে 5×5/7×7 (বা 3×3 স্ট্যাক)।

---

## 4) তিনটা রেডিমেড টেমপ্লেট

### (A) ছোট ইনপুট (MNIST 1×28×28)

```
[Conv3x3,32] → BN → ReLU → MaxPool2
[Conv3x3,64] → BN → ReLU → MaxPool2
Flatten → Linear(64*7*7→128) → ReLU → Linear(128→10)
```

### (B) মাঝারি (CIFAR-10 3×32×32) — GlobalAvgPool সহ

```
[Conv3x3,32]×2 → BN/ReLU → MaxPool2
[Conv3x3,64]×2 → BN/ReLU → MaxPool2
[Conv3x3,128]×2 → BN/ReLU → MaxPool2
GlobalAvgPool → Linear(128→10)
```

### (C) বড় (224×224) — স্টেজভিত্তিক

```
Stem: Conv7x7,64,stride=2 → MaxPool3,stride=2
Stage1: [Conv3x3,64]×2
Stage2: [Conv3x3,128]×2, প্রথম Conv stride=2
Stage3: [Conv3x3,256]×2, প্রথম Conv stride=2
Stage4: [Conv3x3,512]×2, প্রথম Conv stride=2
GlobalAvgPool → Linear(512→num_classes)
```

---

## 5) চেকলিস্ট (ভুল এড়াতে)

* **in\_channels = আগের out\_channels** (shape mismatch এড়াও)
* পুল/স্ট্রাইড বদলালে **Flatten in\_features** রিপ计算/`nn.LazyLinear` ব্যবহার করো
* **logits** ফেরত দাও; `CrossEntropyLoss`-এ আলাদা softmax দিও না
* **ডিভাইস** এক রাখো: `model.to(device)` + `data.to(device)`

---

## 6) টিউনিং শর্টকাট

* Optimizer: **AdamW(lr=1e-3, wd=1e-4)** স্টার্ট; বড় ডেটায় SGD+momentum ও কাজের
* Scheduler: `CosineAnnealingLR` বা `ReduceLROnPlateau(val_loss)`
* Augmentation: ছোটে হালকা (`RandomRotation(10)`), বড়ে শক্ত (`Crop/Flip/ColorJitter`)
* AMP (mixed precision): GPU তে ফ্রি স্পিড-আপ/মেমরি সেভ

---

