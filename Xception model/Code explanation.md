
---

### **1. `import os, glob`**

#### **`os` মডিউল**

* **কাজ:**
  অপারেটিং সিস্টেমের ফাইল সিস্টেম (folders, files) নিয়ে কাজ করা।
* **ভিতরে কীভাবে কাজ করে:**
  Python-এর `os` মডিউল **C লেভেলের OS API** ব্যবহার করে সিস্টেম-ডিরেক্টরি, ফাইল পাথ, ফাইল হ্যান্ডলিং করে।
* **ব্যবহার:**

  * ফাইল পাথ তৈরি → `os.path.join()`
  * ডিরেক্টরি লিস্ট → `os.listdir()`
  * ডিরেক্টরি চেক → `os.path.exists()`
* **উদাহরণ:**

  ```python
  folder = "images"
  file_name = "photo.jpg"
  path = os.path.join(folder, file_name)
  print(path)  # images/photo.jpg
  ```

---

#### **`glob` মডিউল**

* **কাজ:**
  নির্দিষ্ট **pattern** অনুযায়ী ফাইল/ফোল্ডারের নাম খুঁজে বের করা।
* **ভিতরে কীভাবে কাজ করে:**
  ফাইল সিস্টেম স্ক্যান করে pattern (`*.jpg`) এর সাথে মিল খুঁজে লিস্ট আকারে রিটার্ন দেয়।
* **ব্যবহার:**

  * সব `.jpg` ফাইল লিস্ট করা
  * নির্দিষ্ট নাম দিয়ে ফাইল খোঁজা
* **উদাহরণ:**

  ```python
  jpg_files = glob.glob("images/*.jpg")
  print(jpg_files)
  ```

---

### **2. `import torch, torch.nn as nn`**

#### **`torch` লাইব্রেরি**

* **কাজ:**
  ডিপ লার্নিং-এর জন্য টেনসর (Tensor) অপারেশন এবং অটোমেটিক গ্রেডিয়েন্ট ক্যালকুলেশন করা।
* **ভিতরে কীভাবে কাজ করে:**

  * **Tensor** হলো মাল্টি-ডাইমেনশনাল ডেটা অ্যারে, যা NumPy array-এর মতো কিন্তু GPU-তে চলতে পারে।
  * **Autograd Engine** ব্যাকওয়ার্ড প্রোপাগেশনের জন্য স্বয়ংক্রিয়ভাবে ডেরিভেটিভ ক্যালকুলেট করে।
* **ব্যবহার:**

  * ডেটা স্টোর করা → `torch.tensor()`
  * GPU তে পাঠানো → `.to('cuda')`
  * মডেল ট্রেইনিং → ফরওয়ার্ড + ব্যাকওয়ার্ড পাস
* **উদাহরণ:**

  ```python
  x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
  y = x ** 2
  y.sum().backward()
  print(x.grad)
  ```

---

#### **`torch.nn` (এখানে `nn`)**

* **কাজ:**
  নিউরাল নেটওয়ার্ক তৈরি করার জন্য লেয়ার, অ্যাক্টিভেশন, লস ফাংশন ইত্যাদি সরবরাহ করা।
* **ভিতরে কীভাবে কাজ করে:**
  প্রতিটি লেয়ার `nn.Module` থেকে ইনহেরিট করে এবং ফরওয়ার্ড পাস ডিফাইন করে।
* **ব্যবহার:**

  * Dense Layer → `nn.Linear()`
  * Convolution → `nn.Conv2d()`
  * লস → `nn.CrossEntropyLoss()`
* **উদাহরণ:**

  ```python
  layer = nn.Linear(10, 5)  # ইনপুট: 10 ফিচার, আউটপুট: 5 ফিচার
  ```

---

### **3. `from torch.utils.data import Dataset, DataLoader`**

#### **`Dataset`**

* **কাজ:**
  কাস্টম ডেটাসেট তৈরি করতে সাহায্য করে।
* **ভিতরে কীভাবে কাজ করে:**

  * `__len__()` → ডেটার সাইজ রিটার্ন করে
  * `__getitem__()` → ইনডেক্স দিয়ে নির্দিষ্ট ডেটা রিটার্ন করে
* **ব্যবহার:**

  * লোকাল ইমেজ ডেটা লোড করা
  * টেক্সট ফাইল পড়া
* **উদাহরণ:**

  ```python
  class MyDataset(Dataset):
      def __init__(self, files):
          self.files = files
      def __len__(self):
          return len(self.files)
      def __getitem__(self, idx):
          return self.files[idx]
  ```

---

#### **`DataLoader`**

* **কাজ:**
  Dataset থেকে **batch আকারে** ডেটা আনে।
* **ভিতরে কীভাবে কাজ করে:**

  * মাল্টিথ্রেডিং/মাল্টিপ্রসেসিং ব্যবহার করে দ্রুত ডেটা আনে
  * ব্যাচ সাইজ ও শাফল নিয়ন্ত্রণ করে
* **উদাহরণ:**

  ```python
  loader = DataLoader(MyDataset([1,2,3,4]), batch_size=2, shuffle=True)
  ```

---

### **4. `from torchvision import transforms`**

* **কাজ:**
  ইমেজ ডেটা ট্রান্সফর্ম (resize, crop, normalize) করা।
* **ভিতরে কীভাবে কাজ করে:**

  * PIL Image বা Tensor ইনপুট নিয়ে প্রসেস করে
  * চেইন আকারে একাধিক ট্রান্সফর্ম একসাথে প্রয়োগ করে
* **ব্যবহার:**

  ```python
  transform = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.5], std=[0.5])
  ])
  ```

---

### **5. `from PIL import Image`**

* **কাজ:**
  ছবি লোড, এডিট, সেভ করা।
* **ভিতরে কীভাবে কাজ করে:**
  ইমেজ ফাইলকে ডিকোড করে RAM-এ লোড করে এবং Python অবজেক্ট আকারে রিটার্ন দেয়।
* **ব্যবহার:**

  ```python
  img = Image.open("photo.jpg")
  img.show()
  ```

---

### **6. `import timm`**

* **কাজ:**
  প্রি-ট্রেইনড ইমেজ মডেল (ResNet, EfficientNet, Vision Transformer) ব্যবহার করার জন্য লাইব্রেরি।
* **ভিতরে কীভাবে কাজ করে:**
  HuggingFace-এর মতো API দিয়ে মডেল ডাউনলোড ও লোড করে।
* **ব্যবহার:**

  ```python
  model = timm.create_model("resnet50", pretrained=True)
  ```

---

### **7. `from sklearn.metrics import accuracy_score`**

* **কাজ:**
  মডেলের প্রেডিকশন কতটুকু সঠিক হয়েছে, তা হিসাব করে।
* **ভিতরে কীভাবে কাজ করে:**
  `y_true` ও `y_pred` মিলিয়ে সঠিক প্রেডিকশনের সংখ্যা / মোট সংখ্যা বের করে।
* **ব্যবহার:**

  ```python
  acc = accuracy_score([1,0,1], [1,1,1])
  print(acc)  # 0.666...
  ```

---


---

# UADFVDataset Class – Full Detailed Documentation

## ১. Import এবং Configuration

```python
DATA_DIR = "/kaggle/input/uadfv-dataset/UADFV"
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

* **DATA\_DIR**: মূল dataset-এর path। এখানে `fake` ও `real` ফোল্ডার রয়েছে।
* **BATCH\_SIZE**: একবারে কতগুলো image মডেল train/validation এ ব্যবহার করবে।
* **NUM\_EPOCHS**: dataset মডেলের মাধ্যমে কতবার যাবে।
* **LEARNING\_RATE**: optimizer-এর learning rate।
* **DEVICE**: GPU/CPU নির্বাচন। CUDA থাকলে GPU, না থাকলে CPU।

---

## ২. Dataset Class Declaration

```python
class UADFVDataset(Dataset):
```

* PyTorch-এর `Dataset` base class extend করে একটি **custom dataset** তৈরি করা হচ্ছে।
* এটি DataLoader-এর জন্য প্রয়োজনীয় interface দেয়: `__len__` ও `__getitem__`।

---

## ৩. Initialization Function (`__init__`)

```python
def __init__(self, root_dir, transform=None):
    self.transform = transform
    self.image_paths = []
    self.labels = []
```

* **root\_dir**: dataset-এর মূল folder।
* **transform**: optional image augmentation বা normalization function।
* **image\_paths**: dataset-এর সব image path এখানে সংরক্ষিত হবে।
* **labels**: প্রতিটি image-এর label (0=real, 1=fake) এখানে রাখা হবে।

---

### ৩.১. Fake Images Load

```python
fake_dirs = glob.glob(os.path.join(root_dir, "fake", "frames", "*"))
for folder in fake_dirs:
    imgs = glob.glob(os.path.join(folder, "*.png"))
    for img_path in imgs:
        self.image_paths.append(img_path)
        self.labels.append(1)
```

**Step-by-Step Explanation:**

1. **`glob.glob(os.path.join(root_dir, "fake", "frames", "*"))`**

   * `"*"` মানে `"frames"` folder-এর ভিতরের সব **sub-folder** বা file path আনো।
   * এই dataset structure অনুযায়ী `"frames"` folder-এর ভিতরে প্রতিটি video আলাদা folder।
   * ফলাফল: সব video-folder path list হবে।

2. **`for folder in fake_dirs:`**

   * প্রতিটি video-folder loop করে process করা হয়।

3. **`imgs = glob.glob(os.path.join(folder, "*.png"))`**

   * প্রতিটি video-folder-এর সব `.png` ফাইল বের করা হয়।

4. **`self.image_paths.append(img_path)`**

   * প্রতিটি frame-এর path list-এ সংরক্ষণ করা হয়।

5. **`self.labels.append(1)`**

   * fake images-এর জন্য label 1।

**উদাহরণ:**

```
fake/frames/video1/frame001.png -> label 1
fake/frames/video1/frame002.png -> label 1
...
```

---

### ৩.২. Real Images Load

```python
real_dirs = glob.glob(os.path.join(root_dir, "real", "frames", "*"))
for folder in real_dirs:
    imgs = glob.glob(os.path.join(folder, "*.png"))
    for img_path in imgs:
        self.image_paths.append(img_path)
        self.labels.append(0)
```

* Fake images-এর মতোই কাজ করে।
* শুধু **label 0** ব্যবহার করা হয়।

**উদাহরণ:**

```
real/frames/video3/frame001.png -> label 0
real/frames/video3/frame002.png -> label 0
...
```

---

## ৪. `__len__` Function

```python
def __len__(self):
    return len(self.image_paths)
```

**Explanation:**

* PyTorch Dataset এর জন্য `__len__` অবশ্যই থাকা দরকার।
* Dataset-এর মোট number of images return করে।
* DataLoader এই length জানে, যাতে iteration ঠিকঠাক হয়।

---

## ৫. `__getitem__` Function

```python
def __getitem__(self, idx):
    img_path = self.image_paths[idx]
    label = self.labels[idx]
    image = Image.open(img_path).convert("RGB")
    if self.transform:
        image = self.transform(image)
    return image, label
```

**Step-by-Step Explanation:**

1. **`img_path = self.image_paths[idx]`**

   * Index অনুযায়ী image path নেওয়া হয়।

2. **`label = self.labels[idx]`**

   * একই index অনুযায়ী label নেওয়া হয়।

3. **`image = Image.open(img_path).convert("RGB")`**

   * PIL ব্যবহার করে image read করা হয়।
   * `.convert("RGB")` ensures 3-channel color (even if input grayscale)।

4. **`if self.transform:`**

   * যদি transform pass করা থাকে, apply করা হয়।
   * যেমন: resize, normalize, augment।

5. **`return image, label`**

   * Image tensor এবং corresponding label return করে।
   * DataLoader এটি batch-wise load করতে পারে।

---

## ৬. Folder Structure Visualization

```
UADFV/
 ├─ fake/
 │   └─ frames/
 │       ├─ video1/
 │       │   ├─ frame001.png
 │       │   └─ frame002.png
 │       └─ video2/
 │           ├─ frame001.png
 │           └─ frame002.png
 └─ real/
     └─ frames/
         ├─ video3/
         │   ├─ frame001.png
         │   └─ frame002.png
         └─ video4/
             ├─ frame001.png
             └─ frame002.png
```

* **`glob("*")`** দিয়ে video-folder select করা হয়।
* এরপর **`*.png`** দিয়ে সব frame load করা হয়।
* **image\_paths** এবং **labels** list sequentially update হয়।

---

## ৭. Summary

| Function/Attribute | Purpose                                               |
| ------------------ | ----------------------------------------------------- |
| `__init__`         | Dataset initialize, image paths ও labels collect করা। |
| `glob("*")`        | Folder এবং file path list করতে।                       |
| `self.image_paths` | সব image path সংরক্ষণ।                                |
| `self.labels`      | Corresponding labels সংরক্ষণ।                         |
| `__len__`          | Dataset-এর total images return।                       |
| `__getitem__`      | Index অনুযায়ী image এবং label return।                |
| `transform`        | Optional image augmentation বা normalization।         |

**Important Notes:**

* Correct order: `image_paths` এবং `labels` index-matched থাকতে হবে।
* Boolean masks/transform পরে ব্যবহার করা যায়।
* DataLoader batch-wise processing করতে সহজ হয়।

---

```python
real_dirs = glob.glob(os.path.join(root_dir, "real", "frames", "*"))
for folder in real_dirs:
    imgs = glob.glob(os.path.join(folder, "*.png"))
    for img_path in imgs:
        self.image_paths.append(img_path)
        self.labels.append(0)
```

---

## **Details Explanation (Step-by-step in Bangla)**

### **1. এই লাইন:**

```python
real_dirs = glob.glob(os.path.join(root_dir, "real", "frames", "*"))
```

* `os.path.join(root_dir, "real", "frames", "*")`

  * মেইন `root_dir` এর সাথে `"real"`, `"frames"`, `"*"` জুড়ে সম্পূর্ণ একটি পাথ বানাচ্ছে।
  * `"*"` → ওয়াইল্ডকার্ড চিহ্ন, মানে এখানে `frames` ফোল্ডারের ভেতরে যতগুলো সাব-ফোল্ডার আছে, সবগুলো সিলেক্ট হবে।
  * উদাহরণ:

    ```
    /dataset/real/frames/video1
    /dataset/real/frames/video2
    /dataset/real/frames/video3
    ```

* `glob.glob(...)` → এই প্যাটার্নের সাথে মিলে যাওয়া সব পাথকে একটি **list** আকারে রিটার্ন করবে।

  * `real_dirs` এখন এই রকম কিছু থাকবে:

    ```python
    [
        "/dataset/real/frames/video1",
        "/dataset/real/frames/video2",
        "/dataset/real/frames/video3"
    ]
    ```

---

### **2. এই লুপ:**

```python
for folder in real_dirs:
```

* এখানে `folder` একবারে একটি ভিডিও ফোল্ডারের পাথ ধরে।
* উদাহরণ: প্রথম ইটারেশনে `folder = "/dataset/real/frames/video1"`

---

### **3. এই লাইন:**

```python
imgs = glob.glob(os.path.join(folder, "*.png"))
```

* এখানে প্রতিটি ভিডিও ফোল্ডারের ভেতরে `.png` এক্সটেনশনওয়ালা সব ইমেজ ফাইল খুঁজে বের করা হচ্ছে।
* উদাহরণ:
  যদি `folder = "/dataset/real/frames/video1"`,
  তাহলে:

  ```python
  imgs = [
      "/dataset/real/frames/video1/frame_001.png",
      "/dataset/real/frames/video1/frame_002.png",
      "/dataset/real/frames/video1/frame_003.png"
  ]
  ```

---

### **4. এই লুপ:**

```python
for img_path in imgs:
    self.image_paths.append(img_path)
    self.labels.append(0)
```

* প্রতিটি ইমেজ পাথকে `self.image_paths` লিস্টে রাখা হচ্ছে।
* সাথে সাথে `self.labels` লিস্টে **`0`** যোগ করা হচ্ছে।

  * এখানে `0` লেবেল মানে — **Real (আসল) ফ্রেম**।

---

### **সংক্ষেপে পুরো কাজ:**

1. `"real/frames"` ফোল্ডারের সব সাব-ফোল্ডার খুঁজে বের করা।
2. প্রতিটি সাব-ফোল্ডারের `.png` ইমেজ পাথ লিস্টে যোগ করা।
3. প্রতিটি ইমেজকে **Real** লেবেল (`0`) দিয়ে সংরক্ষণ করা।

---

---

### ১. `__getitem__` function

```python
def __getitem__(self, idx):
    img_path = self.image_paths[idx]             # Index অনুযায়ী image path নেওয়া
    image = Image.open(img_path).convert("RGB")  # PIL দিয়ে image read & RGB convert
    label = self.labels[idx]                     # Index অনুযায়ী label নেওয়া
    if self.transform:                           # যদি transform pass করা থাকে
        image = self.transform(image)            # Apply transform
    return image, label                           # Image tensor + label return
```

**Explanation:**

1. `img_path = self.image_paths[idx]`
   → এই index-এর image path fetch করা হলো।

2. `Image.open(img_path).convert("RGB")`
   → Image read করে 3-channel RGB convert করা হলো।

3. `label = self.labels[idx]`
   → একই index-এর label fetch করা হলো।

4. `if self.transform:`
   → যদি transform function pass করা থাকে, সেটা apply হবে।

5. `return image, label`
   → Transformed image + label return করে। DataLoader batch-wise process করবে।

---

### ২. Transform function কীভাবে কাজ করে

তুমি যে `train_transforms` define করেছো:

```python
train_transforms = transforms.Compose([
    transforms.Resize((299, 299)),          # Xception model input size
    transforms.RandomHorizontalFlip(),      # Data augmentation: random flip
    transforms.ToTensor(),                  # PIL image → PyTorch tensor
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    ),                                      # Image normalization
])
```

**Step-by-Step:**

1. **`transforms.Resize((299, 299))`**

   * সব image একই size-এ convert হবে।
   * Xception model 299x299 input চায়।

2. **`transforms.RandomHorizontalFlip()`**

   * Random horizontal flip করা হয়।
   * Data augmentation: training set variability বাড়ায়।

3. **`transforms.ToTensor()`**

   * PIL image কে PyTorch tensor এ convert করে।
   * Pixel value 0-255 থেকে 0-1 range এ normalize হয়।

4. **`transforms.Normalize(mean, std)`**

   * Standard normalization।
   * Channel-wise mean এবং std apply করে।
   * Pretrained model এর জন্য consistency রাখে।

---

### ৩. Validation transforms

```python
val_transforms = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406], 
        std=[0.229,0.224,0.225]
    ),
])
```

**Difference:**

* Validation এ **RandomHorizontalFlip** নেই।
* কারন validation set data augmentation করা হয় না। শুধুই resize, tensor conversion এবং normalization।

---

### ৪. Summary – `transform` workflow

| Stage                 | Purpose                                                 |
| --------------------- | ------------------------------------------------------- |
| `__getitem__`         | Index অনুযায়ী image load করা, label fetch করা           |
| `transform(image)`    | Image augmentation, resize, normalization apply করা     |
| `train_transforms`    | Training image-এ random flip, resize, normalization     |
| `val_transforms`      | Validation image-এ শুধু resize + normalization          |
| `return image, label` | PyTorch tensor-ready image + corresponding label return |

---

💡 **Conclusion:**

* `transform` হলো **dynamic image preprocessing pipeline**, যা DataLoader এর মাধ্যমে batch-wise apply হয়।
* Training এবং validation এর জন্য আলাদা transform ব্যবহার করা হয়।
* তাই `__getitem__` এ `self.transform(image)` call করার ফলে, automatically সেই pipeline apply হয়ে যাবে।

---


---

### **Horizontal Flip এর অর্থ**

* Horizontal Flip মানে **ছবিটিকে আড়াআড়িভাবে উল্টানো**।
* বাম পাশের অংশ ডান দিকে চলে আসে, ডান পাশের অংশ বামে আসে।
* শুধু **left ↔ right** পরিবর্তন হয়, **উপর-নিচ পরিবর্তন হয় না**।

---

### **PyTorch এ RandomHorizontalFlip()**

```python
transforms.RandomHorizontalFlip(p=0.5)
```

* `p=0.5` → ৫০% probability এ ছবিটি flip হবে।
* অর্থাৎ, প্রতিটি image-এ randomly flip হবে বা একই থাকবে।
* **Purpose:** Model কে শেখানো যে object orientation (left-right) পরিবর্তন হলেও সেটা একই class।
* এটি একটি **data augmentation technique**, যা \*\*overfitting কমাতে সাহায্য করে।

---

### **উদাহরণ**

ধরা যাক original image:

```
[🚗][🏠][🌳]  ← left to right
```

Horizontal Flip করলে:

```
[🌳][🏠][🚗]  ← left to right
```

* শুধু চিত্রের দিক উল্টেছে, content ঠিক আছে।
* Model এখন left-right orientation-এর variation recognize করতে পারবে।

---

### **Training vs Validation**

* **Training Dataset** → RandomHorizontalFlip apply করা হয় → model more robust হয়।
* **Validation/Test Dataset** → Flip করা হয় না → accuracy evaluation consistent থাকে।

---

💡 **Key Point:** Horizontal Flip একটি **simple augmentation**, কিন্তু small datasets বা face/video data–এ model generalization অনেক improve করতে পারে।

---



---

### **Normalize এর অর্থ**

* **Normalization** হলো ছবির pixel values কে এমনভাবে পরিবর্তন করা যাতে মডেল training সহজে এবং দ্রুত convergence করতে পারে।
* সাধারণত, RGB image-এ pixel values 0–255 এর মধ্যে থাকে।
* Neural network–এ ভালো performance এর জন্য, pixel values **0–1 বা standardized mean/std** এ আনা হয়।

---

### **PyTorch এ Normalize()**

```python
transforms.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
```

* **mean** → প্রতিটি channel (R, G, B) এর mean value।
* **std** → প্রতিটি channel (R, G, B) এর standard deviation।
* Formula:

$$
x_{\text{normalized}} = \frac{x - \text{mean}}{\text{std}}
$$

* উদাহরণ: যদি red pixel মান 0.6 হয়, mean=0.485, std=0.229, তাহলে normalized value হবে:

$$
(0.6 - 0.485)/0.229 \approx 0.505
$$

---

### **Normalize কেন দরকার?**

1. Model দ্রুত train হয়।
2. Gradient instability কমে।
3. Pretrained models (যেমন **Xception, ResNet**) যেগুলো ImageNet dataset এ train হয়েছে, তাদের input pixel distribution একই pattern অনুসরণ করে।

---

### **Training vs Validation**

* **Training**: Normalize + augmentations (যেমন Horizontal Flip) apply হয়।
* **Validation/Test**: শুধুমাত্র Normalize apply হয় → fairness এবং consistency নিশ্চিত হয়।

---

💡 **Key Point:**
Normalize মুলত **input image কে standardized scale এ নিয়ে আসে**, যাতে pretrained model বা নতুন network সহজে learn করতে পারে।

---




