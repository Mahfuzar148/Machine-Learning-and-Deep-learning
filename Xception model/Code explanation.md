
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

