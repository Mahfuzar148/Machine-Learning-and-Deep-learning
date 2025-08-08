
---

## 🔍 কোড:

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
```

---

## 📄 এটি কেন দরকার?

আমরা যখন PyTorch দিয়ে ছবি ভিত্তিক ডিপ লার্নিং মডেল বানাই (যেমনঃ CNN), তখন ছবিগুলোকে প্রি-প্রসেস করতে হয়, যাতে:

* ছবি টেনসরে কনভার্ট হয় (PyTorch টেনসর সাপোর্ট করে)
* পিক্সেল মানগুলো 0-1 স্কেলে আসে (scaling)
* তারপর ডেটা center হয় (mean=0 এর দিকে আনা হয়)
* মডেল ভালোভাবে এবং দ্রুত শিখতে পারে

এই সব কাজ একসাথে করতে আমরা `transforms.Compose([...])` ব্যবহার করি।

---

## 🧱 লাইন বাই লাইন ডকুমেন্টেশন

---

### ✅ 1. `transforms.Compose([...])`

👉 এটি হলো একটি কম্পোজিশন ক্লাস, যেখানে তুমি একাধিক image transformation বা processing স্টেপ একত্রে দিয়ে রাখতে পারো।
এখানে একাধিক transforms কে চেইন করে একটাই `transform` অবজেক্ট বানানো হয়।

📌 উদাহরণ:
ইমেজ টেনসরে রূপান্তর → তারপর নরমালাইজ → তারপর রিসাইজ → সব একত্রে `Compose` এ রাখা যায়।

---

### ✅ 2. `transforms.ToTensor()`

#### 🔧 কাজ:

* ইমেজ ফাইল (PIL Image or NumPy array) কে **PyTorch Tensor** তে কনভার্ট করে
* পিক্সেল মানগুলোর স্কেল পরিবর্তন করে:
  `0–255 → 0.0–1.0`

#### 🔁 মানে:

* ইমেজের ডেটা float32 টাইপে এবং PyTorch format (`[C, H, W]`) এ রূপান্তরিত হয়।

#### 🔥 না করলে কী হবে?

* মডেল PyTorch টেনসর ইনপুট হিসেবে নেবে না
* Training কাজই করবে না
* ইমেজ float32 না হলে গ্র্যাডিয়েন্ট ক্যালকুলেশন হবে না

---

### ✅ 3. `transforms.Normalize((0.1307,), (0.3081,))`

#### 🔧 কাজ:

ইমেজ টেনসরের প্রতিটি পিক্সেলকে **normalize** করে নিচের ফর্মুলা অনুসারে:

$$
\text{output} = \frac{\text{input} - \text{mean}}{\text{std}}
$$

#### এখানে:

* `mean = 0.1307`
* `std = 0.3081`

📌 এটা MNIST ডেটাসেটের জন্য ব্যবহৃত মেট্রিকস। MNIST ডেটাসেটের সমস্ত পিক্সেল ভ্যালুর গড় হল `0.1307` এবং স্ট্যান্ডার্ড ডেভিয়েশন `0.3081`

#### 🔁 মানে:

* সব ডেটাকে একই স্কেলে নিয়ে আসা হয় (standardization)
* gradient descent দ্রুত ও সঠিকভাবে কাজ করে

#### 🔥 না করলে কী হবে?

* মডেলের ট্রেইনিং ধীরে চলবে
* ওজন আপডেট ঠিকভাবে হবে না
* overfitting বা underfitting হতে পারে
* ভিন্ন ভিন্ন স্কেলের ডেটা নিয়ে মডেল বিভ্রান্ত হতে পারে

---

## 🧪 প্র্যাকটিকাল দৃষ্টিকোণ থেকে:

| কাজ                            | ToTensor() না দিলে            | Normalize() না দিলে            |
| ------------------------------ | ----------------------------- | ------------------------------ |
| Image to tensor                | ❌ মডেল এক্সেপ্ট করবে না       | ✔️ চলবে, তবে বাজে পারফরম্যান্স |
| Scaling (0–1)                  | ❌ gradient ঠিকমতো কাজ করবে না | ✔️ চলবে, কিন্তু ধীর            |
| Training stability             | ❌                             | ❌                              |
| Accuracy                       | ❌                             | ❌                              |
| Dataset-specific preprocessing | ❌                             | ❌                              |

---

## ✅ কখন এই Transform ব্যবহার করব?

* যখন তুমি PyTorch দিয়ে ইমেজ মডেল বানাবে
* বিশেষ করে grayscale ইমেজ নিয়ে (যেমনঃ MNIST)
* যখন মডেল ভাল এবং দ্রুত ট্রেইন করতে চাও

---

## ✅ কীভাবে কাজ করে?

এই কোডটা যখন `datasets.MNIST()` বা `DataLoader()` এ `transform=transform` দিয়ে পাস করা হয়, তখন প্রতিটি ইমেজের ওপর এই দুইটি অপারেশন চালানো হয়:

```python
image = ToTensor(image)
image = Normalize(image)
```

---

## ✅ পুরো ডকুমেন্টেশনের সারাংশ:

```python
# Full Explanation
transform = transforms.Compose([
    transforms.ToTensor(),                     # Step 1: Convert image to PyTorch Tensor and scale to [0,1]
    transforms.Normalize((0.1307,), (0.3081,)) # Step 2: Normalize tensor using MNIST's mean and std
])
```

---

## 🔚 শেষ কথা:

* ✅ `ToTensor()` ও `Normalize()` — এই দুটো মিলে PyTorch মডেলের জন্য ডেটা পারফেক্টলি প্রস্তুত করে।
* ❌ যেকোনো একটা বাদ দিলেও মডেল ঠিকমতো শিখবে না, পারফরম্যান্স খারাপ হবে।
* 🎯 এটা শুধু MNIST নয়, CIFAR, ImageNet — সব ডেটাসেটেই আলাদা mean/std দিয়ে normalization করা হয়।

---

