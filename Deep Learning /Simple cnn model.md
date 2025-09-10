
---

## 📌 Code (Keras CNN Model)

```python
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input
from tensorflow.keras.models import Model

num_classes = 10
img_size = (28,28,1)

inputs = Input(img_size)

x = Conv2D(filters=8, kernel_size=(3,3), activation='relu')(inputs)
x = Conv2D(filters=16, kernel_size=(3,3), activation='relu')(x)
x = Conv2D(filters=32, kernel_size=(3,3), activation='relu')(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
x = Dense(16, activation='relu')(x)
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs, outputs)
model.summary(show_trainable=True)
```

---



ধরা যাক তোমার input image shape = `(28,28,1)` (মানে grayscale, 1 channel)।

---

### 1️⃣ প্রথম Conv2D (filters=8, kernel\_size=3x3)

* এখানে ৮টা আলাদা **3×3 kernel (filter matrix)** initialize হবে।
* প্রতিটি filter input image-এর উপর slide করবে → convolution করবে।
* প্রতিটি filter → একটা **feature map (activation map)** produce করবে।
* তাই output হবে shape: `(26,26,8)` (কারণ 28-3+1 = 26, আর depth = 8 maps)।

---

### 2️⃣ দ্বিতীয় Conv2D (filters=16, kernel\_size=3x3)

* এখন input shape হবে `(26,26,8)` → মানে ৮টা channel।
* প্রতিটি filter এবার **3×3×8 (depth = 8)** dimension-এর হবে।
* কারণ convolution filter-এর depth সবসময় input-এর channel depth-এর সমান হয়।
* তুমি ১৬টা filter দিলে → প্রতিটি filter একটা feature map বানাবে।
* তাই output shape হবে `(24,24,16)`।

---

### 3️⃣ তৃতীয় Conv2D (filters=32, kernel\_size=3x3)

* এখন input shape `(24,24,16)`।
* প্রতিটি filter এখন হবে **3×3×16**।
* তুমি ৩২টা filter দিলে → ৩২টা feature map হবে।
* Output shape হবে `(22,22,32)`।

---

✅ **সারাংশ:**

* `filters=N` মানে ঐ লেয়ারে **N টা আলাদা kernel matrix** শিখবে।
* প্রতিটি kernel দিয়ে convolution করলে একটা feature map পাওয়া যাবে।
* সুতরাং ৮ দিলে → ৮টা kernel, ৮টা feature map।
* কিন্তু পরের layer-এ kernel-এর depth বাড়ে, কারণ সেটা আগের layer-এর সব feature একসাথে দেখে।


---

## 📌 Code Segment

```python
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
x = Dense(16, activation='relu')(x)
```

---

## 📖 Step-by-Step Explanation

### 1️⃣ **Flatten Layer**

```python
x = Flatten()(x)
```

* Convolutional layers শেষে তোমার output shape = `(22, 22, 32)` (height × width × depth)।
* Flatten এটাকে **একটা লম্বা ভেক্টর** বানাবে।
* হিসাব: `22 × 22 × 32 = 15488` → মানে 15488টা সংখ্যা হবে।
* এই ভেক্টরটা হলো CNN দ্বারা শেখা সমস্ত feature এর numerical representation।
* কেন দরকার?
  👉 Fully connected layer (Dense) input হিসেবে শুধু 1D ভেক্টর নিতে পারে।

---

### 2️⃣ **Dense(128, activation='relu')**

```python
x = Dense(128, activation='relu')(x)
```

* Input: 15488 neurons (flatten করা ফিচার ভেক্টর)।
* Output: 128 neurons।
* প্রতিটি neuron আগের সব ইনপুট থেকে weighted sum শিখবে।
* Activation = **ReLU** → nonlinear mapping যোগ করে।
* কাজ:
  👉 শেখা ফিচারগুলোকে compressed করে একটা ছোট্ট (128 dimension) representation বানানো।

---

### 3️⃣ **Dense(64, activation='relu')**

```python
x = Dense(64, activation='relu')(x)
```

* Input: 128 neurons।
* Output: 64 neurons।
* মানে: network এখন আরও compact একটা representation বানাচ্ছে।
* ReLU activation এখনও nonlinearity বজায় রাখছে।
* কাজ:
  👉 Feature extraction → আরও বেশি high-level abstraction তৈরি করা।

---

### 4️⃣ **Dense(32, activation='relu')**

```python
x = Dense(32, activation='relu')(x)
```

* Input: 64 neurons।
* Output: 32 neurons।
* মানে feature space এখন আরও ছোট হলো।
* কাজ:
  👉 Network ধাপে ধাপে গুরুত্বপূর্ণ feature ধরে রাখছে, কম দরকারি জিনিস ফেলে দিচ্ছে।

---

### 5️⃣ **Dense(16, activation='relu')**

```python
x = Dense(16, activation='relu')(x)
```

* Input: 32 neurons।
* Output: 16 neurons।
* খুব ছোট এবং compressed representation।
* এরপরে final output layer থাকবে (softmax, 10 classes)।
* কাজ:
  👉 এই লেয়ার basically network-এর decision boundary বানানোর আগে “last hidden representation” হিসেবে কাজ করছে।

---

## ✅ সারাংশ

* **Flatten:** 2D feature maps → 1D ভেক্টর।
* **Dense 128 → 64 → 32 → 16:** ধাপে ধাপে feature dimension ছোট করছে, যাতে network সবচেয়ে useful features retain করে।
* প্রতিটি Dense layer nonlinear mapping শেখে (ReLU দিয়ে)।
* এগুলো একসাথে মিলে network-এর **classifier head** তৈরি করে, যা শেষে softmax output layer-এ গিয়ে class prediction দেয়।

---


