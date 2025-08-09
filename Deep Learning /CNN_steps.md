

---

## ১. ডেটা প্রিপারেশন

প্রথমেই ডেটা ঠিকঠাক করে প্রস্তুত করতে হয়, কারণ মডেল ভালো শিখতে হলে ডেটাও ভালো হতে হবে। এই ধাপে মূলত তিনটা কাজ হয়:

1. **ডেটা ক্লিনিং (Data Cleaning)**

   * ডেটার মধ্যে ভুল, মিসিং ভ্যালু বা ডুপ্লিকেট রেকর্ড থাকলে সেগুলো ঠিক করা বা বাদ দেওয়া হয়।
   * উদাহরণ: ছবির মধ্যে নষ্ট ফাইল মুছে ফেলা, ভুল লেবেল ঠিক করা ইত্যাদি।

2. **ডেটা নরমালাইজেশন (Normalization)**

   * ডেটাকে একটি নির্দিষ্ট স্কেলে নিয়ে আসা হয়, যাতে মডেল দ্রুত শিখতে পারে এবং ট্রেনিং স্থিতিশীল হয়।
   * যেমন ছবির পিক্সেল ভ্যালুকে `0-255` থেকে `0-1` স্কেলে আনা।

3. **ডেটা অগমেন্টেশন (Data Augmentation)**

   * ডেটার কৃত্রিম বৈচিত্র্য তৈরি করা হয়, যেমন ছবি ঘোরানো, উল্টানো, জুম করা, কালার পরিবর্তন করা ইত্যাদি।
   * এর ফলে মডেল একই জিনিসের ভিন্ন ভিন্ন রূপ দেখে শিখতে পারে, যা **Generalization** ক্ষমতা বাড়ায়।
   * *Generalization* মানে হলো মডেল নতুন, আগে না দেখা ডেটাতেও ভালো পারফর্ম করবে।

---

## ২. মডেল আর্কিটেকচার ডিজাইন

CNN মডেলের মূল কাঠামো তৈরি হয় এই ধাপে। এখানে সাধারণত ব্যবহৃত লেয়ারগুলো হলো:

* **Convolutional Layer** → ছবির ফিচার বের করে (যেমন ধার, আকার, রঙের প্যাটার্ন)
* **Pooling Layer** → ইমেজ ছোট করে গুরুত্বপূর্ণ তথ্য রেখে বাকি বাদ দেয়
* **Flatten Layer** → 2D ডেটাকে 1D ভেক্টরে রূপান্তর করে যাতে ডেন্স লেয়ারে দেওয়া যায়
* **Dense Layer** → চূড়ান্ত প্রেডিকশন দেয়

---

## ৩. হাইপারপ্যারামিটার টিউনিং

* **হাইপারপ্যারামিটার** হলো মডেলের সেটিংস যেমন লার্নিং রেট, ব্যাচ সাইজ, লেয়ার সংখ্যা, ফিল্টার সাইজ ইত্যাদি।
* **অটো-টিউনিং (Auto-tuning)** করলে এই সেটিংসগুলো স্বয়ংক্রিয়ভাবে বদলে মডেলের পারফরম্যান্স সর্বোচ্চ করা হয়।
* *মূল পার্থক্য*:

  * **Augmentation** → ডেটার বৈচিত্র্য বাড়ায়
  * **Auto-tuning** → মডেলের সেটিংস এমনভাবে বদলায় যাতে পারফরম্যান্স ভালো হয়
* এই দুই প্রক্রিয়া একসাথে ব্যবহার করলে মডেল আরও বেশি নির্ভুল হয়।

---

## ৪. মডেল কম্পাইল ও ট্রেনিং

1. **মডেল কম্পাইল (Compile)**

   * এখানে বলা হয় কোন **Loss Function** ব্যবহার হবে, কোন **Optimizer** দিয়ে মডেল শিখবে, আর কোন **Metrics** মাপা হবে।
2. **ট্রেনিং (Training)**

   * ট্রেনিং ডেটা দিয়ে মডেল শিখে নেয়।
3. **কলব্যাক (Callbacks)**

   * **Early Stopping** → পারফরম্যান্স উন্নতি বন্ধ হলে ট্রেনিং থামিয়ে দেয়।
   * **Model Checkpoint** → ট্রেনিং চলাকালীন সেরা মডেলটি সেভ করে রাখে।

---

## ৫. ভ্যালিডেশন ও ইভ্যালুয়েশন

* **ভ্যালিডেশন ডেটা** → ট্রেনিং চলাকালীন মডেলের পারফরম্যান্স চেক করা হয়।
* **ইয়ারলি স্টপিং** ব্যবহার করলে সময় বাঁচে এবং ওভারফিটিং কমে।
* **ট্রেনিং কার্ভ** প্লট করলে বোঝা যায় মডেলের লস ও অ্যাকুরেসি সময়ের সাথে কেমন বদলাচ্ছে।

---

## ৬. ইনফারেন্স (Inference)

* ট্রেনিং শেষ হলে মডেল নতুন ডেটার উপর প্রেডিকশন দিতে পারে।
* এই ধাপে ট্রেনিংয়ের তুলনায় অনেক দ্রুত কাজ হয়।
* বাস্তব জীবনের ডেটায় মডেল চালিয়ে ফলাফল বের করার জন্য ইনফারেন্স ব্যবহার করা হয়।

---

---

## **১. ডেটা প্রিপারেশন (Data Preparation)**

### 1.1 ডেটা ক্লিনিং (Data Cleaning)

মডেল যেন ভুল ডেটা দেখে বিভ্রান্ত না হয়, তাই প্রথমে ডেটা পরিষ্কার করতে হয়।

* **উদাহরণ**: যদি তোমার কাছে বিড়াল ও কুকুরের ছবি থাকে, কিন্তু কিছু ছবিতে ফাইল করাপ্ট বা লেবেল ভুল থাকে, তাহলে সেটা মুছে ফেলতে হবে বা ঠিক করতে হবে।
* **রিয়েল লাইফ**: মেডিকেল এক্স-রে ডেটাসেটে কিছু ছবি হয়ত খুব ডার্ক বা ব্লারি—এগুলো বাদ দিতে হয়।

### 1.2 ডেটা নরমালাইজেশন (Normalization)

পিক্সেল ভ্যালু সাধারণত 0 থেকে 255 এর মধ্যে থাকে। নরমালাইজেশন করে আমরা এই ভ্যালুকে 0–1 স্কেলে আনি।

* **কেন দরকার**: বড় ভ্যালু থাকলে gradient calculation-এ numerical instability হতে পারে, আর মডেলের কনভার্জ হতে দেরি হয়।
* **উদাহরণ**:

```python
# পিক্সেল ভ্যালু 0-255 থেকে 0-1 এ রূপান্তর
X_train = X_train / 255.0
```

### 1.3 ডেটা অগমেন্টেশন (Data Augmentation)

এখানে ডেটাকে নতুন ভ্যারিয়েশনে রূপান্তর করা হয়, যাতে মডেল এক জিনিসের অনেক রকম দেখতে শিখে।

* **প্রযুক্তি**:

  * Rotation (ছবি ঘোরানো)
  * Horizontal/Vertical flip
  * Zoom in/out
  * Brightness পরিবর্তন
* **উদাহরণ**:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(rotation_range=20, horizontal_flip=True, zoom_range=0.2)
datagen.fit(X_train)
```

* **প্রভাব**: যদি তুমি কেবল সোজা বিড়ালের ছবি দেখাও, মডেল উল্টো বা বাঁকা বিড়াল চিনতে পারবে না। Augmentation এটা সমাধান করে।

---

## **২. মডেল আর্কিটেকচার ডিজাইন (Model Architecture Design)**

CNN মূলত কয়েক ধরনের লেয়ার নিয়ে তৈরি হয়:

1. **Convolutional Layer** – ফিল্টার দিয়ে ইমেজের ফিচার বের করে (যেমন ধার, টেক্সচার)।
2. **Pooling Layer** – ইমেজ সাইজ ছোট করে, কিন্তু গুরুত্বপূর্ণ তথ্য রেখে দেয়।
3. **Flatten Layer** – 2D ডেটাকে 1D ভেক্টরে রূপান্তর করে।
4. **Dense Layer** – শেষ ধাপে সিদ্ধান্ত নেয় কোন ক্লাস হবে।

**উদাহরণ আর্কিটেকচার**:

```python
from tensorflow.keras import layers, models
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
```

---

## **৩. হাইপারপ্যারামিটার টিউনিং (Hyperparameter Tuning)**

### 3.1 কী কী হাইপারপ্যারামিটার টিউন হয়:

* Learning Rate
* Batch Size
* Number of Epochs
* Filter size এবং সংখ্যা
* Dropout rate

### 3.2 অটো-টিউনিং (Auto-tuning)

AutoML বা Keras Tuner এর মতো টুল দিয়ে স্বয়ংক্রিয়ভাবে সেরা সেটিং খুঁজে পাওয়া যায়।

* **উদাহরণ**:

```python
import keras_tuner as kt
def build_model(hp):
    model = models.Sequential()
    model.add(layers.Conv2D(hp.Int('filters', 32, 128, step=32), (3,3), activation='relu'))
    ...
```

**পার্থক্য**:

* **Augmentation** → ডেটা সমৃদ্ধ করে
* **Auto-tuning** → মডেলের সেটিং ঠিক করে

---

## **৪. মডেল কম্পাইল এবং ট্রেনিং**

### 4.1 কম্পাইল (Compile)

```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

এখানে মডেলকে জানানো হয়:

* কোন অপটিমাইজার ব্যবহার হবে
* কোন লস ফাংশন
* কোন মেট্রিক ট্র্যাক হবে

### 4.2 ট্রেনিং (Fit) + কলব্যাকস

```python
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', save_best_only=True)
]
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, callbacks=callbacks)
```

* **Early Stopping** → পারফরম্যান্স না বাড়লে থামায়
* **Model Checkpoint** → সেরা ভার্সন সেভ করে

---

## **৫. ভ্যালিডেশন ও ইভ্যালুয়েশন**

* ট্রেনিং চলাকালীন validation loss ও accuracy দেখে বোঝা যায় ওভারফিটিং হচ্ছে কিনা।
* **উদাহরণ ট্রেনিং কার্ভ**:

  * যদি ট্রেনিং accuracy বাড়ছে কিন্তু validation accuracy কমছে → ওভারফিটিং
  * যদি দুইটাই একসাথে বাড়ছে → ভালো ট্রেনিং

---

## **৬. ইনফারেন্স (Inference)**

ট্রেনিং শেষ হওয়ার পর নতুন ডেটায় প্রেডিকশন:

```python
import numpy as np
img = preprocess_image('cat.jpg')
pred = model.predict(np.expand_dims(img, axis=0))
print("Cat" if pred > 0.5 else "Dog")
```

* রিয়েল লাইফে: রোডে থাকা সাইনবোর্ড চিনতে সেলফ-ড্রাইভিং কার ইনফারেন্স ব্যবহার করে
* মেডিকেল স্ক্যান থেকে ক্যান্সারের সম্ভাবনা বের করাও ইনফারেন্সের উদাহরণ

---

## **প্র্যাক্টিক্যাল এক্সাম্পল (ফ্লো চার্ট)**

1. ডেটা সংগ্রহ → পরিষ্কার → নরমালাইজেশন → Augmentation
2. CNN আর্কিটেকচার তৈরি
3. হাইপারপ্যারামিটার ঠিক করা
4. কম্পাইল + ট্রেনিং (কলব্যাক সহ)
5. ভ্যালিডেশন → মডেল সেভ
6. নতুন ডেটায় ইনফারেন্স

---


---

## **১. ডেটা প্রিপারেশন (Data Preparation)**

এটা পুরো মেশিন লার্নিং পাইপলাইনের ভিত্তি। ডেটা যত ভালো, মডেলের শেখার ক্ষমতাও তত ভালো হবে।

### **1.1 ডেটা ক্লিনিং (Data Cleaning)**

* **কাজ**: ভুল বা খারাপ ডেটা সরানো, লেবেল ঠিক করা, মিসিং ভ্যালু পূরণ করা।
* **উদাহরণ**: যদি বিড়াল-কুকুর ডেটাসেটে কিছু ছবিতে ভুল লেবেল থাকে (বিড়ালকে কুকুর বলা), সেটা ঠিক করা দরকার।

### **1.2 নরমালাইজেশন (Normalization)**

* পিক্সেল ভ্যালু সাধারণত `0-255`। নরমালাইজ করে `0-1` স্কেলে আনা হয় যাতে মডেল দ্রুত শিখতে পারে এবং গ্রেডিয়েন্ট স্টেবল থাকে।

```python
X_train = X_train / 255.0
```

### **1.3 ডেটা অগমেন্টেশন (Data Augmentation)**

* একই ছবির বিভিন্ন ভ্যারিয়েশন তৈরি করে, যাতে মডেল জেনারেলাইজ করতে পারে।
* টেকনিক: Rotation, Flip, Zoom, Brightness change

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(rotation_range=20, horizontal_flip=True, zoom_range=0.2)
datagen.fit(X_train)
```

* **উপকারিতা**: যদি শুধু সোজা ছবি দিয়ে ট্রেনিং করো, মডেল উল্টো/বাঁকা ছবি চিনতে পারবে না। Augmentation সেই ফাঁক পূরণ করে।

---

## **২. মডেল আর্কিটেকচার ডিজাইন (Model Architecture Design)**

### **CNN-এর প্রধান লেয়ার**

1. **Convolutional Layer** → ছবি থেকে ফিচার (edge, shape, texture) বের করে
2. **Pooling Layer** → ছবির সাইজ ছোটায় কিন্তু দরকারি তথ্য রাখে
3. **Flatten Layer** → 2D ডেটা → 1D vector
4. **Dense Layer** → চূড়ান্ত প্রেডিকশন

```python
from tensorflow.keras import layers, models
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
```

---

## **৩. হাইপারপ্যারামিটার টিউনিং (Hyperparameter Tuning)**

### **কি টিউন করা হয়**

* Learning rate
* Batch size
* Epochs সংখ্যা
* Filter size
* Dropout rate

### **অটো-টিউনিং (Auto-tuning)**

* মডেলের জন্য সেরা সেটিং স্বয়ংক্রিয়ভাবে বের করা হয়।
* **Augmentation** ডেটা উন্নত করে, **Auto-tuning** মডেলের সেটিং উন্নত করে।

```python
import keras_tuner as kt
def build_model(hp):
    model = models.Sequential()
    model.add(layers.Conv2D(hp.Int('filters', 32, 128, step=32), (3,3), activation='relu'))
    ...
```

---

## **৪. মডেল কম্পাইল ও ট্রেনিং**

### **কম্পাইল (Compile)**

```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

* **Optimizer** → কিভাবে ওয়েট আপডেট হবে
* **Loss Function** → প্রেডিকশন আর আসল ভ্যালুর মধ্যে ত্রুটি মাপা
* **Metrics** → পারফরম্যান্স মাপা

### **ট্রেনিং + কলব্যাকস**

```python
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', save_best_only=True)
]
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, callbacks=callbacks)
```

* **Early Stopping** → পারফরম্যান্স না বাড়লে ট্রেনিং বন্ধ
* **Model Checkpoint** → সেরা মডেল সেভ

---

## **৫. ভ্যালিডেশন ও ইভ্যালুয়েশন**

* **Validation Set** → ট্রেনিং চলাকালে মডেল কেমন করছে সেটা দেখা
* **Training Curve** → Accuracy ও Loss গ্রাফ দেখে ওভারফিটিং চেক করা

  * যদি ট্রেনিং অ্যাকুরেসি বাড়ে কিন্তু ভ্যালিডেশন অ্যাকুরেসি কমে → ওভারফিটিং

---

## **৬. ইনফারেন্স (Inference)**

ট্রেনিং শেষে নতুন ডেটায় প্রেডিকশন:

```python
img = preprocess_image('cat.jpg')
pred = model.predict(np.expand_dims(img, axis=0))
print("Cat" if pred > 0.5 else "Dog")
```

* রিয়েল লাইফে: ফেস রিকগনিশন, মেডিকেল ডায়াগনোসিস, সেলফ-ড্রাইভিং গাড়ি ইত্যাদি ইনফারেন্সে CNN ব্যবহার করে।

---

## **বাস্তব উদাহরণ**

**প্রজেক্ট**: বিড়াল বনাম কুকুর চিনতে পারা মডেল

1. **ডেটা সংগ্রহ** → Kaggle Cat vs Dog Dataset
2. **প্রসেসিং** → সব ছবি 64x64 রিসাইজ, নরমালাইজ, Augmentation
3. **CNN আর্কিটেকচার** → ২টা Convolution + Pooling লেয়ার, তারপর Dense লেয়ার
4. **টিউনিং** → লার্নিং রেট 0.001, ব্যাচ সাইজ 32
5. **ট্রেনিং** → 20 epoch, Early Stopping সহ
6. **টেস্ট** → নতুন বিড়াল-কুকুর ছবিতে পরীক্ষা
7. **ইনফারেন্স** → ওয়েবক্যাম থেকে লাইভ প্রেডিকশন

---


