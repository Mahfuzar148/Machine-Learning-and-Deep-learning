
---

## **ফাংশনের কাজ**

`image_dataset_from_directory()` → কোনো ফোল্ডারের ভেতরের ইমেজ ডেটা লোড করে `tf.data.Dataset` বানায়, যাতে ইমেজ আর লেবেল একসাথে পাওয়া যায়।

* সাব-ফোল্ডারের নাম থেকে লেবেল তৈরি হয় (যদি `labels='inferred'` হয়)।
* `.jpeg`, `.jpg`, `.png`, `.bmp`, `.gif` ফরম্যাট সাপোর্ট করে (animated gif হলে শুধু প্রথম ফ্রেম নেয়)।

---

## **প্যারামিটার ব্যাখ্যা**

### **1. directory**

* ইমেজ ডেটা যেখানে রাখা আছে সেই মূল ফোল্ডারের পথ।
* যদি `labels='inferred'` হয় → ডিরেক্টরির মধ্যে প্রতিটি সাব-ফোল্ডারকে একটি ক্লাস ধরা হবে।

---

### **2. labels**

* `"inferred"` → সাব-ফোল্ডারের নাম থেকে লেবেল তৈরি হবে (যেমন: class\_a → 0, class\_b → 1)।
* `None` → কোনো লেবেল দেওয়া হবে না (শুধু ছবি)।
* কাস্টম লিস্ট/টিউপল → তোমাকে নিজে থেকে প্রতিটি ছবির জন্য লেবেল দিতে হবে।

---

### **3. label\_mode**

লেবেল কিভাবে ফরম্যাট হবে:

* `"int"` → লেবেল হবে ইন্টিজার (যেমন `0`, `1`, `2`) → `sparse_categorical_crossentropy` এর জন্য।
* `"categorical"` → লেবেল হবে one-hot encoded ভেক্টর (যেমন `[1,0,0]`) → `categorical_crossentropy` এর জন্য।
* `"binary"` → ২ ক্লাস হলে লেবেল হবে `0` বা `1` ফ্লোট ভ্যালু → `binary_crossentropy` এর জন্য।
* `None` → কোনো লেবেল থাকবে না।

---

### **4. class\_names**

* শুধু তখনই কাজ করবে যখন `labels="inferred"`।
* সাব-ফোল্ডারের নামের তালিকা দিয়ে দিলে সেটার ক্রম অনুসারে লেবেল হবে।
* না দিলে আলফাবেটিক্যাল অর্ডারে হবে।

---

### **5. color\_mode**

* `"grayscale"` → ১ চ্যানেল (কালো-সাদা)।
* `"rgb"` → ৩ চ্যানেল (কালার)।
* `"rgba"` → ৪ চ্যানেল (কালার + আলফা চ্যানেল)।

---

### **6. batch\_size**

* এক ব্যাচে কয়টা ছবি লোড হবে (ডিফল্ট 32)।
* `None` দিলে প্রতিবার একটি করে ছবি পাওয়া যাবে।

---

### **7. image\_size**

* (উচ্চতা, প্রস্থ) → সব ছবি এই সাইজে রিসাইজ হবে।
* উদাহরণ: `(256, 256)`।

---

### **8. shuffle**

* `True` → ছবিগুলোর ক্রম এলোমেলো হবে।
* `False` → আলফাবেটিক্যাল ক্রমে আসবে।

---

### **9. seed**

* র‌্যান্ডম শাফল করার জন্য নির্দিষ্ট seed ভ্যালু, যাতে প্রতিবার একই রকম শাফল হয়।

---

### **10. validation\_split**

* ডেটার একটি অংশ ভ্যালিডেশন সেট হিসেবে রাখার অনুপাত (যেমন 0.2 মানে ২০%)।

---

### **11. subset**

* `"training"` → শুধু ট্রেনিং সেট।
* `"validation"` → শুধু ভ্যালিডেশন সেট।
* `"both"` → ট্রেনিং এবং ভ্যালিডেশন দুইটাই রিটার্ন করবে (টাপল আকারে)।

---

### **12. interpolation**

* রিসাইজ করার সময় কোন পদ্ধতি ব্যবহার হবে (যেমন `"bilinear"`, `"nearest"`, `"bicubic"` ইত্যাদি)।

---

### **13. follow\_links**

* `True` হলে সিম্বলিক লিংকের ফোল্ডারের ছবিগুলোও লোড করবে।

---

### **14. crop\_to\_aspect\_ratio**

* `True` হলে রিসাইজ করার সময় অ্যাসপেক্ট রেশিও ঠিক রেখে ছবি ক্রপ করবে।

---

### **15. pad\_to\_aspect\_ratio**

* `True` হলে অ্যাসপেক্ট রেশিও ঠিক রাখতে ছবির চারপাশে প্যাড যোগ করবে।

---

### **16. data\_format**

* `"channels_last"` (ডিফল্ট) → শেপ হবে `(batch, height, width, channels)`।
* `"channels_first"` → শেপ হবে `(batch, channels, height, width)`।

---

### **17. verbose**

* `True` হলে ক্লাস সংখ্যা ও ফাইল সংখ্যা প্রিন্ট করবে।

---

## **রিটার্ন ভ্যালু**

* যদি `label_mode=None` → শুধু ইমেজ টেনসর।
* যদি লেবেল থাকে → `(images, labels)` টাপল আকারে ডেটা রিটার্ন করবে।

---

---

## **1. `directory`**

* **নিলে:** কোন ফোল্ডার থেকে ছবি লোড হবে সেটি নির্দিষ্ট করবে।
* **না নিলে:** ফাংশন চলবে না, কারণ এটি বাধ্যতামূলক।

---

## **2. `labels`**

* **`"inferred"` নিলে:** সাব-ফোল্ডারের নাম থেকে লেবেল তৈরি হবে।
* **`None` নিলে:** শুধু ছবি আসবে, লেবেল থাকবে না।
* **কাস্টম লিস্ট দিলে:** তুমি নিজে লেবেল সেট করতে পারবে।
* **না দিলে:** ডিফল্ট `"inferred"` হয়ে যাবে।

---

## **3. `label_mode`**

* **নিলে:** লেবেল ফরম্যাট নিয়ন্ত্রণ করতে পারবে (`int`, `binary`, `categorical`)।
* **না নিলে:** ডিফল্ট `int` হবে। কিছু ক্ষেত্রে loss function-এর সাথে mismatch হতে পারে।

---

## **4. `class_names`**

* **নিলে:** লেবেলের অর্ডার নিজের মতো করে সেট করতে পারবে।
* **না নিলে:** সাব-ফোল্ডারের নাম আলফাবেটিক অর্ডারে হবে।

---

## **5. `color_mode`**

* **`"rgb"` নিলে:** ৩ চ্যানেল কালার ছবি হবে (ডিফল্ট)।
* **`"grayscale"` নিলে:** ১ চ্যানেল ছবি হবে।
* **না নিলে:** ডিফল্ট `rgb` হবে।

---

## **6. `batch_size`**

* **নিলে:** প্রতি ব্যাচে কতগুলো ছবি আসবে সেটা ঠিক হবে।
* **না নিলে:** ডিফল্ট 32 হবে।
* **`None` দিলে:** একবারে এক ছবি আসবে।

---

## **7. `image_size`**

* **নিলে:** সব ছবি একই সাইজে রিসাইজ হবে।
* **না নিলে:** ডিফল্ট `(256, 256)` হবে, mismatch হলে error হবে।

---

## **8. `shuffle`**

* **`True` নিলে:** ডেটা এলোমেলো হবে (training-এর জন্য ভালো)।
* **`False` নিলে:** ডেটা অর্ডারে আসবে (যেমন testing-এ কাজে লাগে)।
* **না নিলে:** ডিফল্ট `True`।

---

## **9. `seed`**

* **নিলে:** প্রতিবার একইভাবে shuffle হবে (reproducibility)।
* **না নিলে:** প্রতিবার shuffle আলাদা হবে।

---

## **10. `validation_split`**

* **নিলে:** ডেটার নির্দিষ্ট অংশ ভ্যালিডেশন হিসেবে আলাদা হবে।
* **না নিলে:** সব ডেটা একই সেটে আসবে।

---

## **11. `subset`**

* **`"training"` নিলে:** শুধু ট্রেনিং সেট আসবে।
* **`"validation"` নিলে:** শুধু ভ্যালিডেশন সেট আসবে।
* **`"both"` নিলে:** দুই সেট আলাদা করে আসবে।
* **না নিলে:** সব একসাথে আসবে।

---

## **12. `interpolation`**

* **নিলে:** রিসাইজের সময় মান নিয়ন্ত্রণ করতে পারবে (যেমন quality বা speed অনুযায়ী)।
* **না নিলে:** ডিফল্ট `"bilinear"` হবে।

---

## **13. `follow_links`**

* **`True` নিলে:** symbolic link করা ফোল্ডারের ছবিও লোড হবে।
* **না নিলে:** লোড হবে না।

---

## **14. `crop_to_aspect_ratio`**

* **`True` নিলে:** ছবি aspect ratio ঠিক রেখে ক্রপ হবে।
* **না নিলে:** aspect ratio বদলাতে পারে।

---

## **15. `pad_to_aspect_ratio`**

* **`True` নিলে:** ছবি aspect ratio ঠিক রাখতে padding যোগ হবে।
* **না নিলে:** সরাসরি রিসাইজ হবে।

---

## **16. `data_format`**

* **নিলে:** চ্যানেল অর্ডার ঠিক করতে পারবে (`channels_last` বা `channels_first`)।
* **না নিলে:** ডিফল্ট global config অনুযায়ী হবে।

---

## **17. `verbose`**

* **`True` নিলে:** কতগুলো ছবি ও ক্লাস আছে তা প্রিন্ট হবে।
* **`False` নিলে:** কোনো তথ্য প্রিন্ট হবে না।
* **না নিলে:** ডিফল্ট `True`।

---
`image_dataset_from_directory()`-তে আসলে খুব অল্প কয়েকটা প্যারামিটার **must** (অবশ্যই) নিতে হয়, বাকিগুলো optional।

---

## **অবশ্যই লাগবে**

1. **`directory`**

   * ছবি কোথায় আছে সেটা না দিলে ফাংশন কাজই করবে না।
   * এটা সবসময় দিতে হবে, এবং সঠিক path হতে হবে।

---

## **যদি লেবেল চান, তবে এগুলোও প্রয়োজন**

2. **`labels`**

   * না দিলে ডিফল্ট `"inferred"` হয়, কিন্তু লেবেল চাইলে এই সেটিং বুঝে নিতে হবে।
   * `"inferred"` নিলে সাব-ফোল্ডার স্ট্রাকচার থাকতে হবে।

3. **`label_mode`**

   * লেবেলের ফরম্যাট ঠিক করার জন্য।
   * না দিলে ডিফল্ট `"int"` হবে, কিন্তু কিছু loss function-এর জন্য (`categorical_crossentropy` ইত্যাদি) বদলাতে হবে।

---

## **শুধু ডিফল্টে না চললে প্রয়োজন হবে**

* `image_size` → ডিফল্ট `(256,256)` না চাইলে দিতে হবে।
* `batch_size` → ডিফল্ট 32 না চাইলে দিতে হবে।

---

## **সারসংক্ষেপ**

* **Minimum must**: `directory`
* **লেবেলসহ ট্রেনিং-এর জন্য**: `directory` + `labels` (ডিফল্ট হলেও বোঝা দরকার) + `label_mode` (loss-এর সাথে মিলিয়ে)
* **বাকিগুলো**: ইমেজ প্রসেসিং কাস্টমাইজ করতে বা নির্দিষ্ট সেটিং চাইলে ব্যবহার করো, না হলে ডিফল্টেই চলবে।

---

Dog–Cat ক্লাসিফিকেশনের জন্য `image_dataset_from_directory()` ব্যবহার করলে **মিনিমাম দরকারি সেটিংস** হবে এরকম:

---

## **অবশ্যই নিতে হবে**

1. **`directory`**

   * ডেটা ফোল্ডারের লোকেশন।
   * উদাহরণ স্ট্রাকচার:

     ```
     dataset/
         dogs/
             dog1.jpg
             dog2.jpg
         cats/
             cat1.jpg
             cat2.jpg
     ```

---

## **লেবেল সেটিংস**

2. **`labels`** → `"inferred"`

   * সাব-ফোল্ডারের নাম থেকে লেবেল তৈরি হবে (`dogs` → 0, `cats` → 1)।
   * না দিলেও ডিফল্ট এইটাই হবে।

3. **`label_mode`** → `"binary"`

   * Binary classification (dog vs cat) হলে এইটা দরকার।
   * লেবেল হবে `0` বা `1` float মানে, যা `binary_crossentropy` loss-এর জন্য উপযুক্ত।

---

## **ইমেজ সাইজ**

4. **`image_size`**

   * সব ছবিকে একই সাইজে রিসাইজ করবে।
   * ডিফল্ট `(256,256)` ঠিক আছে, কিন্তু চাইলে ছোটও করতে পারো (যেমন `(128,128)` ট্রেনিং দ্রুত করার জন্য)।

---

## **উদাহরণ কোড**

```python
from tensorflow.keras.utils import image_dataset_from_directory

train_ds = image_dataset_from_directory(
    "dataset/",
    labels="inferred",
    label_mode="binary",
    image_size=(256, 256),
    batch_size=32
)
```

---

নিচে **Dog vs Cat** ক্লাসিফিকেশনের জন্য শুরু থেকে শেষ পর্যন্ত **ফুল রানযোগ্য কোড** দিলাম।
এতে আছে: ডেটা লোড (`image_dataset_from_directory`), train/validation split, প্রি-প্রসেসিং (rescale + augmentation), CNN মডেল, callbacks, ট্রেনিং, ইভ্যালুয়েশন, আর একক ছবিতে প্রেডিকশন—সব এক জায়গায়।

> **ডিরেক্টরি স্ট্রাকচার (আবশ্যিক):**
>
> ```
> dataset/
>   cats/
>     cat1.jpg
>     cat2.jpg
>     ...
>   dogs/
>     dog1.jpg
>     dog2.jpg
>     ...
> ```
>
> তারপর এই স্ক্রিপ্টে `DATA_DIR = "dataset"` সেট করলেই চলবে।

---

```python
# dog_cat_classifier.py
# Keras/TensorFlow 2.x/3 compatible (tf.keras + keras)
# ----------------------------------------------------

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.utils import image_dataset_from_directory, load_img, img_to_array

# ----------------------------
# 0) Basic config (paths & params)
# ----------------------------
DATA_DIR = "dataset"         # <-- আপনার ডেটাসেট ফোল্ডার
IMG_SIZE = (224, 224)        # ইমেজ রিসাইজ সাইজ (ছোট নিলে দ্রুত ট্রেনিং)
BATCH_SIZE = 32
VAL_SPLIT = 0.2
SEED = 1337
EPOCHS = 20                  # প্রাথমিকভাবে 15-30 যথেষ্ট
MODEL_DIR = "checkpoints"
MODEL_PATH = os.path.join(MODEL_DIR, "best.keras")  # Keras 3 default format

os.makedirs(MODEL_DIR, exist_ok=True)

# ----------------------------
# 1) Load datasets with split
# ----------------------------
# labels='inferred' -> সাব-ফোল্ডারের নাম থেকে লেবেল হবে
# label_mode='binary' -> 0/1 লেবেল, binary_crossentropy এর জন্য ঠিক
train_ds = image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="binary",
    validation_split=VAL_SPLIT,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_ds = image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="binary",
    validation_split=VAL_SPLIT,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,  # validation/test এর জন্য সাধারণত shuffle=False
)

print("Class names (alphabetical):", train_ds.class_names)  # ['cats', 'dogs'] এই ক্রমে হলে লেবেল: cats->0, dogs->1

# ----------------------------
# 2) tf.data pipeline optimize
# ----------------------------
AUTOTUNE = tf.data.AUTOTUNE

# Cache + prefetch করলে ট্রেনিং দ্রুত হয়
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds   = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ----------------------------
# 3) Preprocessing & Augmentation
# ----------------------------
# Rescaling: 0-255 -> 0-1
rescale = layers.Rescaling(1./255)

# Augmentation: ট্রেনিং ডেটায় কৃত্রিম বৈচিত্র্য
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
], name="augmentation")

# ----------------------------
# 4) Build CNN model
# ----------------------------
def build_model(input_shape=IMG_SIZE + (3,)):
    inputs = keras.Input(shape=input_shape)

    x = rescale(inputs)
    x = data_augmentation(x)

    # Block 1
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    # Block 2
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    # Block 3
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    # Optional regularization
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    # Binary classification: sigmoid
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="dog_cat_cnn")
    return model

model = build_model()
model.summary()

# ----------------------------
# 5) Compile
# ----------------------------
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[
        keras.metrics.BinaryAccuracy(name="acc"),
        keras.metrics.AUC(name="auc"),
        keras.metrics.Precision(name="prec"),
        keras.metrics.Recall(name="rec"),
    ],
)

# ----------------------------
# 6) Callbacks
# ----------------------------
callbacks = [
    keras.callbacks.ModelCheckpoint(
        MODEL_PATH, monitor="val_acc", mode="max",
        save_best_only=True, verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_acc", mode="max",
        patience=5, restore_best_weights=True, verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, verbose=1
    ),
]

# ----------------------------
# 7) Train
# ----------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# ----------------------------
# 8) Evaluate
# ----------------------------
print("\nEvaluating best model on validation set:")
best_model = keras.models.load_model(MODEL_PATH)
val_metrics = best_model.evaluate(val_ds, verbose=2)
print("Validation metrics:", dict(zip(best_model.metrics_names, val_metrics)))

# ----------------------------
# 9) Single image inference helper
# ----------------------------
def predict_single_image(img_path):
    """
    একক ছবি থেকে প্রেডিকশন: 0 ~ cat, 1 ~ dog (alphabetical class_names অনুসারে)
    """
    img = load_img(img_path, target_size=IMG_SIZE)         # PIL image
    arr = img_to_array(img)                                # -> (H, W, 3)
    arr = arr / 255.0                                      # rescale (same as layer)
    arr = np.expand_dims(arr, axis=0)                      # (1, H, W, 3)

    prob = best_model.predict(arr, verbose=0)[0][0]        # sigmoid prob
    cls_idx = int(prob >= 0.5)
    class_names = train_ds.class_names                     # ['cats', 'dogs'] expected

    # Probability nicely formatted
    label = class_names[cls_idx]
    confidence = prob if cls_idx == 1 else (1 - prob)

    print(f"Image: {img_path}")
    print(f"Predicted: {label} | Confidence: {confidence:.3f} (dog-prob={prob:.3f})")
    return label, float(confidence)

# উদাহরণ (ইচ্ছামতো পাথ বদলে নিন):
# predict_single_image("dataset/cats/cat123.jpg")
# predict_single_image("dataset/dogs/dog456.jpg")
```

---

## রান করার টিপস

* **GPU থাকলে** ট্রেনিং দ্রুত হবে (Colab/Local CUDA)।
* `IMG_SIZE` ছোট করলে (যেমন `(160,160)` বা `(128,128)`) ট্রেনিং দ্রুত, কিন্তু একুরেসি একটু কম হতে পারে।
* ডেটা কম হলে `EPOCHS` 15–30 রাখুন, `EarlyStopping` আছে—খারাপ হলে নিজে থেমে যাবে।
* `train_ds.class_names` প্রিন্ট দেখে নিন—সাধারণত `['cats', 'dogs']` থাকে। এতে **cats→0, dogs→1** ম্যাপিং নিশ্চিত হয়।
* যদি overfitting দেখা যায়: `Dropout` বাড়ান, Augmentation বাড়ান, বা `ReduceLROnPlateau` ব্যবহার করুন (ইতিমধ্যেই আছে)।




