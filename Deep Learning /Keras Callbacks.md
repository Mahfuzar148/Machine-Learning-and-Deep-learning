
---

# Keras Callbacks — ফুল গাইড (বাংলায়)

## Callback কী?

**Callback** হলো এমন অবজেক্ট যা **ট্রেনিং চলার সময়** (batch/epoch শুরু–শেষে, বা কোনো শর্ত পূরণ হলে) কিছু **অটোমেটিক কাজ** করে দেয়:

* সেরা মডেল **save** করা
* **early stop** করা
* **learning rate** কমানো/বদলানো
* **লগিং/গ্রাফ** তৈরি
* **ক্র্যাশ থেকে রিকভারি** ইত্যাদি

কোথায় দিই? → `model.fit(..., callbacks=[...])`

---

## Callback কিভাবে ট্রিগার হয় (লাইফসাইকেল)

Keras অভ্যন্তরে এই মেথডগুলো ডাকে:

* `on_train_begin / on_train_end`
* `on_epoch_begin / on_epoch_end`
* `on_batch_begin / on_batch_end`
* `on_test_begin / on_test_end`, `on_predict_begin / on_predict_end`

তুমি চাইলে **কাস্টম callback** বানিয়ে এগুলোর যেকোনোটা ওভাররাইড করতে পারো।

---

## দ্রুত শুরু (Most useful 3)

### 1) ModelCheckpoint — “সেরা মডেল সেভ”

```python
from keras.callbacks import ModelCheckpoint

ckpt = ModelCheckpoint(
    filepath="checkpoints/best.keras",   # Keras v3 ফরম্যাট
    monitor="val_loss",                  # কোন মেট্রিক দেখবে
    mode="min",                          # কম হলে ভালো
    save_best_only=True,                 # শুধু সেরাটা সেভ
    verbose=1
)
```

### 2) EarlyStopping — “অতিরিক্ত ট্রেনিং বন্ধ”

```python
from keras.callbacks import EarlyStopping

early = EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=5,              # 5 epoch উন্নতি না হলে থামো
    restore_best_weights=True,
    verbose=1
)
```

### 3) ReduceLROnPlateau — “উন্নতি থামলে LR কমাও”

```python
from keras.callbacks import ReduceLROnPlateau

rlrop = ReduceLROnPlateau(
    monitor="val_loss",
    mode="min",
    factor=0.5,      # LR অর্ধেক
    patience=2,      # 2 epoch উন্নতি না হলে
    min_lr=1e-6,
    verbose=1
)
```

### একসাথে ব্যবহার

```python
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=[ckpt, early, rlrop]
)
```

---

## আরও দরকারি Built-in Callbacks (কবে ব্যবহার করবেন)

### TensorBoard — “গ্রাফ/স্কেলার/ইমেজ লগ”

```python
from keras.callbacks import TensorBoard

tb = TensorBoard(
    log_dir="logs/run1",
    histogram_freq=1,  # weights histogram প্রতি epoch
    write_graph=True
)
# টার্মিনাল: tensorboard --logdir logs/run1
```

**কখন**: ট্রেনিং কার্ভ, লেয়ারগুলোর ডিস্ট্রিবিউশন দেখতে।

---

### CSVLogger — “লগ CSV ফাইলে”

```python
from keras.callbacks import CSVLogger
csv = CSVLogger("training_log.csv", append=True)
```

**কখন**: পরবর্তীতে pandas দিয়ে বিশ্লেষণ/প্লট করতে।

---

### LearningRateScheduler — “নিজের নিয়মে LR”

```python
from keras.callbacks import LearningRateScheduler

def step_decay(epoch, lr):
    # প্রতি 5 epoch এ LR অর্ধেক
    return lr * 0.5 if (epoch+1) % 5 == 0 else lr

lrs = LearningRateScheduler(step_decay, verbose=1)
```

**কখন**: কাস্টম LR নীতি (warmup, cosine, ইত্যাদি) লাগলে।

---

### TerminateOnNaN — “NaN দেখলেই থামাও”

```python
from keras.callbacks import TerminateOnNaN
stop_on_nan = TerminateOnNaN()
```

**কখন**: লস NaN হলে সঙ্গে সঙ্গে থামাতে।

---

### BackupAndRestore — “ক্র্যাশ-প্রুফ ট্রেনিং”

```python
from keras.callbacks import BackupAndRestore
backup = BackupAndRestore(backup_dir="backup_ckpts")  # ক্র্যাশ হলে resume হবে
```

**কখন**: লং-রান ট্রেনিং, ক্লাউড/প্রি-এম্পটিবল VM।

---

### ProgbarLogger (ডিফল্ট) — “প্রোগ্রেস বার”

সাধারণত নিজে যোগ দিতে হয় না, Keras নিজেই দেখায়।

---

## মিনিমাল এন্ড-টু-এন্ড উদাহরণ (Dog–Cat)

```python
from keras import layers, models, optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.utils import image_dataset_from_directory
import tensorflow as tf

# 1) ডেটা
train_ds = image_dataset_from_directory(
    "dataset",
    labels="inferred",
    label_mode="binary",
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=(224, 224),
    batch_size=32,
)
val_ds = image_dataset_from_directory(
    "dataset",
    labels="inferred",
    label_mode="binary",
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=(224, 224),
    batch_size=32,
    shuffle=False
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds   = val_ds.cache().prefetch(AUTOTUNE)

# 2) মডেল
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(224,224,3)),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.Conv2D(32, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dropout(0.3),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer=optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# 3) Callbacks
ckpt  = ModelCheckpoint("checkpoints/best.keras", monitor="val_accuracy",
                        mode="max", save_best_only=True, verbose=1)
early = EarlyStopping(monitor="val_accuracy", mode="max",
                      patience=5, restore_best_weights=True, verbose=1)
rlrop = ReduceLROnPlateau(monitor="val_loss", mode="min",
                          factor=0.5, patience=2, min_lr=1e-6, verbose=1)
csv   = CSVLogger("training_log.csv", append=True)

# 4) Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=25,
    callbacks=[ckpt, early, rlrop, csv]
)

# 5) Evaluate best
best = models.load_model("checkpoints/best.keras")
best.evaluate(val_ds, verbose=2)
```

---

## কাস্টম Callback (নিজের নিয়ম বানাও)

উদাহরণ: **target accuracy** পেলে ট্রেনিং থামাও, আর **LR** প্রিন্ট করো।

```python
from keras.callbacks import Callback
import tensorflow as tf

class StopAtTargetAcc(Callback):
    def __init__(self, target=0.95):
        super().__init__()
        self.target = target

    def on_epoch_end(self, epoch, logs=None):
        acc = logs.get("val_accuracy")
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        print(f"\n[epoch {epoch+1}] val_acc={acc:.4f}, lr={lr:.6f}")
        if acc is not None and acc >= self.target:
            print(f"Target {self.target:.2%} reached. Stopping training.")
            self.model.stop_training = True

# ব্যবহার:
# model.fit(..., callbacks=[StopAtTargetAcc(0.95)])
```

---

## কোনটা কখন ব্যবহার করবেন? (চিটশিট)

* **ModelCheckpoint**: সেরা মডেল দরকার/ইনফারেন্স করবে—সবসময় রাখো
* **EarlyStopping**: ওভারফিটিং/সময় বাঁচাতে—প্রায়ই রাখো
* **ReduceLROnPlateau**: ভ্যালিডেশন উন্নতি আটকে গেলে—ভালো ফল দেয়
* **TensorBoard**: গ্রাফ/স্কেলার/হিস্টোগ্রাম দেখতে—ডিবাগ+প্রেজেন্টেশন
* **CSVLogger**: মেট্রিক্স CSV তে—পরে প্লট/বিশ্লেষণ
* **LearningRateScheduler**: কাস্টম LR নীতি—এডভান্সড কেস
* **BackupAndRestore**: ক্র্যাশ-প্রুফ—লং-রান/ক্লাউড
* **TerminateOnNaN**: লস NaN হলে থামাও—সেফটি

---

## কমন ভুল ও টিপস

* **`monitor`/`mode` মিসম্যাচ**:

  * `val_loss` হলে **mode="min"**
  * `val_accuracy` হলে **mode="max"**
* **`patience` খুব ছোট**: মডেল উন্নতি করার সময় পায় না
* **LR খুব বড়**: NaN/ডাইভার্জ—`ReduceLROnPlateau` বা `Scheduler` দিয়ে কমাও
* **Checkpoint path** সঠিক নেই: ফোল্ডার তৈরি করে নাও (`os.makedirs`)
* **TensorBoard** চালাও: `tensorboard --logdir logs/`

---

## 60-সেকেন্ড রেসিপি (Plug-n-Play)

```python
callbacks = [
    ModelCheckpoint("checkpoints/best.keras", monitor="val_accuracy",
                    mode="max", save_best_only=True, verbose=1),
    EarlyStopping(monitor="val_accuracy", mode="max",
                  patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", mode="min",
                      factor=0.5, patience=2, min_lr=1e-6, verbose=1),
]
model.fit(train_ds, validation_data=val_ds, epochs=30, callbacks=callbacks)
```

---
