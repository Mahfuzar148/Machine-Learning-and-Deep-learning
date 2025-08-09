
---

# Training Loop — কোড + লাইনে–লাইনে ব্যাখ্যা

```python
def train_one_epoch(model, device, train_loader, optimizer, criterion, log_interval=100):
    model.train()                                    # 1
    running_loss = 0.0                                # 2
    total, correct = 0, 0                             # 3

    for batch_idx, (data, targets) in enumerate(train_loader):  # 4
        data, targets = data.to(device), targets.to(device)     # 5

        optimizer.zero_grad()                         # 6
        outputs = model(data)                         # 7
        loss = criterion(outputs, targets)            # 8
        loss.backward()                               # 9
        optimizer.step()                              # 10

        running_loss += loss.item()                   # 11

        # (ঐচ্ছিক) ট্রেনিং অ্যাকুরেসি ট্র্যাক করা
        preds = outputs.argmax(dim=1)                 # 12
        correct += (preds == targets).sum().item()    # 13
        total   += targets.size(0)                    # 14

        if (batch_idx % log_interval) == 0:          # 15
            print(f"Batch {batch_idx:5d}/{len(train_loader):5d} | "
                  f"Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(train_loader)       # 16
    train_acc = (correct / total) if total else 0.0   # 17
    return avg_loss, train_acc                        # 18
```

---

## লাইনে–লাইনে ব্যাখ্যা

### 1) `model.train()`

* মডেলকে **training mode**-এ দেয়।
* **Dropout** সক্রিয় হয় (র‍্যান্ডম ইউনিট ড্রপ), **BatchNorm** ব্যাচ-স্ট্যাটস আপডেট করে।
* এটা না দিলে মডেল শেখা ঠিকমতো নাও হতে পারে (eval আচরণ থাকবে)।

### 2) `running_loss = 0.0`

* ইপকের সব ব্যাচের লস **জমা** রাখবে। শেষে গড় লস দেখানো হবে।
* ট্রেনিং প্রগ্রেস বোঝার জন্য সহায়ক।

### 3) `total, correct = 0, 0`

* (ঐচ্ছিক) ট্রেনিং অ্যাকুরেসি হিসাবের জন্য কাউন্টার।
* অনেকেই ট্রেনিং-এ অ্যাকুরেসি না দেখিয়ে কেবল ভ্যালিডেশনে দেখে—দুটোই ঠিক।

### 4) `for batch_idx, (data, targets) in enumerate(train_loader):`

* `DataLoader` থেকে **ব্যাচ** ধরে ডেটা ও টার্গেট নেয়।
* `batch_size` বড় হলে ইটারেশন কমে (দ্রুত), কিন্তু **GPU RAM** বেশি লাগে। ছোট হলে উল্টোটা।

### 5) `data, targets = data.to(device), targets.to(device)`

* মডেল যেখানে আছে (CPU/GPU), ডেটাও সেখানে পাঠাও।
* **device mismatch** হলে RuntimeError হবে।

### 6) `optimizer.zero_grad()`

* আগের স্টেপের **গ্রেডিয়েন্ট ক্লিয়ার** করে।
* এটা না করলে গ্রেডিয়েন্ট জমে ভুল আপডেট হবে (gradient accumulation ছাড়া সবসময় দরকার)।

### 7) `outputs = model(data)`

* **forward pass**: ইনপুট থেকে প্রেডিকশন/লজিটস।
* শেপ সঠিক কিনা দেখ—যেমন CrossEntropy হলে `[N, C]`।

### 8) `loss = criterion(outputs, targets)`

* লস ফাংশন (যেমন **CrossEntropyLoss**) দিয়ে ত্রুটি গণনা।
* **CrossEntropy/BCEWithLogits**: logits দাও, **আগে softmax/sigmoid দেবে না**।

### 9) `loss.backward()`

* **backward pass**: গ্রেডিয়েন্ট হিসাব করে প্রতিটি trainable প্যারামিটারে জমা করে।
* **Exploding gradient** হলে পরে clipping করা যায় (নীচে টেমপ্লেট আছে)।

### 10) `optimizer.step()`

* গ্রেডিয়েন্ট দিয়ে **ওজন আপডেট**।
* Adam/SGD ইত্যাদির আচরণ আলাদা; **LR** বড় হলে অস্থির/NaN, ছোট হলে ধীর।

### 11) `running_loss += loss.item()`

* টেনসর → scalar float; গড় লস বের করতে যোগ করা হচ্ছে।

### 12) `preds = outputs.argmax(dim=1)`

* সবচেয়ে বড় স্কোরের **ক্লাস ইনডেক্স** (predicted label)।
* `dim` ভুল দিলে ভুল অ্যাকুরেসি আসবে।

### 13) `correct += (preds == targets).sum().item()`

* এই ব্যাচে কতগুলো সঠিক হয়েছে **গণনা**।

### 14) `total += targets.size(0)`

* মোট স্যাম্পলের **সংখ্যা** যোগ করা—accuracy বের করতে দরকার।

### 15) `if (batch_idx % log_interval) == 0: ...`

* প্রতি `log_interval` ব্যাচ পরপর প্রগ্রেস প্রিন্ট।
* খুব ঘন ঘন প্রিন্ট করলে I/O ওভারহেডে ট্রেনিং ধীর হতে পারে; খুব কম দিলে প্রগ্রেস বোঝা কঠিন।
* **টিউন:** ছোট ডেটায় `20/50`, বড় ডেটায় `200/500`।

### 16) `avg_loss = running_loss / len(train_loader)`

* পুরো ইপকের **গড় ট্রেনিং লস**।
* ইপক→ইপক `avg_loss` কমতে থাকা স্বাস্থ্যকর; স্থির/বাড়তে থাকলে LR/ডেটা/মডেল চেক করো।

### 17) `train_acc = (correct / total) if total else 0.0`

* (ঐচ্ছিক) **ট্রেনিং অ্যাকুরেসি**।
* নোট: শুধু ট্রেনিং অ্যাকুরেসির উপর ভরসা কোরো না—**ভ্যালিডেশন** অ্যাকুরেসি/লসই জেনারালাইজেশন দেখায়।

### 18) `return avg_loss, train_acc`

* কলার ফাংশনে গড় লস ও ট্রেনিং অ্যাকুরেসি ফেরত দিচ্ছি, যাতে লগ/স্কেডিউলার/আর্লি-স্টপিংয়ে ব্যবহার করা যায়।

---

## কী টিউন করলে কী প্রভাব

* **`log_interval`**: কমালে ঘন ঘন লগ—ডিবাগে সুবিধা, কিন্তু ধীর; বাড়ালে দ্রুত, কম তথ্য।
* **`optimizer`**: `AdamW(lr=1e-3, wd=1e-4)` ভালো শুরু; বড় ভিশনে `SGD(momentum=0.9)` + LR schedule।
* **`lr` (লার্নিং রেট)**: বড় → অস্থির/NaN; ছোট → স্লো/প্লেটো। LR scheduler খুব কাজে দেয়।
* **`batch_size`**: বড় → দ্রুত/স্ট্যাবল গ্রেডিয়েন্ট (RAM বেশি লাগে); ছোট → ধীর, কিন্তু কখনো ভালো জেনারালাইজ।

---

## বোনাস: প্রো-টেমপ্লেট (AMP, Gradient Clipping, Scheduler)

```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()  # AMP

def train_one_epoch_amp(model, device, train_loader, optimizer, criterion,
                        log_interval=100, max_grad_norm=None, scheduler=None):
    model.train()
    running_loss, total, correct = 0.0, 0, 0

    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()

        with autocast():                         # AMP on
            outputs = model(data)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()            # scaled backward

        if max_grad_norm is not None:            # gradient clipping (optional)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        scaler.step(optimizer)                   # optimizer.step() with AMP
        scaler.update()

        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total   += targets.size(0)

        if (batch_idx % log_interval) == 0:
            print(f"Batch {batch_idx:5d}/{len(train_loader):5d} | Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(train_loader)
    train_acc = (correct / total) if total else 0.0

    if scheduler is not None:                    # e.g., ReduceLROnPlateau → scheduler.step(val_loss) বাইরে
        try:
            scheduler.step()                     # Cosine/Step ইত্যাদি হলে এখানে
        except TypeError:
            pass

    return avg_loss, train_acc
```

**কখন ব্যবহার করবে?**

* **AMP**: GPU-তে দ্রুত ও মেমরি-সাশ্রয়ী ট্রেনিং।
* **Clipping**: RNN/Transformer/অস্থির ট্রেনিং—`max_grad_norm=1.0` দিয়ে শুরু করো।
* **Scheduler**: Cosine/Step হলে ইপক শেষে `.step()`; **ReduceLROnPlateau** হলে **ভ্যালিডেশন লস** দিয়ে বাইরে `.step(val_loss)`।

---

## কমন ভুল/সমাধান (Quick Fix)

* **device mismatch** → data/model এক ডিভাইসে `.to(device)`।
* **zero\_grad() মিস** → গ্রেডিয়েন্ট জমে—প্রতি ব্যাচে দাও।
* **CrossEntropy + Softmax** → ডাবল softmax; logits-ই দাও।
* **BCEWithLogits + Sigmoid** → দু’বার sigmoid; logits-ই দাও।
* **Shape mismatch** → CE: outputs `[N, C]`, targets `[N]`; BCE: দুটোই `[N, K]` float 0/1।
* **NaN/Exploding** → LR কমাও, clipping দাও, নরমালাইজ/ডেটা চেক।

---

