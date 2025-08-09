
---

## 1) `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`

### কী করে

* **কোন ডিভাইসে (GPU/CPU)** টেনসর ও মডেল রাখবে, সেটা নির্ধারণ করে।
* `torch.cuda.is_available()` → `True` হলে **CUDA GPU** ব্যবহার করবে, নইলে **CPU**।

### কেন দরকার

* GPU থাকলে গণনা (বিশেষত ডিপ লার্নিং) **অনেক দ্রুত** হয়।
* একই কোড CPU/GPU—দুই জায়গায়ই চলতে পারে; শুধু টেনসর/মডেলকে সঠিক **device**-এ পাঠাতে হয়।

### ছোট উদাহরণ

```python
x = torch.randn(2, 3)     # ডিফল্টে CPU-তে
x = x.to(device)          # device অনুযায়ী GPU/CPU-তে যাবে
```

### কমন ভুল

* মডেল GPU-তে কিন্তু ডেটা CPU-তে (বা উল্টো) থাকলে **RuntimeError (device mismatch)** হবে।
  ⇒ সবসময় `data, target = data.to(device), target.to(device)` ব্যবহার করো।

---

## 2) `model = DigitClassifier().to(device)`

### কী করে

* `DigitClassifier()` দিয়ে মডেল বানায়।
* `.to(device)` → মডেলের সব **শিক্ষণযোগ্য প্যারামিটার** (weights/bias) নির্ধারিত ডিভাইসে নিয়ে যায়।

### কেন দরকার

* ইনপুট টেনসর ও মডেল **একই ডিভাইসে** না থাকলে অপারেশন হবে না।

### মনে রাখবে

* ট্রেনিং লুপে প্রতিটি ব্যাচের `data, labels` অবশ্যই `.to(device)` করতে হবে।

---

## 3) `criterion = nn.CrossEntropyLoss()`

### কী করে

* **Classification loss** নির্ধারণ করে।
* `CrossEntropyLoss` = **LogSoftmax + NLLLoss** (কম্বো) → ইনপুটে **raw logits** চায়, আলাদা **Softmax দরকার নেই**।

### ইনপুট ফরম্যাট

* **model output (logits):** shape `[batch_size, num_classes]`
* **target labels (int):** shape `[batch_size]` (one-hot নয়)

### কেন দরকার

* প্রেডিকশন vs সত্য লেবেল—এর **ত্রুটি (loss)** মাপে।
* ট্রেনিংয়ের লক্ষ্য: **loss কমানো**।

### কমন ভুল

* আগে Softmax দিয়ে logits → probability বানিয়ে `CrossEntropyLoss`-এ দিলে **ডাবল Softmax**/স্ট্যাবিলিটি ইস্যু হয়।
  ⇒ **logits সরাসরি** `criterion(output, target)`-এ দাও।

---

## 4) `optimizer = optim.Adam(model.parameters(), lr=0.001)`

### কী করে

* **Optimizer** লসের গ্রেডিয়েন্ট দেখে মডেলের **weights** আপডেট করে।
* `Adam` (Adaptive Moment Estimation): লার্নিং রেটকে গ্রেডিয়েন্টের মোমেন্ট (mean/variance) অনুযায়ী **এডাপ্টিভলি** সমন্বয় করে—সাধারণত **ফাস্ট কনভার্জেন্স**।

### মূল প্যারামিটার

* `model.parameters()` → শেখার যোগ্য সব প্যারামিটার।
* `lr=0.001` → **লার্নিং রেট** (শেখার গতি)

  * বড় `lr` → দ্রুত কিন্তু **unstable** হতে পারে
  * ছোট `lr` → স্থিতিশীল কিন্তু **স্লো** (plateau হলে কমিয়ে দাও)

### আরও দরকারি হাইপারপ্যারামিটার (প্রয়োজনে)

* `betas=(0.9, 0.999)` → প্রথম/দ্বিতীয় মোমেন্টের জন্য ডিকেই।
* `weight_decay` → L2 রেগুলারাইজেশন (ওভারফিটিং কমাতে সাহায্য করে; উদাহরণ: `1e-4`)

### ট্রেনিং সাইকেল (৩ ধাপ)

1. `optimizer.zero_grad()` → পুরনো গ্রেডিয়েন্ট মুছে ফেলো
2. `loss.backward()` → নতুন গ্রেডিয়েন্ট হিসাব
3. `optimizer.step()` → প্যারামিটার আপডেট

---

## 5) একসাথে কীভাবে কাজ করে? (মানসিক মডেল)

1. **device সেট** → CPU/GPU নির্ধারণ
2. **model.to(device)** → মডেল সঠিক ডিভাইসে
3. **forward pass** → `output = model(data)` (logits)
4. **loss** → `loss = criterion(output, target)`
5. **backward** → `optimizer.zero_grad(); loss.backward()`
6. **update** → `optimizer.step()`
7. **eval** → ভ্যালিডেশনে loss/accuracy মাপো, early stopping/plateau হলে LR কমাও

---

## 6) কখন কোনটা টিউন করবে?

* **LR (`lr`)**

  * স্টার্ট: `1e-3` (Adam)
  * ভ্যাল লস না কমলে LR **কমাও** (`ReduceLROnPlateau`/scheduler)
* **Optimizer পছন্দ**

  * `Adam` → দ্রুত/রবাস্ট; বেশিরভাগ কেসে ভালো স্টার্ট
  * `SGD(momentum=0.9)` → বড় ডেটায় শক্তিশালী, সঠিক LR+schedule দরকার
* **Weight decay**

  * ওভারফিটিং কমাতে: `1e-4`–`5e-4` (ভিশনে কমন)
* **Mixed Precision**

  * GPU-তে দ্রুত/মেমরি-সাশ্রয়ী ট্রেনিং: `torch.cuda.amp.autocast` + `GradScaler`

---

## 7) সাধারণ ভুল/সমাধান (Quick Fix)

* **Device mismatch:**

  * ❌ model GPU-তে, data CPU-তে
  * ✅ `data, target = data.to(device), target.to(device)`
* **CrossEntropy-এর আগে Softmax:**

  * ❌ probability বানিয়ে loss—ডাবল softmax
  * ✅ logits সরাসরি দিন
* **Gradient accumulation ভুলে যাওয়া:**

  * ❌ `optimizer.zero_grad()` না দিলে গ্রেডিয়েন্ট জমে → ভুল আপডেট
  * ✅ প্রতিবার আপডেটের আগে **zero\_grad()**
* **Exploding/NaN loss:**

  * ✅ LR কমাও, gradient clipping (যেমন `clip_grad_norm_`), ডেটা নরমালাইজ চেক

---

## 8) ক্ষুদে উদাহরণ (পড়ার সুবিধার জন্য)

**ট্রেনিং স্টেপ (ছদ্মকোড):**

```
for each batch:
  data, target -> device
  logits = model(data)
  loss = CrossEntropy(logits, target)

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
```

**ইভ্যালুয়েশন স্টেপ:**

```
with torch.no_grad():
  logits = model(val_data)
  preds = argmax(logits, dim=1)
  acc = (preds == val_labels).float().mean()
```

---

## 9) দ্রুত চিটশিট

* `device` → `"cuda"` if available else `"cpu"`
* `model.to(device)` + `data.to(device)` **সবসময়**
* `CrossEntropyLoss` logits নেয়; **Softmax দেবে না**
* `Adam(lr=1e-3)` ভালো শুরু; না কমলে LR কমাও/weight decay দাও
* প্রতি স্টেপে: **zero\_grad → backward → step**

---



---

## 1) দ্রুত চিটশিট — কোন টাস্কে কোন লস?

| টাস্ক                                                 | ইনপুট (মডেল আউটপুট)           | টার্গেট                        | লস ফাংশন                                                                 |
| ----------------------------------------------------- | ----------------------------- | ------------------------------ | ------------------------------------------------------------------------ |
| **মাল্টি-ক্লাস ক্লাসিফিকেশন** (একটা ইমেজ = ১টা ক্লাস) | **logits**: `[N, C]`          | `int` লেবেল: `[N]`             | `nn.CrossEntropyLoss`                                                    |
| **বাইনারি ক্লাসিফিকেশন**                              | **logits**: `[N, 1]` বা `[N]` | `float` 0/1: `[N, 1]` বা `[N]` | `nn.BCEWithLogitsLoss`                                                   |
| **মাল্টি-লেবেল ক্লাসিফিকেশন** (এক ইমেজে একাধিক লেবেল) | **logits**: `[N, K]`          | `float` 0/1: `[N, K]`          | `nn.BCEWithLogitsLoss`                                                   |
| **রিগ্রেশন**                                          | `prediction`: `[N, d]`        | `float`: `[N, d]`              | `nn.MSELoss`, `nn.L1Loss`, `nn.SmoothL1Loss`                             |
| **সিকোয়েন্স-টু-সিকোয়েন্স/CTC**                        | লগ-প্রবাবিলিটি                | টার্গেট সিকোয়েন্স              | `nn.CTCLoss`                                                             |
| **র‍্যাঙ্কিং/মেট্রিক লার্নিং**                        | এমবেডিং/স্কোর                 | রিলেশনশিপ লেবেল                | `nn.TripletMarginLoss`, `nn.MarginRankingLoss`, `nn.CosineEmbeddingLoss` |
| **ডিস্ট্রিবিউশন ম্যাচিং**                             | log-prob                      | prob টার্গেট                   | `nn.KLDivLoss` (ইনপুট log-prob হওয়া চাই)                                 |
| **কাউন্ট/ইভেন্ট রেট**                                 | `prediction`                  | `count`                        | `nn.PoissonNLLLoss`                                                      |

> **logits মানে:** Softmax/Sigmoid দেওয়ার **আগের** কাঁচা স্কোর। `CrossEntropyLoss` ও `BCEWithLogitsLoss` **নিজেই** softmax/sigmoid হ্যান্ডেল করে।

---

## 2) লস ফাংশনসমূহ — কাজ, যখন ব্যবহার, প্যারামিটার

### 2.1 `nn.CrossEntropyLoss`

* **কাজ:** মাল্টি-ক্লাস; logits→softmax→NLL, সব একসাথে।
* **ইনপুট:** `input=[N, C]`, `target=[N]` (int লেবেল)
* **প্যারামিটার:** `weight` (ক্লাস-ওয়েট), `ignore_index`, `label_smoothing`, `reduction`
* **কখন:** single-label multi-class (MNIST, CIFAR-10)
* **ভুল এড়াও:** আগে softmax দিয়ো না।

### 2.2 `nn.BCEWithLogitsLoss`

* **কাজ:** বাইনারি/মাল্টি-লেবেল; logits + sigmoid + BCE।
* **ইনপুট:** `input=[N]` বা `[N, K]`, `target` একই শেপে float 0/1
* **প্যারামিটার:** `pos_weight` (ক্লাস ইম্ব্যাল্যান্সে helpful), `weight`, `reduction`
* **কখন:** multi-label or binary classification।
* **ভুল এড়াও:** আগে sigmoid দিয়ো না; logits-ই দাও।

### 2.3 `nn.NLLLoss`

* **কাজ:** Negative Log Likelihood; **ইনপুট log-prob** চাই।
* **কখন:** যদি নিজে `log_softmax` করে দাও; নইলে `CrossEntropyLoss` ই নাও।

### 2.4 রিগ্রেশন লস

* **`nn.MSELoss`**: স্কোয়ার্ড এরর; আউটলাইয়ারে সেনসিটিভ।
* **`nn.L1Loss`**: অ্যাবসোলিউট এরর; আউটলাইয়ারে রোবাস্ট; কনভার্জ স্লো হতে পারে।
* **`nn.SmoothL1Loss`** (Huber): L1+L2 মিশ্র—রোবাস্ট ও স্মুথ।

  * **প্যারামিটার:** `beta`/`reduction`

### 2.5 `nn.CTCLoss`

* **কাজ:** alignment-free সিকোয়েন্স টাস্ক (ASR, OCR)।
* **ইনপুট:** log-prob `[T, N, C]`, টার্গেট `[N, S]`, সাথে lengths।
* **টিপস:** ঠিক ফরম্যাট/লেংথ না দিলে NaN/inf আসতে পারে।

### 2.6 মেট্রিক/এমবেডিং লস

* **`nn.TripletMarginLoss(margin)`**: anchor–positive কাছাকাছি, anchor–negative দূরে।
* **`nn.MarginRankingLoss(margin)`**: স্কোর a, b; a>b করতে শেখাও।
* **`nn.CosineEmbeddingLoss`**: লেবেল 1 হলে কোসাইন সিমিলারিটি বাড়াও, -1 হলে কমাও।

### 2.7 `nn.KLDivLoss`

* **কাজ:** ডিস্ট্রিবিউশন ম্যাচ; **ইনপুট log-prob**, **টার্গেট prob**।
* **জায়গা:** knowledge distillation ইত্যাদি।

### 2.8 `nn.PoissonNLLLoss`

* **কাজ:** কাউন্ট/রেট-ভিত্তিক মডেলিং (Poisson)।
* **ইনপুট:** rate (কখনো log-rate); `log_input` ফ্ল্যাগ।

> সব লসেই `reduction` আছে: `'mean'` (ডিফল্ট), `'sum'`, `'none'`।

---

## 3) অপ্টিমাইজারসমূহ — কাজ, কখন, মূল প্যারামিটার

### 3.1 `optim.SGD`

* **কাজ:** সবচেয়ে বেসিক; `momentum` দিলে ভালো হয়।
* **প্যারামিটার:** `lr`, `momentum=0.9`, `nesterov=True/False`, `weight_decay`
* **কখন:** বড় ডেটা/ভালো টিউনিং করলে খুব শক্তিশালী (ResNet/ImageNet ক্ল্যাসিক)।

### 3.2 `optim.Adam`

* **কাজ:** Adaptive; দ্রুত কনভার্জেন্স, কম টিউনিং দরকার।
* **প্যারামিটার:** `lr=1e-3` স্টার্ট, `betas=(0.9,0.999)`, `eps=1e-8`, `weight_decay`
* **কখন:** শুরুতে “সেইফ” পছন্দ; NLP/Vison—সবখানেই ভালো।

### 3.3 `optim.AdamW`

* **কাজ:** Adam + **ডিকাপলড weight decay** (রেগুলারাইজেশন সঠিকভাবে কাজ করে)।
* **কখন:** আধুনিক ট্রান্সফরমার/ভিশন—ডিফল্টলি AdamW ইউজ করো।
* **প্যারামিটার:** Adam-এর মতোই + `weight_decay` (1e-2…1e-4 ব্যবহার্য)।

### 3.4 `optim.RMSprop`

* **কাজ:** Adagrad-এর ভ্যারিয়েন্ট; non-stationary লক্ষ্য।
* **কখন:** RNN/পুরনো ভিশন কাজ; আজকাল Adam/AdamW বেশি জনপ্রিয়।

### 3.5 `optim.Adagrad`

* **কাজ:** ইনফরমেটিভ ফিচারে বড় আপডেট; সময়ের সাথে LR কমে।
* **কখন:** স্পার্স ফিচার/কিছু NLP টাস্কে।

### 3.6 `optim.Adadelta`

* **কাজ:** Adagrad-এর উন্নত সংস্করণ; অটোমেটিক স্কেলিং।

### 3.7 `optim.Adamax`

* **কাজ:** Adam-এর infinity norm ভ্যারিয়েন্ট; কিছু কেসে স্টেবল।

### 3.8 `optim.Rprop`

* **কাজ:** শুধুমাত্র গ্রেডিয়েন্টের সাইন ব্যবহার করে; ব্যাচ-লার্নিংয়ে প্রাসঙ্গিক।

### 3.9 `optim.ASGD`

* **কাজ:** Averaged SGD; কিছু থিওরেটিক সুবিধা, বাস্তবে কমন নয়।

### 3.10 `optim.SparseAdam`

* **কাজ:** স্পার্স গ্রেডিয়েন্ট (যেমন sparse embeddings)।

### 3.11 `optim.LBFGS`

* **কাজ:** সেকেন্ড-অর্ডার অ্যাপ্রক্সিমেশন; ছোট মডেল/ফাইনটিউনে মাঝে মাঝে।
* **টিপস:** ক্লোজড-ফর্ম ক্লোশার দরকার, বড় নেটওয়ার্কে ধীর।

> **Weight decay (L2)**: ওভারফিটিং কমাতে। AdamW-তে ডিকাপলড; Adam/SGD-তে কস্টে যোগ হয়।
> **Gradient clipping**: RNN/আদিতে exploding gradients ঠেকাতে (`clip_grad_norm_` / `clip_grad_value_`)।

---

## 4) স্কেডিউলার (লার্নিং রেট বদলানো)

* **`StepLR(step_size, gamma)`**: নির্দিষ্ট ইন্টারভালে LR × `gamma`।
* **`MultiStepLR(milestones, gamma)`**: কয়েকটা মাইলস্টোনে LR কমাও।
* **`ExponentialLR(gamma)`**: ক্রমাগত LR decay।
* **`CosineAnnealingLR(T_max)`**: কসমিক কার্ভে LR কমে।
* **`ReduceLROnPlateau(monitor)`**: ভ্যাল লস/মেট্রিক না কমলে LR কমাও।
* **`OneCycleLR`**: ট্রেনিং জুড়ে এক সাইকেলে LR বাড়া-কমা; দ্রুত কনভার্জেন্স।

**টিপ:** Adam/AdamW + `ReduceLROnPlateau` বা `CosineAnnealingLR`—দুইটাই ব্যবহার্য কম্বো।

---

## 5) টেমপ্লেট — ট্রেনিং লুপ (লস+অপ্টিমাইজার)

```python
model.train()
for data, target in train_loader:
    data, target = data.to(device), target.to(device)

    logits = model(data)                 # [N, C] or [N, 1]
    loss   = criterion(logits, target)   # e.g., CrossEntropy or BCEWithLogits

    optimizer.zero_grad()                # পুরনো গ্রেডিয়েন্ট ক্লিয়ার
    loss.backward()                      # নতুন গ্রেডিয়েন্ট
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # (optional)
    optimizer.step()                     # আপডেট

# scheduler থাকলে
# scheduler.step(val_loss)  # ReduceLROnPlateau হলে
# scheduler.step()          # অন্যদের ক্ষেত্রে
```

---

## 6) সাধারণ ভুল—দ্রুত সমাধান

* **CrossEntropy + Softmax** (ডাবল softmax) → logits সরাসরি `CrossEntropyLoss`-এ দাও।
* **BCE দু’বার Sigmoid** → `BCEWithLogitsLoss` ইউজ করো, আগে sigmoid কোরো না।
* **Shape mismatch** → `input=[N,C]`, `target=[N]` (int); BCE-তে দুইটার শেপ মিলাও।
* **Device mismatch** → `data, target, model` একই device-এ।
* **লার্নিং রেট বড়/ছোট** → `lr` টিউন + স্কেডিউলার।
* **ইম্ব্যালান্সড ক্লাস** → `class_weight`/`pos_weight`, **WeightedRandomSampler**, Focal-like লস (নিজে ইমপ্লিমেন্ট/থার্ড-পার্টি)।

---

## 7) “কখন কী নেব?” — ব্যবহারিক গাইড

* **শুরুতে:** `AdamW(lr=1e-3, weight_decay=1e-4)` + `CrossEntropy`/`BCEWithLogits`
* **বড় ভিশন ক্লাসিফিকেশন:** `SGD(momentum=0.9, lr=0.1)` with cosine/step LR (সঠিক টিউন দরকার)
* **NLP/Transformer ফাইন-টিউন:** `AdamW(lr=2e-5 ~ 5e-5, weight_decay=0.01)` + `LinearWarmup + Cosine`
* **রিগ্রেশন:** `MSELoss` (নয়েজি/আউটলাইয়ার হলে `SmoothL1Loss`)
* **মাল্টি-লেবেল:** `BCEWithLogitsLoss` (প্রতি লেবেলে স্বাধীন sigmoid)
* **সিকোয়েন্স আলাইনমেন্ট:** `CTCLoss`
* **র‍্যাঙ্কিং/রিট্রিভাল:** `TripletMarginLoss` / `MarginRankingLoss`

---

## 8) মিনিমাল কনফিগ সাজেশন (MNIST উদাহরণ)

* **Loss:** `nn.CrossEntropyLoss()`
* **Optimizer:** `optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)`
* **Scheduler:** `ReduceLROnPlateau` (monitor: val\_loss) বা `CosineAnnealingLR(T_max=epochs)`
* **Extras:** গ্রেডিয়েন্ট ক্লিপ (ঐচ্ছিক), Early stopping

---

