
```python

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------
# 1) Gradient Reversal
# ------------------------
class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_out):
        return -ctx.alpha * grad_out, None

def grl(x, alpha=1.0):
    return GradientReversalFn.apply(x, alpha)


# ------------------------
# 2) Backbones / Heads
# ------------------------
class FeatureExtractor(nn.Module):
    """Very small CNN backbone for 1x28x28 or 3x32x32-like images.
       আপনার ডেটা অনুযায়ী চ্যানেল/সাইজ অ্যাডজাস্ট করুন।
    """
    def __init__(self, in_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, padding=2),  # -> 32xHxW
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                            # H/2, W/2

            nn.Conv2d(32, 48, 5, padding=2),           # -> 48x(H/2)x(W/2)
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                            # H/4, W/4
        )
        # ফিচার ডাইমেনশন নির্ধারণ (MNIST 28x28 ধরলে 48*7*7=2352)
        self.out_dim = 48 * 7 * 7  # আপনার ইনপুট সাইজ আলাদা হলে এটি বদলান

    def forward(self, x):
        f = self.net(x)
        return f.view(x.size(0), -1)  # (B, feat_dim)


class LabelPredictor(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(100, num_classes),
        )

    def forward(self, f):
        return self.head(f)


class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim, num_domains=2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 100),
            nn.BatchNorm1d(100),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(100, num_domains),
        )

    def forward(self, f):
        return self.head(f)


# ------------------------
# 3) DANN (end-to-end)
# ------------------------
class DANN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, num_domains=2):
        super().__init__()
        self.feature_extractor = FeatureExtractor(in_channels=in_channels)
        feat_dim = self.feature_extractor.out_dim
        self.label_predictor = LabelPredictor(feat_dim, num_classes)
        self.domain_discriminator = DomainDiscriminator(feat_dim, num_domains)

        self.num_updates = 0  # alpha schedule এর জন্য

    def alpha(self):
        """Standard DANN schedule: ধীরে ধীরে 0→1 (স্মুথ)"""
        p = min(self.num_updates / 10000.0, 1.0)
        # 2/(1+exp(-10p)) - 1: 0→1 (স্মুথ), scale চাইলে গুণ দিন
        return 2.0 / (1.0 + torch.exp(torch.tensor(-10.0 * p))) - 1.0

    def forward(self, x, *, inference=False):
        f = self.feature_extractor(x)                  # shared features
        class_logits = self.label_predictor(f)         # label head

        if inference:
            # ইনফারেন্সে GRL দরকার নেই
            domain_logits = self.domain_discriminator(f)
            return class_logits, domain_logits, f

        a = float(self.alpha())                        # dynamic alpha
        f_rev = grl(f, a)                              # GRL
        domain_logits = self.domain_discriminator(f_rev)

        self.num_updates += 1
        return class_logits, domain_logits, f


# ------------------------
# 4) Training step (example)
# ------------------------
def dann_train_step(model, batch_source, batch_target, optimizer, device):
    """
    batch_source: dict{ 'x': images, 'y': labels, 'd': domain_ids(0 for source) }
    batch_target: dict{ 'x': images,             'd': domain_ids(1 for target) }
    """
    model.train()
    xs, ys, ds = batch_source['x'].to(device), batch_source['y'].to(device), batch_source['d'].to(device)
    xt, dt     = batch_target['x'].to(device), batch_target['d'].to(device)

    # 1) Label prediction loss (শুধু source লেবেল থাকে)
    cls_logits, dom_logits_s, _ = model(xs, inference=False)
    cls_loss = F.cross_entropy(cls_logits, ys)

    # 2) Domain loss (source + target; দুটোই ডোমেইন লেবেল জানি)
    # source pass already done -> dom_logits_s
    # target pass (শুধু domain head-এর জন্য লাগে)
    with torch.no_grad():
        model.num_updates -= 1  # alpha schedule যেন এক স্টেপে দু'বার না বাড়ে
    _, dom_logits_t, _ = model(xt, inference=False)

    dom_logits = torch.cat([dom_logits_s, dom_logits_t], dim=0)
    dom_labels = torch.cat([ds, dt], dim=0)
    domain_loss = F.cross_entropy(dom_logits, dom_labels)

    # 3) Total loss
    loss = cls_loss + domain_loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss.item()),
        "cls_loss": float(cls_loss.item()),
        "domain_loss": float(domain_loss.item()),
        "alpha": float(model.alpha().item())
    }


# ------------------------
# 5) Usage skeleton
# ------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DANN(in_channels=1, num_classes=10, num_domains=2).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=2e-4)

    # Dummy example (MNIST-like shapes)
    B = 32
    xs = torch.randn(B, 1, 28, 28)  # source images
    ys = torch.randint(0, 10, (B,)) # source labels
    ds = torch.zeros(B, dtype=torch.long)  # domain id = 0

    xt = torch.randn(B, 1, 28, 28)  # target images
    dt = torch.ones(B, dtype=torch.long)   # domain id = 1

    stats = dann_train_step(
        model,
        batch_source={"x": xs, "y": ys, "d": ds},
        batch_target={"x": xt, "d": dt},
        optimizer=optim,
        device=device,
    )
    print(stats)
```

---

### কীভাবে এটা DANN + GRL ফ্লো মেনে চলে

* **Shared FeatureExtractor**: সোর্স/টার্গেট—দুটো ডোমেইনের ইমেজ থেকেই একই ফিচার স্পেসে এমবেড করে।
* **LabelPredictor (source supervised)**: শুধু সোর্সের লেবেল দেওয়া থাকে, তাই `cls_loss` ক্যালকুলেট হয় সোর্সেই।
* **DomainDiscriminator (GRL সহ)**:

  * `f_rev = grl(f, alpha)` — backward এ gradient `-alpha` গুণে উল্টে দেয়।
  * Domain head `source+target` উভয় ব্যাচে ট্রেন হয়, আর GRL-এর জন্য **FeatureExtractor** শিখে **domain-invariant** ফিচার বানাতে।
* **Alpha schedule**: শুরুতে ছোট, পরে বড় — ট্রেনিং স্টেবল থাকে, ধীরে ধীরে ডোমেইন ইনভ্যারিয়্যান্স চাপ বাড়ে।
* **Training step**: সোর্সে ক্লাসিফিকেশন, সোর্স+টার্গেটে ডোমেইন ক্লাসিফিকেশন—দুই লস যোগ করে অপ্টিমাইজ করা হয়েছে।





```python
import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):
    """Very small CNN backbone for 1x28x28 or 3x32x32-like images.
       আপনার ডেটা অনুযায়ী in_channels অ্যাডজাস্ট করুন।
    """
    def __init__(self, in_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),  # -> 32 x H x W
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> 32 x (H/2) x (W/2)

            nn.Conv2d(32, 48, kernel_size=5, padding=2),          # -> 48 x (H/2) x (W/2)
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> 48 x (H/4) x (W/4)
        )

        # এখানে MNIST (28x28) ধরে out_dim হার্ডকোড
        self.out_dim = 48 * 7 * 7

    def forward(self, x):
        f = self.net(x)                # আউটপুট: (B, 48, 7, 7)
        return f.view(x.size(0), -1)   # ফ্ল্যাটেন: (B, 2352)
```

---

### ব্যাখ্যা (লাইন ধরে)

* `self.net = nn.Sequential(...)`
  → দুটো কনভ-ব্লক বানানো হয়েছে। প্রতিটা ব্লক: `Conv → BN → ReLU → MaxPool`।

* প্রথম ব্লক:

  * `Conv2d(in_channels, 32, 5, padding=2)` → ইনপুটকে 32 ফিচারম্যাপে কনভ করে, সাইজ অপরিবর্তিত থাকে।
  * `BatchNorm2d(32)` → আউটপুট নর্মালাইজ।
  * `ReLU` → নন-লিনিয়ারিটি।
  * `MaxPool2d(2)` → H, W কে অর্ধেক করে ফেলে।

* দ্বিতীয় ব্লক:

  * আবার Conv (32→48 চ্যানেল), BN, ReLU, MaxPool।
  * ফলে শেষ আউটপুট হয় `(B, 48, H/4, W/4)`।
  * MNIST (28×28) হলে → `(B, 48, 7, 7)`।

* `self.out_dim = 48 * 7 * 7`
  → MNIST সাইজ ধরে ফ্ল্যাটেন ফিচার ডাইমেনশন প্রি-ডিফাইন (2352)।

* `forward`:

  * `f = self.net(x)` → CNN চালানো।
  * `f.view(x.size(0), -1)` → আউটপুট ফ্ল্যাটেন করে `(B, 2352)` বানানো—যেটা fully connected লেয়ারে দেওয়া যাবে।

---

👉 যদি ইনপুট সাইজ MNIST না হয় (যেমন CIFAR-10, 32×32), তাহলে `self.out_dim` নতুন করে ক্যালকুলেট করতে হবে (hardcoded 48*7*7 এর বদলে 48*8*8 হবে)।

---


---

## `forward(self, x)` এর কাজ কী?

PyTorch-এর প্রতিটি `nn.Module`-এ একটা **forward** মেথড থাকে।
👉 এইখানে লিখে দিই **ডেটা মডিউলের ভেতর কিভাবে যাবে**।

তুমি যখন লিখবে:

```python
features = model(images)
```

তখন আসলে ভেতরে `model.forward(images)` কল হয়।
মানে—এখানে CNN ফিচার এক্সট্র্যাকশন আর ফ্ল্যাটেন করা হচ্ছে।

---

## `f = self.net(x)`

* `self.net` হলো `nn.Sequential` এ গঠিত তোমার CNN পাইপলাইন (`Conv → BN → ReLU → Pool → Conv → ...`)।
* `x` ইনপুট টেনসর (শেপ `(B, C, H, W)`) CNN দিয়ে চালিয়ে `f` বানানো হচ্ছে।

উদাহরণ (MNIST, 28×28):

* ইনপুট: `(B, 1, 28, 28)`
* ১ম Conv+Pool শেষে: `(B, 32, 14, 14)`
* ২য় Conv+Pool শেষে: `(B, 48, 7, 7)`
  👉 তাই `f` এর শেপ হয় `(B, 48, 7, 7)`।

---

## `return f.view(x.size(0), -1)`

এখানে দুটি জিনিস হচ্ছে:

1. **`x.size(0)`**

   * `x.size(0)` হলো batch size `B`।
   * আমরা চাই আউটপুটে প্রথম dimension সবসময় batch dimension থাকুক।
   * তাই ফ্ল্যাটেন করার সময়ও `(B, …)` শেপ রাখা হয়েছে।

2. **`-1`**

   * PyTorch-এ `view`-এর `-1` মানে:

     > "এখানকার সাইজটা নিজে নিজে অটোম্যাটিক ক্যালকুলেট করো।"
   * উদাহরণ: `f` এর শেপ `(B, 48, 7, 7)` → ব্যাচ ছাড়া বাকি সাইজ = `48*7*7 = 2352`।
   * তাই `f.view(B, -1)` → `(B, 2352)`।
   * এটা handy কারণ সঠিক সংখ্যাটা হাতে ক্যালকুলেট করতে হয় না, কোড ইনপুট সাইজে ফ্লেক্সিবল থাকে।

---

## সারসংক্ষেপ

* `forward`: মডেলের ভিতরে ডেটা ফ্লো কীভাবে হবে সেটার সংজ্ঞা।
* `self.net(x)`: CNN ব্লক চালিয়ে ফিচার বের করা → `(B, C, H, W)`।
* `.view(x.size(0), -1)`:

  * ব্যাচ ডাইমেনশন আলাদা রাখা (`B`)।
  * বাকি সব dimension ফ্ল্যাট করে ১টাতে রূপান্তর (`2352`)।
    👉 ফলে ফাইনাল আউটপুট হয় `(B, feature_dim)` — যেটা fully connected লেয়ারে feed দেওয়া সহজ হয়।

---



---

### `self.head` কী?

* `self.head` হলো একটা **`nn.Sequential` container**।
* ভেতরে কয়েকটা লেয়ার একটার পর একটা সাজানো আছে:

  ```python
  self.head = nn.Sequential(
      nn.Linear(in_dim, 100),
      nn.ReLU(inplace=True),
      nn.Dropout(0.2),
      nn.Linear(100, num_classes),
  )
  ```
* তাই যখন তুমি `self.head(f)` কল করো, তখন `f` এই লেয়ারগুলো একে একে পেরিয়ে যায়।

---

### ধাপে ধাপে `self.head(f)` এ কী হয়?

ধরা যাক `f.shape = (B, in_dim)`

1. **`nn.Linear(in_dim, 100)`**

   * ফিচার ভেক্টরকে 100 ডাইমেনশনে প্রজেক্ট করে।
   * আউটপুট: `(B, 100)`

2. **`nn.ReLU(inplace=True)`**

   * নন-লিনিয়ারিটি অ্যাপ্লাই করে (negative → 0, positive → same)।
   * আউটপুট এখনো `(B, 100)`।

3. **`nn.Dropout(0.2)`**

   * ট্রেনিং চলাকালে র‍্যান্ডমলি ২০% নিউরনকে "ড্রপ" করবে (০ করে দেবে)।
   * ফলে মডেল ওভারফিট হবে না, জেনারালাইজ ভালো হবে।
   * আউটপুট এখনো `(B, 100)`।

4. **`nn.Linear(100, num_classes)`**

   * ফাইনাল লেয়ার: 100-ডাইমেনশন ভেক্টর থেকে `num_classes` সংখ্যক লজিট আউটপুট করে।
   * যেমন MNIST হলে `num_classes=10` → আউটপুট `(B, 10)`।

---

### সারাংশ

👉 `self.head(f)` মানে:
`f` কে এই চারটা লেয়ারের ভেতর একটার পর একটা চালানো।
ফাইনাল রেজাল্ট = প্রতিটি স্যাম্পলের জন্য **ক্লাস logits** (softmax এর আগে raw স্কোর)।

---


---

## কোড

```python
class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim, num_domains=2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 100),
            nn.BatchNorm1d(100),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(100, num_domains),
        )

    def forward(self, f):
        return self.head(f)
```

---

## ব্যাখ্যা (লাইন ধরে)

* `in_dim` : ফিচার ডাইমেনশন (FeatureExtractor থেকে পাওয়া ভেক্টর)।
* `num_domains` : কয়টা ডোমেইন আলাদা করতে হবে (যেমন Source=0, Target=1 → 2 domain)।

**Layers (`self.head`)**

1. `nn.Linear(in_dim, 100)`

   * ইনপুট ফিচারকে 100-ডাইমেনশন স্পেসে ম্যাপ করে।

2. `nn.BatchNorm1d(100)`

   * এক ব্যাচে নিউরনের আউটপুট নর্মালাইজ করে → ট্রেনিং স্টেবল হয়।

3. `nn.LeakyReLU(0.2, inplace=True)`

   * নন-লিনিয়ার অ্যাক্টিভেশন, কিন্তু শূন্যের নিচে গিয়ে গেলেও ছোট slope (0.2) রাখে।
   * Dead neuron problem কমায়।

4. `nn.Dropout(0.3)`

   * ৩০% নিউরন র‍্যান্ডমলি ড্রপ → ওভারফিটিং কমে।

5. `nn.Linear(100, num_domains)`

   * ফাইনাল আউটপুট → `num_domains` সংখ্যক logit (যেমন 2 → সোর্স বনাম টার্গেট ডোমেইন)।

**Forward Pass:**

* `self.head(f)` ইনপুট `f` কে এই সিকোয়েন্সে চালায়।
* আউটপুট শেপ: `(B, num_domains)`।
* CrossEntropy Loss দিয়ে ট্রেন করানো হয়।

---

## ❓ এখন প্রশ্ন: `self.net` বনাম `self.head`

এটা শুধু নামকরণের ব্যাপার — তুমি চাইলে `self.net` বা `self.head` যেকোনো নাম দিতে পারো।

* **`self.net`** সাধারণত ব্যবহার হয় বড় CNN বা feature extractor ব্লককে বোঝাতে (Conv → BN → Pool ... stack)।
* **`self.head`** সাধারণত বোঝায় শেষের fully-connected/classifier অংশ (MLP, prediction layer ইত্যাদি)।

👉 অর্থাৎ convention হলো:

* **net** → core feature extraction pipeline
* **head** → output head (classification, regression, domain prediction ইত্যাদি)

কোডে কোনও পার্থক্য নেই, কেবল কোড পড়তে সুবিধার জন্য আলাদা নাম রাখা হয়।



---

## 🔹 ১. `self.net`

* সাধারণত `FeatureExtractor` এর মতো মডিউলে ব্যবহার হয়।
* এখানে CNN বা বড় ব্লক থাকে যেটা **ইনপুট (x, ছবি)** থেকে **ফিচার (f)** বের করে।

```python
self.net = nn.Sequential(
    nn.Conv2d(...),
    nn.ReLU(),
    nn.MaxPool2d(...)
)
```

👉 তাই `self.net(x)` মানে হলো → **raw data → feature maps**

---

## 🔹 ২. `self.head`

* সাধারণত `LabelPredictor` বা `DomainDiscriminator` এ ব্যবহার হয়।
* এগুলো **শেষের prediction করার block**।
* মানে feature vector `f` কে ইনপুট নিয়ে → এটাকে **class logits / domain logits** এ রূপান্তর করে।

```python
self.head = nn.Sequential(
    nn.Linear(in_dim, 100),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(100, num_classes)
)
```

👉 তাই `self.head(f)` মানে হলো → **features → final prediction (class scores)**

---

## 🔹 পার্থক্য এক কথায়:

* **`self.net` = Feature Extraction block** (raw input থেকে features বের করা)
* **`self.head` = Prediction block** (features থেকে চূড়ান্ত output দেওয়া)

---

## 📌 উদাহরণ দিয়ে বুঝুন:

ধরা যাক একটা pipeline হলো:

```
Input Image (x)
   ↓
self.net(x)        # CNN দিয়ে features বের হলো
   ↓
f (features)
   ↓
self.head(f)       # Fully Connected Layer দিয়ে class prediction হলো
   ↓
Output (logits)
```

---

## 🧾 কেন আলাদা করা হলো?

* **Modularity (ভাগ করা সুবিধার জন্য):**

  * `self.net` শুধু feature বের করার জন্য।
  * `self.head` শুধু output prediction করার জন্য।
* **Reuse:** একই features দিয়ে ভিন্ন ভিন্ন head লাগানো যায় (যেমন: এক head = class prediction, অন্য head = domain prediction)।
* **Clarity:** বোঝা সহজ হয় কোন অংশ feature extractor আর কোন অংশ classifier।

---

👉 তাই এখানে **`self.head()` নেওয়া হয়েছে কারণ এটা Label Predictor এর শেষের prediction block**।
`self.net()` থাকলে সেটা FeatureExtractor এর অংশ হতো।

---


---

## 🖥️ Model Pipeline

```
Input (x - image / data)
        │
        ▼
   ┌───────────┐
   │ self.net  │   ← Feature Extractor
   └───────────┘
        │
        ▼
 f (features vector)
        │
        ▼
   ┌───────────┐
   │ self.head │   ← Prediction Head (Classifier / Discriminator)
   └───────────┘
        │
        ▼
 Output (class logits / domain logits)
```

---

## 📌 সংক্ষেপে

* **`self.net(x)`** → Raw data (image) থেকে **features (f)** বের করে।
* **`self.head(f)`** → Features (f) থেকে **চূড়ান্ত আউটপুট (prediction)** তৈরি করে।

---

👉 এইভাবে মডেলকে দুই ভাগে ভাঙা হয়:

1. **Feature Extractor (`self.net`)**
2. **Prediction Head (`self.head`)**

---


---

### উদাহরণ

* `FeatureExtractor` → `self.net` নাম দিয়েছি, কারণ ওটা পুরো CNN নেটওয়ার্ক।
* `LabelPredictor` ও `DomainDiscriminator` → `self.head`, কারণ ওগুলো সোজা classifier “হেড”।

---

👉 সহজভাবে:

* Feature extraction / main pipeline → `self.net`
* Prediction / শেষ অংশ → `self.head`

---


---

## ফ্লোটা এভাবে হয় 👇

1. **FeatureExtractor (self.net)**

   * ইনপুট ইমেজ `(B, C, H, W)` কে কনভ-লেয়ার চালিয়ে ফ্ল্যাট ফিচার ভেক্টরে কনভার্ট করে।
   * আউটপুট: `f.shape = (B, in_dim)`

     * MNIST এর জন্য `in_dim=2352` (48×7×7)।
     * CIFAR এর জন্য `in_dim=48×8×8=3072`।

---

2. **LabelPredictor (self.head)**

   * ফিচার ভেক্টর `f` ইনপুট নেয়।
   * আউটপুট: **ক্লাস logits** (যেমন MNIST এ 10 ক্লাস → `(B, 10)`)।
   * লস: `CrossEntropy(class_logits, true_labels)`

---

3. **DomainDiscriminator (self.head)**

   * ফিচার ভেক্টর `f` ইনপুট নেয়, তবে GRL (Gradient Reversal Layer) এর পর।

     ```python
     f_rev = grad_reverse(f, alpha)
     domain_logits = DomainDiscriminator(f_rev)
     ```
   * আউটপুট: **ডোমেইন logits** (যেমন Source বনাম Target → `(B, 2)`)।
   * লস: `CrossEntropy(domain_logits, domain_labels)`

---

## কেন দুটোই FeatureExtractor এর আউটপুট নেয়?

* কারণ FeatureExtractor হলো **শেয়ার্ড নেটওয়ার্ক** → একবার ইমেজ থেকে ফিচার বের করলে,

  * সেই ফিচার দিয়ে ক্লাসিফিকেশন করা যায় (LabelPredictor)।
  * একই ফিচার দিয়ে সোর্স/টার্গেট আলাদা করা যায় (DomainDiscriminator)।

---

## মূল ধারণা

* **LabelPredictor** চায় ফিচারগুলোতে **ক্লাস সম্পর্কিত তথ্য** থাকুক।
* **DomainDiscriminator** চায় ফিচারগুলোতে **ডোমেইন সম্পর্কিত তথ্য** থাকুক।
* **GRL এর কারণে FeatureExtractor শিখে DomainDiscriminator কে confuse করতে**।
  👉 ফলে শেষ পর্যন্ত ফিচার হয় **domain-invariant + class-discriminative**।

---



```python
class DANN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, num_domains=2):
        super().__init__()
        self.feature_extractor = FeatureExtractor(in_channels=in_channels)
        feat_dim = self.feature_extractor.out_dim
        self.label_predictor = LabelPredictor(feat_dim, num_classes)
        self.domain_discriminator = DomainDiscriminator(feat_dim, num_domains)

        self.num_updates = 0  # alpha schedule এর জন্য
```

* **FeatureExtractor**: শেয়ার্ড CNN—ইমেজ → ফিচার ভেক্টর `f`।
* `feat_dim`: ফিচার ভেক্টরের দৈর্ঘ্য (যেমন MNIST হলে 48×7×7=2352)।
* **LabelPredictor**: `f` থেকে **ক্লাস logits** (B×num\_classes)।
* **DomainDiscriminator**: `f` (বা GRL-এর পরের `f_rev`) থেকে **ডোমেইন logits** (B×num\_domains)।
* `num_updates`: ট্রেনিং চলাকালে কত স্টেপ হলো—এটা দিয়ে GRL-এর `alpha` ধীরে ধীরে বাড়ানো হবে।

---

```python
    def alpha(self):
        """Standard DANN schedule: ধীরে ধীরে 0→1 (স্মুথ)"""
        p = min(self.num_updates / 10000.0, 1.0)
        # 2/(1+exp(-10p)) - 1: 0→1 (স্মুথ), scale চাইলে গুণ দিন
        return 2.0 / (1.0 + torch.exp(torch.tensor(-10.0 * p))) - 1.0
```

* **`alpha()`**: DANN-এর ক্লাসিক স্কেডিউল—শুরুর দিকে ছোট, পরে বড়।

  * `p ∈ [0,1]` (num\_updates বাড়লে 0→1)।
  * ফর্মুলা: $\alpha(p) = \frac{2}{1 + e^{-10p}} - 1$ → 0 থেকে 1-এ স্মুথলি যায়।
* কেন দরকার? শুরুতেই জোরে গ্রেডিয়েন্ট রিভার্স করলে ট্রেনিং অস্থিতিশীল হয়। তাই ধীরে ধীরে ডোমেইন-অ্যাডভার্সারিয়াল চাপ বাড়াই।

> নোট: চাইলে এই `alpha`-কে 0.5 বা 0.7 দিয়ে স্কেল করতে পারো (স্ট্যাবিলিটি টিউনিং)।




---

## 🔹 1. যখন **α = 0**

* Gradient Reversal Layer (GRL) কোনো প্রভাব ফেলে না।
* অর্থাৎ feature extractor **শুধু label prediction এর জন্য feature শিখছে**, domain-invariant কিছু শিখছে না।
* এই পর্যায়ে মডেল মূলত source data থেকে ভালো ক্লাসিফিকেশন শিখে নেয়।

👉 **শুরুতে GRL বন্ধ রাখা হয়** যাতে মডেল প্রথমে stable classification feature শিখতে পারে।

---

## 🔹 2. যখন **α ধীরে ধীরে বাড়ে (0 → 1)**

* GRL আস্তে আস্তে বেশি প্রভাব ফেলতে শুরু করে।
* মানে feature extractor কে **দ্বন্দ্বময় সিগন্যাল** দেওয়া হয়:

  * Label predictor বলছে: *“source label ঠিকমতো classify করতে শিখো”*
  * Domain discriminator বলছে: *“source আর target কে আলাদা করতে না শেখো”*
* ফলে extractor এমন feature শিখে যেটা দুই domain এর জন্যই সাধারণ (domain-invariant)।

👉 এখানে **adversarial game** হয় →

* Domain discriminator চেষ্টা করে source/target আলাদা করতে।
* Extractor চেষ্টা করে এমন feature বানাতে যাতে আলাদা করা না যায়।

---

## 🔹 3. যখন **α → 1 (সর্বোচ্চ)**

* GRL পুরো শক্তিতে কাজ করে (gradient পুরোপুরি উল্টো হয়ে যায়)।
* এখন feature extractor কে জোর করে domain-invariant feature শিখানো হয়।
* এর ফলে মডেল target domain এও ভালো কাজ করতে শুরু করে (Domain Adaptation সম্পূর্ণ হয়)।

---

## 📌 সংক্ষেপে ধাপগুলো

1. **α=0** → GRL বন্ধ → শুধু ক্লাসিফিকেশন শেখা।
2. **α মাঝামাঝি** → ধীরে ধীরে adversarial training শুরু → feature domain-invariant হতে থাকে।
3. **α=1** → GRL পূর্ণ শক্তিতে → extractor domain-invariant feature শিখে ফেলে।

---

👉 সহজভাবে মনে রাখুন:

* **α ছোট = ট্রেনিং স্থিতিশীল করা**
* **α বড় = ডোমেইন এডাপ্টেশন শক্তিশালী করা**

---


## 🔹 Domain Discriminator আসলে কী?

* এটা একটা **classifier head**, যেটা ফিচার ভেক্টর `f` ইনপুট নেয়।
* এর কাজ: **ফিচারটা Source domain থেকে এসেছে নাকি Target domain থেকে এসেছে সেটা প্রেডিক্ট করা**।
* উদাহরণ:

  * যদি source data MNIST হয় আর target data SVHN হয়,
  * তাহলে Domain Discriminator চেষ্টা করবে feature দেখে বুঝতে “এটা MNIST না SVHN”?

👉 মানে → **Domain Discriminator = Source vs Target আলাদা করার classifier**

---

## 🔹 GRL (Gradient Reversal Layer) যুক্ত হলে কী হয়?

* Domain Discriminator নিজের দিক থেকে তো আলাদা করতে চাইবেই (source=0, target=1)।
* কিন্তু Gradient Reversal Layer (GRL) উল্টো সিগন্যাল পাঠায় feature extractor এ।
* ফলে feature extractor কে বাধ্য করে **এমন feature শিখতে যাতে source আর target আলাদা করা না যায়**।

👉 তাই আমি বলেছিলাম:

> "Domain Discriminator বলছে: *source আর target কে আলাদা করতে না শেখো*"

আসলে ব্যাপারটা হলো:

* Discriminator চাইছে আলাদা করতে।
* GRL উল্টে দিয়ে extractor কে জোর করছে “আলাদা করা যেন না যায়।”

---

## 🔹 সহজ উদাহরণ

ধরা যাক,

* Source domain = হাতের লেখা সংখ্যা (কালো-সাদা, MNIST)
* Target domain = রঙিন সংখ্যা (SVHN)

**Domain Discriminator এর ভূমিকা:**

* ফিচার দেখে চেষ্টা করবে ধরতে → “এটা কালো-সাদা থেকে এসেছে নাকি রঙিন থেকে এসেছে?”

**Feature Extractor + GRL এর ভূমিকা:**

* এমন ফিচার বানাবে যেগুলো থেকে বোঝাই যাবে না এটা কালো-সাদা নাকি রঙিন।
* শুধু digit (0–9) ক্লাসিফিকেশনের জন্য দরকারি ইনফরমেশন থাকবে, domain এর তথ্য মুছে যাবে।

👉 এর ফলে মডেল **Domain-Invariant Features** শিখে →
যা Source এও ভালো কাজ করবে, Target এও ভালো কাজ করবে।

---

## 📌 সংক্ষেপে

* **Domain Discriminator** = ফিচার দেখে source/target আলাদা করতে চায়।
* **GRL + Feature Extractor** = এমন ফিচার বানায় যাতে Discriminator confuse হয়।
* **ফলাফল** = feature extractor domain-invariant feature শিখে → Target domain এও ভালো জেনারালাইজ করে।

---



## 🖥️ DANN Flow

```
           Input (x - image)
                  │
                  ▼
        ┌─────────────────────┐
        │  Feature Extractor  │
        └─────────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
        ▼                   ▼
 ┌─────────────┐     ┌──────────────────────┐
 │ Label       │     │ Gradient Reversal    │
 │ Predictor   │     │ Layer (GRL)          │
 └─────────────┘     └──────────────────────┘
        │                   │
        ▼                   ▼
 Class Prediction      ┌───────────────┐
 (Digits, etc.)        │ Domain        │
                       │ Discriminator │
                       └───────────────┘
                               │
                               ▼
                     Source vs Target Prediction
```

---

## 📌 এখানে কী হচ্ছে?

1. **Feature Extractor** → ইনপুট থেকে feature `f` বের করে।
2. **Label Predictor** → `f` ব্যবহার করে ক্লাস (digit/class) প্রেডিক্ট করে।
3. **Domain Discriminator** → `f` থেকে বলে দেয় → এটা Source থেকে নাকি Target থেকে এসেছে।
4. **GRL (Gradient Reversal Layer)** → Domain Discriminator এর gradient উল্টে দিয়ে Feature Extractor কে জোর করে **domain-invariant feature** বানাতে শেখায়।

👉 ফলে, মডেল শুধু source এ ভালো না, target domain এও ভালো জেনারালাইজ করে।

---


👉 এখন পর্যন্ত আমরা যা শিখলাম:

* **Feature Extractor** → কাঁচা data থেকে feature বের করে।
* **Label Predictor** → সেই feature দিয়ে ক্লাসিফিকেশন করে।
* **Domain Discriminator + GRL** → ফিচারগুলো যেন domain (Source/Target) অনুযায়ী আলাদা না হয়, বরং general হয়।
* **α (alpha)** → GRL এর প্রভাব ধীরে ধীরে বাড়ায় (শুরুতে 0 → পরে 1)।

---

📌 এক কথায় DANN এর মূল আইডিয়া:

> মডেলকে এমন feature শিখতে বাধ্য করা যা দিয়ে ক্লাস চেনা যাবে, কিন্তু domain (source/target) চেনা যাবে না।

---



* **Scatter plot (ভিজ্যুয়াল)** → চোখে দেখা যাবে source–target ফিচার কিভাবে আলাদা থাকে আর ধীরে ধীরে মিশে যায়।
* **Analogy (সহজ উদাহরণ)** → বাস্তব জীবনের তুলনা দিয়ে বোঝা সহজ হবে।

👉 আমি প্রস্তাব দিচ্ছি—
আগে আমি একটা **analogy** দিই যাতে আইডিয়াটা মাথায় বসে যায়,
তারপর চাইলে আমি একটা **scatter plot** এঁকে দেখাবো।

---

## 🎓 সহজ analogy (ছাত্র–শিক্ষক উদাহরণ)

ধরুন আপনি শিক্ষক।

* Source ছাত্র = ঢাকার ছাত্র (শুধু বাংলা বলে)
* Target ছাত্র = চট্টগ্রামের ছাত্র (চাটগাইয়া ভাষা বলে)

---

### 🔹 যখন α=0 (GRL কাজ করছে না)

* আপনি শুধু source ছাত্রকে পড়াচ্ছেন।
* Feature Extractor (ছাত্রের শেখা) এভাবে হচ্ছে যাতে ঢাকার ছাত্রের ভাষা-স্টাইল একদম আলাদা।
* Domain Discriminator সহজেই ধরতে পারছে → "এই ছাত্র ঢাকার, ওইটা চট্টগ্রামের।"
  👉 Source আর Target দুইটা আলাদা আলাদা গ্রুপে থেকে যাচ্ছে।

---

### 🔹 যখন α ধীরে ধীরে বাড়ে

* এখন GRL উল্টো চাপ দিচ্ছে:

  * Domain Discriminator চেষ্টা করছে আলাদা করতে → "ঢাকা না চট্টগ্রাম?"
  * কিন্তু Extractor এমন feature বানাতে শিখছে যেটা দিয়ে আলাদা করা কঠিন হয়।

👉 ফলে ধীরে ধীরে Source আর Target ছাত্রদের লেখাপড়ার style এক রকম হয়ে যাচ্ছে।

---

### 🔹 যখন α≈1 (GRL পুরো শক্তিতে)

* এখন Extractor এমন feature বানাচ্ছে যেটা দেখে বোঝাই যায় না কোন ছাত্র ঢাকার, কোনটা চট্টগ্রামের।
* শুধু “কে কেমন পড়ায় ভালো?” (label/class) সেটা বোঝা যায়।
  👉 Source–Target একসাথে মিশে গেছে → domain-invariant feature।

---

## 📌 এক কথায় analogy

* Domain Discriminator = শিক্ষক, চেষ্টা করছে ছাত্রের **অঞ্চল** চিনতে।
* GRL + Extractor = ছাত্রকে বাধ্য করছে এমন style শিখতে যেটা দেখে অঞ্চল বোঝা যায় না, শুধু **ক্লাসে কেমন** বোঝা যায়।

---








---

```python
    def forward(self, x, *, inference=False):
        f = self.feature_extractor(x)                  # shared features
        class_logits = self.label_predictor(f)         # label head
```

* **Forward-এর শুরু**

  1. `x` (B,C,H,W) → `FeatureExtractor` → `f` (B, feat\_dim)
  2. `f` → `LabelPredictor` → `class_logits` (B, num\_classes)
* এই পর্যন্ত **ক্লাস হেড** সবসময়ই চলে—ট্রেনিং/ইনফারেন্স দুই মোডেই।

---

```python
        if inference:
            # ইনফারেন্সে GRL দরকার নেই
            domain_logits = self.domain_discriminator(f)
            return class_logits, domain_logits, f
```

* **ইনফারেন্স মোড** (`inference=True`)

  * GRL শুধুই ব্যাকওয়ার্ডে কাজ করে; ইনফারেন্সে ব্যাকওয়ার্ড নেই, তাই GRL লাগেই না।
  * তবু `domain_logits` ক্যালকুলেট করলে মনিটরিং/অ্যানালাইসিস করা যায় (যেমন ফিচার কতটা ডোমেইন-ইনভারিয়্যান্ট হলো)।

---

```python
        a = float(self.alpha())                        # dynamic alpha
        f_rev = grl(f, a)                              # GRL
        domain_logits = self.domain_discriminator(f_rev)

        self.num_updates += 1
        return class_logits, domain_logits, f
```

* **ট্রেনিং মোড (adversarial part)**

  * `a = alpha()` → বর্তমান স্টেপ অনুযায়ী ডায়নামিক `alpha`।
  * `f_rev = grl(f, a)` → **Gradient Reversal Layer**:

    * **Forward**: `f_rev = f` (কোনো পরিবর্তন নয়)
    * **Backward**: $\frac{\partial L_d}{\partial f}$ গ্রেডিয়েন্টকে **`-a`** দিয়ে গুণ করে **উল্টো** পাঠায়।
      👉 DomainDiscriminator **L\_d মিনিমাইজ** করে, কিন্তু FeatureExtractor **L\_d ম্যাক্সিমাইজ** করতে শেখে → **domain-invariant features**।
  * `domain_logits = DomainDiscriminator(f_rev)` → ডোমেইন হেড logits।
  * `self.num_updates += 1` → পরের স্টেপে `alpha` একটু বড় হবে।
  * রিটার্ন করি তিনটা জিনিস:

    1. `class_logits` (B×num\_classes)
    2. `domain_logits` (B×num\_domains)
    3. `f` (raw features; লস/মেট্রিক্স/লগিংয়ে কাজে লাগতে পারে)

---

## শেপ ট্র্যাকার (MNIST উদাহরণ)

* ইনপুট `x`: `(B, 1, 28, 28)`
* `f = FeatureExtractor(x)`: `(B, 2352)`
* `class_logits = LabelPredictor(f)`: `(B, 10)`
* `f_rev = grl(f, a)`: `(B, 2352)` (forward-এ অপরিবর্তিত)
* `domain_logits = DomainDiscriminator(f_rev)`: `(B, 2)`

---

## ট্রেনিং-টাইম লস কিভাবে ধরবে?

সাধারণত একটা ট্রেনিং স্টেপে:

* সোর্স ব্যাচে `class_logits` থেকে **classification CE loss**।
* সোর্স+টার্গেট জোড়া দিয়ে `domain_logits` থেকে **domain CE loss**।
* **টোটাল লস**: `cls_loss + λ * domain_loss` (λ=1 ধরেছ, তুমি চাইলে স্কেল করতে পারো)।
* ব্যাকওয়ার্ডে GRL স্বয়ংক্রিয়ভাবে ফিচার এক্সট্র্যাক্টরের গ্রেডিয়েন্ট উল্টো করে দেয়।

---

## ছোট টিপস

* `alpha()`-তে `torch.tensor(-10.0 * p)` CPU টেনসর—এটা ঠিক আছে, কারণ শেষে `float()` নেওয়া হচ্ছে; চাইলে `.item()` করেও নিতে পারো।
* ইনফারেন্স ব্রাঞ্চে `domain_logits` নাও লাগতে পারে—মনিটরিং না চাইলে বাদ দিতে পারো।
* `num_updates` তুমি **প্রতি optimizer.step**-এ আপডেট করবে—এই কোডে forward-এর শেষে করা হয়েছে, তাই একবার forward মানে একবার বাড়বে। (আমরা আগের উদাহরণে সোর্স+টার্গেট চালানোর সময় ডাবল-কাউন্ট এড়াতে adjust করেছিলাম।)

---

### সংক্ষেপে ফ্লো (টেক্সট ডায়াগ্রাম)

```
x (B,C,H,W)
   │
   ▼
FeatureExtractor ───► f (B,feat_dim)
   │                      │
   │                      ├────────► LabelPredictor ──► class_logits (B,num_classes)
   │
   └───── GRL (with alpha) on f ──► DomainDiscriminator ─► domain_logits (B,num_domains)
```

এটাই DANN-এর মূল ধারণা—**ক্লাসিফিকেশন ভালো রাখতে রাখতে ডোমেইন পার্থক্য মুছে ফেলা**।


---

## 🖥️ DANN `forward()` এ আসলে কী হচ্ছে?

1. **Input (x)** → মডেলে ঢুকছে।
2. **Feature Extractor** → `f` (ফিচার) বের করে।
3. **Label Predictor** → সেই ফিচার দিয়ে ক্লাসিফিকেশন করে (digit 0–9 বা যেটা দরকার)।

---

### যখন `inference=True` (শুধু প্রেডিকশন চাই)

* ফিচার `f` সরাসরি Domain Discriminator এ পাঠানো হয়।
* কোনো GRL (Gradient Reversal Layer) ব্যবহার হয় না।
* শুধু আউটপুট দিই:

  * ক্লাস logits (class score)
  * ডোমেইন logits (source/target score)
  * ফিচার `f`

👉 এটা হয় শুধু **ইভ্যালুয়েশন/প্রেডিকশন টাইমে**।

---

### যখন `inference=False` (ট্রেনিং টাইমে, ডিফল্ট)

* প্রথমে **alpha (α)** বের করা হয় → যেটা ধীরে ধীরে 0 থেকে 1 বাড়ে।
* `f` কে GRL এর ভেতর পাঠানো হয় → এতে গ্রেডিয়েন্ট উল্টে যায়।
* উল্টানো ফিচার `f_rev` → Domain Discriminator এ যায়।
* ফলে Domain Discriminator আর Feature Extractor-এর মধ্যে **adversarial game** হয়।
* শেষে আউটপুট হয়:

  * ক্লাস logits
  * ডোমেইন logits
  * ফিচার

👉 এটা হয় শুধু **ট্রেনিং টাইমে**, যাতে মডেল domain-invariant ফিচার শিখতে পারে।

---

## 📌 এক কথায়

* **Training (inference=False)** → GRL চালু → domain adaptation শেখানো হয়।
* **Inference (inference=True)** → GRL বন্ধ → শুধু প্রেডিকশন নেওয়া হয়।

---








---

## কোড

```python
def dann_train_step(model, batch_source, batch_target, optimizer, device):
    """
    batch_source: dict{ 'x': images, 'y': labels, 'd': domain_ids(0 for source) }
    batch_target: dict{ 'x': images,             'd': domain_ids(1 for target) }
    """
```

* এই ফাংশনটা একটা **single training step** চালায়।
* ইনপুট:

  * `batch_source`: source ডোমেইন থেকে ইমেজ + লেবেল + domain id (=0)।
  * `batch_target`: target ডোমেইন থেকে ইমেজ + domain id (=1)।
* Target ডোমেইনের **ক্লাস লেবেল থাকে না** (unsupervised DA), শুধু domain label থাকে।

---

```python
    model.train()
    xs, ys, ds = batch_source['x'].to(device), batch_source['y'].to(device), batch_source['d'].to(device)
    xt, dt     = batch_target['x'].to(device), batch_target['d'].to(device)
```

* `model.train()` → dropout, BN ইত্যাদি training মোডে যাবে।
* Source ব্যাচ থেকে:

  * `xs`: source images
  * `ys`: source class labels
  * `ds`: source domain labels (সব 0)
* Target ব্যাচ থেকে:

  * `xt`: target images
  * `dt`: target domain labels (সব 1)

---

```python
    # 1) Label prediction loss (শুধু source লেবেল থাকে)
    cls_logits, dom_logits_s, _ = model(xs, inference=False)
    cls_loss = F.cross_entropy(cls_logits, ys)
```

* Source images `xs` মডেলে চালালাম → আউটপুট তিনটা:

  * `cls_logits`: class prediction (B, num\_classes)
  * `dom_logits_s`: domain prediction (B, num\_domains)
  * `_`: ফিচার (unused here)
* Source data-র জন্য আমরা জানি `ys` (ক্লাস লেবেল), তাই `CrossEntropy` দিয়ে **classification loss** ক্যালকুলেট।

---

```python
    # 2) Domain loss (source + target; দুটোই ডোমেইন লেবেল জানি)
    # source pass already done -> dom_logits_s
    # target pass (শুধু domain head-এর জন্য লাগে)
    with torch.no_grad():
        model.num_updates -= 1  # alpha schedule যেন এক স্টেপে দু'বার না বাড়ে
    _, dom_logits_t, _ = model(xt, inference=False)
```

* Domain loss-এর জন্য Source + Target দুই ডোমেইনের domain logits লাগবে।
* Source এরটা আগেই পেয়েছি (`dom_logits_s`)।
* Target এরটার জন্য আবার মডেলে চালালাম → `dom_logits_t`।

**কিন্তু:**

* `model.forward()` এর শেষে `num_updates += 1` ছিল।
* আমরা এই স্টেপে model দু’বার কল করছি (source + target), তাহলে update দুইবার বেড়ে যেত।
* তাই `with torch.no_grad(): model.num_updates -= 1` দিয়ে একবার কমানো হলো → মোটে ১ বার বাড়বে, ঠিক থাকবে।

---

```python
    dom_logits = torch.cat([dom_logits_s, dom_logits_t], dim=0)
    dom_labels = torch.cat([ds, dt], dim=0)
    domain_loss = F.cross_entropy(dom_logits, dom_labels)
```

* Source + Target এর domain logits একসাথে concatenate করলাম।
* Domain labels-ও (0 + 1) একসাথে করলাম।
* CrossEntropy দিয়ে **domain classification loss** ক্যালকুলেট করলাম।

---

```python
    # 3) Total loss
    loss = cls_loss + domain_loss
```

* মোট লস = **classification loss + domain loss**।
* অনেক পেপারে λ (lambda weight) থাকে domain\_loss এর সাথে, এখানে সরলভাবে 1:1 রাখা হয়েছে।

---

```python
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

* অপ্টিমাইজার দিয়ে গ্রেডিয়েন্ট রিসেট।
* ব্যাকওয়ার্ড পাস → GRL এর জন্য FeatureExtractor-এর গ্রেডিয়েন্ট উল্টো হয়ে যাবে।
* অপ্টিমাইজার স্টেপ নিয়ে ওজন আপডেট হবে।

---

```python
    return {
        "loss": float(loss.item()),
        "cls_loss": float(cls_loss.item()),
        "domain_loss": float(domain_loss.item()),
        "alpha": float(model.alpha().item())
    }
```

* স্ট্যাটস রিটার্ন করছে: মোট লস, ক্লাস লস, ডোমেইন লস, আর `alpha`-র কারেন্ট ভ্যালু (মনিটরিংয়ের জন্য)।

---

## সারসংক্ষেপ

এই `dann_train_step` এ যা হচ্ছে:

1. **Source data** → Label predictor loss (কারণ class labels শুধু source এ আছে)।
2. **Source + Target data** → Domain discriminator loss (কারণ domain labels দুটোতেই আছে)।
3. Loss যোগ করে ব্যাকওয়ার্ড।
4. GRL-এর কারণে FeatureExtractor শিখছে domain-invariant ফিচার বানাতে।

---




---

## কোড বিশ্লেষণ

```python
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
```

* স্ক্রিপ্ট সরাসরি রান করলে (`python file.py`) এই ব্লক এক্সিকিউট হবে।
* যদি GPU থাকে → `"cuda"` ব্যবহার করবে, নাহলে CPU।

---

```python
    model = DANN(in_channels=1, num_classes=10, num_domains=2).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=2e-4)
```

* `DANN` মডেল বানানো হলো:

  * `in_channels=1` → MNIST (গ্রেস্কেল ইমেজ)।
  * `num_classes=10` → 0–9 ডিজিট।
  * `num_domains=2` → Source (0) ও Target (1) ডোমেইন।
* `.to(device)` → GPU/CPU তে মডেল পাঠানো হলো।
* Optimizer: **Adam**, লার্নিং রেট = 2e-4।

---

```python
    # Dummy example (MNIST-like shapes)
    B = 32
    xs = torch.randn(B, 1, 28, 28)  # source images
    ys = torch.randint(0, 10, (B,)) # source labels
    ds = torch.zeros(B, dtype=torch.long)  # domain id = 0

    xt = torch.randn(B, 1, 28, 28)  # target images
    dt = torch.ones(B, dtype=torch.long)   # domain id = 1
```

* একটা **ডামি ব্যাচ** বানানো হলো MNIST-এর মতো:

  * `xs`: Source images → শেপ `(32, 1, 28, 28)`
  * `ys`: Random source labels (0–9), শেপ `(32,)`
  * `ds`: সবগুলো domain id = 0 (source domain)
  * `xt`: Target images → শেপ `(32, 1, 28, 28)`
  * `dt`: সবগুলো domain id = 1 (target domain)

👉 এখানে target images-এর class labels নেই, শুধু domain labels আছে (unsupervised domain adaptation-এর ধরন)।

---

```python
    stats = dann_train_step(
        model,
        batch_source={"x": xs, "y": ys, "d": ds},
        batch_target={"x": xt, "d": dt},
        optimizer=optim,
        device=device,
    )
    print(stats)
```

* আমাদের লেখা `dann_train_step` ফাংশন চালানো হলো এক স্টেপের জন্য।
* ইনপুট ডিকশনারি আকারে দেওয়া হয়েছে (`x`, `y`, `d`)।
* ফাংশন রিটার্ন করবে লস ও অন্যান্য স্ট্যাটস → প্রিন্ট করা হলো।

---

## আউটপুট কেমন হবে?

`print(stats)` করলে এরকম কিছু আসবে (random data বলে মান আলাদা হবে):

```python
{
 'loss': 2.312,
 'cls_loss': 2.301,
 'domain_loss': 0.011,
 'alpha': 0.0002
}
```

* **loss** → মোট লস (cls\_loss + domain\_loss)
* **cls\_loss** → সোর্সে ক্লাসিফিকেশন লস
* **domain\_loss** → সোর্স+টার্গেট ডোমেইন ক্লাসিফিকেশন লস
* **alpha** → GRL এর কারেন্ট ভ্যালু (শুরুতে 0 এর কাছাকাছি, ধীরে ধীরে বাড়ে)

---

## সারসংক্ষেপ

এই skeleton অংশটা আসলে দেখাচ্ছে:

1. মডেল বানানো → Optimizer বানানো
2. Source ও Target-এর ডামি ব্যাচ তৈরি
3. একবার `dann_train_step` চালানো
4. লস ভ্যালুগুলো প্রিন্ট করা

---




