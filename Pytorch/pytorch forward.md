

---

## ⚙️ `forward()` ফাংশনের মূল কাজ

`forward()` ফাংশন হলো —
👉 **মডেলের ইনপুট ডেটা কীভাবে এক লেয়ার থেকে আরেক লেয়ারে যাবে, সেটা নির্ধারণ করা।**

অর্থাৎ, `forward()` বলে দেয় **data flow / computation flow** কেমন হবে।

এক কথায়:

> “forward() = ইনপুট ডেটা কীভাবে আউটপুটে রূপান্তরিত হবে।”

---

## 🧠 উদাহরণ দিয়ে বুঝি

ধরা যাক, তুমি একটা simple network বানালে:

```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(4, 3)   # Layer 1: 4 → 3
        self.fc2 = nn.Linear(3, 1)   # Layer 2: 3 → 1
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Step 1: fc1 এর পর ReLU activation
        x = self.fc2(x)              # Step 2: fc2 layer
        return x
```

### 🔍 এখানে কী হচ্ছে?

ধরা যাক ইনপুট:

```python
input_data = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
```

তাহলে `forward()` কাজ করবে এভাবে:

1️⃣ `self.fc1(x)`
→ Linear transformation করবে:
  `x1 = x @ W1.T + b1`

2️⃣ `torch.relu(x1)`
→ Negative মানগুলো 0 করে দেবে।

3️⃣ `self.fc2(x)`
→ আবার linear transformation:
  `output = x2 @ W2.T + b2`

4️⃣ `return output`
→ শেষ পর্যন্ত network এর prediction।

---

## 🔄 গুরুত্বপূর্ণ ব্যাপার: “Automatic Call”

তুমি কখনও সরাসরি `model.forward(x)` ডাকে না।
বরং `model(x)` লিখলেই PyTorch স্বয়ংক্রিয়ভাবে `forward()` ফাংশন কল করে।

```python
model = MyModel()
output = model(input_data)   # internally calls model.forward(input_data)
```

---

## ⚡ অতিরিক্ত সুবিধা

`forward()` হলো pure function —
এখানে শুধু **computation** define করা হয়, training সম্পর্কিত কিছু নয়।
Gradient, loss calculation, optimization — এগুলো পরে `backward()` phase-এ হয়।

---

## 🔁 সংক্ষেপে:

| ধাপ          | কাজ                                       |
| ------------ | ----------------------------------------- |
| `__init__()` | Layer define করা হয়                       |
| `forward()`  | Data কীভাবে flow করবে, সেটা define করা হয় |
| `model(x)`   | আসলে `forward(x)` কল হয়                   |
| Output       | Final prediction পাওয়া যায়                |

---

## 🧩 ছোট উদাহরণ (Flow দেখা)

```python
x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
model = MyModel()
y = model(x)
print(y)
```

এখানে model করবে:

```
x → fc1 (linear) → ReLU → fc2 (linear) → output
```

---

### 🔸 মনে রাখো:

`forward()` শুধু forward pass-এর জন্য —
অর্থাৎ prediction বা output তৈরি করার সময় চালানো হয়।
এর পর `loss.backward()` করলে backward pass (gradient হিসাব) শুরু হয়।

---

