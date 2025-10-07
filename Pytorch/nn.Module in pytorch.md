

---

## 🧠 `nn.Module` কী?

`torch.nn.Module` হলো এমন একটা class,
যেটা PyTorch-কে বলে দেয়—

> “এই ক্লাসটা একটা neural network model,
> যার মধ্যে কিছু trainable parameter আছে (weights, biases)
> এবং একটা forward computation আছে।”

অর্থাৎ, এটা হলো PyTorch-এর **model container**, যেখানে তুমি তোমার লেয়ারগুলো (layer), activation, forward logic ইত্যাদি define করবে।

---

## 🧩 `nn.Module` কীভাবে কাজ করে?

যখন তুমি একটা class বানাও যা `nn.Module` থেকে inherit করে, তখন নিচের তিনটা জিনিস সবচেয়ে গুরুত্বপূর্ণ:

### 1️⃣ `__init__()`

👉 এখানে তুমি network-এর লেয়ারগুলো define করবে, যেমন—
`nn.Linear`, `nn.Conv2d`, `nn.ReLU`, ইত্যাদি।
সব লেয়ার `self`-এর property হিসেবে রাখতে হবে যাতে PyTorch তাদের parameters track করতে পারে।

```python
import torch
import torch.nn as nn

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.fc1 = nn.Linear(10, 20)   # input layer
        self.fc2 = nn.Linear(20, 1)    # output layer
```

---

### 2️⃣ `forward()`

👉 এখানে তুমি বলবে, ইনপুট ডেটা model-এর মধ্য দিয়ে কীভাবে যাবে (data flow)।
এখানে define করা হয় পুরো computation process।

```python
    def forward(self, x):
        x = torch.relu(self.fc1(x))   # hidden layer + ReLU
        x = self.fc2(x)               # output layer
        return x
```

---

### 3️⃣ Model instance তৈরি করা

এখন তুমি `MyNetwork()` থেকে একটা model বানিয়ে নিতে পারো:

```python
model = MyNetwork()
print(model)
```

---

## ⚙️ PyTorch-এর ভেতরে `nn.Module` কী করে?

যখন তুমি `nn.Module` inherit করো, তখন এটা তোমাকে নিচের সুবিধাগুলো দেয়:

| কাজ                             | ব্যাখ্যা                                                                            |
| ------------------------------- | ----------------------------------------------------------------------------------- |
| ✅ **Parameter tracking**        | `self.parameters()` দিয়ে সব trainable parameter স্বয়ংক্রিয়ভাবে track হয়।            |
| ✅ **Device handling (CPU/GPU)** | `.to(device)` দিয়ে পুরো model-টা GPU-তে পাঠানো যায়।                                 |
| ✅ **Saving/Loading**            | `torch.save(model.state_dict(), path)` ও `model.load_state_dict(...)` সহজে কাজ করে। |
| ✅ **Nested model support**      | অন্য `nn.Module` ক্লাসের ভেতরে আরও `nn.Module` রাখা যায় (layer composition)।        |
| ✅ **Hooks & debugging**         | Intermediate layer output দেখতে hooks ব্যবহার করা যায়।                              |

---

## 🧱 Example (Full Code)

```python
import torch
import torch.nn as nn

# Define the network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(5, 10)
        self.fc2 = nn.Linear(10, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the model
model = SimpleNet()

# Random input
input_data = torch.randn(3, 5)  # batch_size=3, features=5
output = model(input_data)

print("Input:", input_data)
print("Output:", output)
```

---

## 🧩 মনে রাখো

* `nn.Module` হলো তোমার neural network-এর **blueprint**।
* `__init__()` → structure define করে
* `forward()` → data কীভাবে যাবে তা বলে
* PyTorch-এর autograd system এরপর স্বয়ংক্রিয়ভাবে gradient হিসাব করে নেয়।

---

