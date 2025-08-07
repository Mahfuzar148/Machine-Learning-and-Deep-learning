
---

## ✅ Full PyTorch Code for Logistic Regression (Breast Cancer Dataset)

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load and prepare the data
x, y = load_breast_cancer(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Standardize the features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Convert to PyTorch tensors
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Define the logistic regression model
model = nn.Sequential(
    nn.Linear(x_train.shape[1], 1),
    nn.Sigmoid()
)

# Define the loss function and optimizer
loss_fn = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model
losses = []
for epoch in range(100):
    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)
    losses.append(loss.item())

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Evaluate the model
with torch.no_grad():
    train_acc = (model(x_train).round() == y_train).float().mean()
    test_acc = (model(x_test).round() == y_test).float().mean()

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Plot the training loss
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid(True)
plt.show()
```

---



---

## 🔢 পূর্ণ কোডের লাইন-বাই-লাইন ব্যাখ্যা:

```python
import torch
```

* ✅ `torch`: PyTorch লাইব্রেরি, যা tensor operation, model design, training ইত্যাদির জন্য ব্যবহৃত হয়।

---

```python
import torch.nn as nn
```

* ✅ `torch.nn`: PyTorch-এর neural network module।
* `nn` হল alias — এখন আমরা `nn.Linear`, `nn.Sigmoid` ইত্যাদি ব্যবহার করতে পারব।

---

```python
import torch.optim as optim
```

* ✅ `torch.optim`: Optimization tools (যেমন SGD, Adam ইত্যাদি) পাওয়ার জন্য ব্যবহৃত হয়।

---

```python
from sklearn.datasets import load_breast_cancer
```

* ✅ `load_breast_cancer`: sklearn থেকে breast cancer dataset লোড করে।

---

```python
from sklearn.model_selection import train_test_split
```

* ✅ `train_test_split`: ডেটাকে ট্রেন ও টেস্ট সেটে ভাগ করার ফাংশন।

---

```python
from sklearn.preprocessing import StandardScaler
```

* ✅ `StandardScaler`: সব feature-কে standardize (mean=0, std=1) করে।

---

```python
import matplotlib.pyplot as plt
```

* ✅ `matplotlib.pyplot`: ডেটা visualization এর জন্য plotting tool।

---

```python
x, y = load_breast_cancer(return_X_y=True)
```

* `x`: Features (30টা)
* `y`: Labels (0 বা 1 — malignant বা benign)

---

```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
```

* ডেটাকে ৮০% train ও ২০% test সেটে ভাগ করে।
* `random_state=0`: repeatable result পেতে।

---

```python
scaler = StandardScaler()
```

* Scaler object তৈরি করলো।

---

```python
x_train = scaler.fit_transform(x_train)
```

* ✅ `fit_transform`: train ডেটার mean ও std দিয়ে scale করে।

---

```python
x_test = scaler.transform(x_test)
```

* ✅ `transform`: একই scaling rule test set এ apply করে।

---

```python
x_train = torch.tensor(x_train, dtype=torch.float32)
```

* Numpy array → PyTorch Tensor (float32: standard datatype deep learning এর জন্য)

---

```python
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
```

* `.view(-1, 1)` → একমাত্রিক vector কে 2D (column) tensor বানায়।
* Binary classification এর জন্য `[n_samples, 1]` shape দরকার।

---

```python
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
```

* টেস্ট ডেটার জন্যও একই রকম রূপান্তর।

---

```python
model = nn.Sequential(
    nn.Linear(x_train.shape[1], 1),
    nn.Sigmoid()
)
```

* ✅ `nn.Sequential`: একটি simple stack of layers।
* `nn.Linear`: input → output (এখানে 30 input → 1 output)
* `nn.Sigmoid`: output কে 0-1 probability তে রূপান্তর করে।

---

```python
loss_fn = nn.BCELoss()
```

* ✅ `BCELoss`: Binary Cross Entropy Loss — classification error পরিমাপ করে।

---

```python
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

* ✅ `SGD`: Stochastic Gradient Descent optimizer।
* `model.parameters()`: model এর weight & bias।
* `lr=0.01`: learning rate → কতটা ধাপে weight আপডেট হবে।

---

```python
losses = []
```

* ✅ Loss values store করার জন্য খালি লিস্ট।

---

```python
for epoch in range(100):
```

* ১০০ বার model কে train করানো হবে (epochs = full passes over data)

---

```python
    y_pred = model(x_train)
```

* Model input নেয় এবং output (prediction) করে।

---

```python
    loss = loss_fn(y_pred, y_train)
```

* Prediction আর true label এর মধ্যে error মাপা হয়।

---

```python
    losses.append(loss.item())
```

* `.item()` → PyTorch tensor থেকে scalar বের করে নেয়।
* `losses` লিস্টে সেই মান যোগ করে।

---

```python
    loss.backward()
```

* Gradient calculate করে (backpropagation)।

---

```python
    optimizer.step()
```

* Weight আপডেট করে (gradient descent step)।

---

```python
    optimizer.zero_grad()
```

* আগের gradient clear করে, যাতে accumulate না হয়।

---

```python
with torch.no_grad():
```

* এই block এর ভেতরে gradient tracking বন্ধ থাকে → faster evaluation।

---

```python
    train_acc = (model(x_train).round() == y_train).float().mean()
```

* Model prediction কে 0/1 তে রাউন্ড করে।
* সঠিক prediction এর সংখ্যার গড় হিসেব করে → accuracy।

---

```python
    test_acc = (model(x_test).round() == y_test).float().mean()
```

* Test dataset এর accuracy হিসাব করে।

---

```python
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
```

* Accuracy print করে ৪ দশমিক পর্যন্ত।

---

```python
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)
plt.show()
```

* Loss curve plot করে → দেখতে কেমন করে loss কমেছে।

---

## 📘 গুরুত্বপূর্ণ PyTorch কিওয়ার্ড ও ব্যাখ্যা:

| কিওয়ার্ড               | ব্যাখ্যা                                    |
| ---------------------- | ------------------------------------------- |
| `torch.tensor()`       | NumPy → PyTorch টেন্সর রূপান্তর করে         |
| `nn.Linear`            | একটি লিনিয়ার লেয়ার তৈরি করে (Wx + b)      |
| `nn.Sigmoid()`         | Output কে 0-1 এর মধ্যে স্কেল করে            |
| `nn.BCELoss()`         | Binary classification এর জন্য loss function |
| `optimizer.step()`     | Weights আপডেট করে                           |
| `loss.backward()`      | Gradient গুলো calculate করে                 |
| `zero_grad()`          | আগের gradient মুছে ফেলে                     |
| `with torch.no_grad()` | Evaluation এর সময় gradient হিসাব না করে     |
| `.round()`             | Probabilities → 0 বা 1 (threshold 0.5)      |

---

