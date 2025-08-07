
---

## ✅ Full Code: PyTorch Linear Regression Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# 1. Generate synthetic regression dataset
x_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=10, random_state=1)

# 2. Convert numpy arrays to PyTorch tensors
x = torch.from_numpy(x_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)  # reshape to (100, 1)

# 3. Model definition
model = nn.Linear(in_features=1, out_features=1)

# 4. Define Loss Function and Optimizer
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 5. Training Loop
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    y_pred = model(x)

    # Compute loss
    loss = loss_fn(y_pred, y)

    # Backward pass
    loss.backward()

    # Update weights
    optimizer.step()

    # Zero gradients
    optimizer.zero_grad()

    # Print every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 6. Visualize the regression line
predicted = model(x).detach().numpy()
plt.plot(x_numpy, y_numpy, 'ro', label='Original data')
plt.plot(x_numpy, predicted, label='Fitted line')
plt.legend()
plt.show()
```

---

## 🧠 Step-by-Step Explanation in Bengali

---

### 🔢 **1. ডেটাসেট তৈরি করা**

```python
x_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=10, random_state=1)
```

✅ এটা `scikit-learn` লাইব্রেরির `make_regression()` ফাংশন, যা কৃত্রিম ডেটা তৈরি করে।

* `n_samples=100`: ১০০টি ডেটা পয়েন্ট তৈরি করে।
* `n_features=1`: প্রতি ডেটা পয়েন্টে ১টি ফিচার থাকবে।
* `noise=10`: র‍্যান্ডম শব্দ যোগ করে বাস্তবের মতো করে তোলে।

---

### 🔁 **2. PyTorch টেন্সরে রূপান্তর**

```python
x = torch.from_numpy(x_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)
```

✅ NumPy array → PyTorch টেন্সরে রূপান্তর করা হয়েছে।

🔄 `y.view(...)` দিয়ে `y` এর শেপ করা হয়েছে `[100, 1]`, যেন মডেলের আউটপুটের সঙ্গে মেলে।

---

### 🏗️ **3. মডেল ডিফাইন করা**

```python
model = nn.Linear(in_features=1, out_features=1)
```

✅ `nn.Linear()` ব্যবহার করে একটি সোজা লিনিয়ার রিগ্রেশন মডেল তৈরি করা হয়েছে:
$\hat{y} = w \cdot x + b$

এখানে:

* `in_features=1`: ইনপুট একমাত্রা।
* `out_features=1`: আউটপুট একমাত্রা।

---

### 🧮 **4. লস ফাংশন এবং অপটিমাইজার**

```python
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

* **Loss Function:**

  ```python
  loss_fn = nn.MSELoss()
  ```

  📏 Mean Squared Error ব্যবহার করা হয়েছে।
  $\text{Loss} = \frac{1}{n} \sum (y - \hat{y})^2$
  এটি predicted ও actual মানের পার্থক্য পরিমাপ করে।

* **Optimizer:**

  ```python
  optimizer = optim.SGD(model.parameters(), lr=0.01)
  ```

  🔁 Stochastic Gradient Descent (SGD) ব্যবহার করে ওজন আপডেট করা হবে।

---

### 🔁 **5. Training Loop**

```python
for epoch in range(num_epochs):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

⏱️ **প্রতি epoch-এ যা হচ্ছে:**

1. **Forward Pass:** মডেল দিয়ে প্রেডিকশন করা হচ্ছে।
2. **Loss Calculation:** `MSELoss` দিয়ে error মাপা হচ্ছে।
3. **Backward Pass:** `.backward()` দিয়ে গ্র্যাডিয়েন্ট হিসাব করা হচ্ছে।
4. **Weight Update:** `optimizer.step()` দিয়ে ওজন আপডেট হচ্ছে।
5. **Gradient Reset:** `optimizer.zero_grad()` দিয়ে পুরনো গ্র্যাডিয়েন্ট মুছে ফেলা হচ্ছে।

📤 প্রতি ১০০ epoch পরপর প্রিন্ট:

```python
if (epoch + 1) % 100 == 0:
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
```

---

### 📊 **6. ফাইনাল ফলাফল ভিজ্যুয়ালাইজ করা**

```python
predicted = model(x).detach().numpy()
plt.plot(x_numpy, y_numpy, 'ro', label='Original data')
plt.plot(x_numpy, predicted, label='Fitted line')
plt.legend()
plt.show()
```

* মডেলের প্রেডিকশন `.detach().numpy()` করে NumPy array বানানো হয়েছে।
* `matplotlib` দিয়ে আসল ডেটা ও প্রেডিক্ট করা রেখা প্লট করা হয়েছে।

---

## ✅ এই কোড দিয়ে আপনি কী শিখলেন?

| বিষয়                | বর্ণনা                           |
| ------------------- | -------------------------------- |
| Dataset Preparation | `make_regression` দিয়ে ডেটা তৈরি |
| Model Building      | `nn.Linear` দিয়ে রিগ্রেশন মডেল   |
| Loss Function       | `nn.MSELoss` দিয়ে ভুল হিসাব      |
| Optimizer           | `SGD` দিয়ে ওজন আপডেট             |
| Training            | Manual forward + backward pass   |
| Visualization       | মডেলের শেখা রেখা দেখানো হয়েছে    |

---


