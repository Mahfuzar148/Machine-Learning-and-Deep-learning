
---

## **Full Code**

```python
import torch

# Relationship: f = w * x
# Our dataset follows f = 2 * x

# Training data
x = torch.tensor([1, 2, 3, 4], dtype=torch.float32)    # Inputs
y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)    # Outputs

# Initial weight (parameter to learn)
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# Forward pass function
def forward(x):
    return w * x

# Loss function (Mean Squared Error)
def loss(y, y_pred):
    return ((y_pred - y) ** 2).mean()

# Initial prediction check
print(f'Prediction before training: {forward(torch.tensor(5.0)).item():.3f}')

# Hyperparameters
learning_rate = 0.01
n_iters = 1000

# Training loop
for epoch in range(n_iters):
    # 1. Forward pass (prediction)
    y_pred = forward(x)
    
    # 2. Compute loss
    l = loss(y, y_pred)
    
    # 3. Backward pass (compute gradient)
    l.backward()
    dw = w.grad.item()
    
    # 4. Update weight
    with torch.no_grad():
        w -= learning_rate * dw
    
    # 5. Zero the gradients for the next iteration
    w.grad.zero_()
    
    # 6. Print progress every 10 epochs
    if epoch % 10 == 0:
        print(f'Epoch {epoch + 1}/{n_iters}, Loss: {l.item():.4f}, w: {w.item():.4f}')

# Final prediction check
print(f'Prediction after training: {forward(torch.tensor(5.0)).item():.3f}')
```

---

## **Step-by-Step Explanation**

### **1️⃣ Import Library**

```python
import torch
```

* PyTorch ইম্পোর্ট করা হচ্ছে যাতে আমরা টেনসর (tensor) নিয়ে কাজ করতে পারি, গ্রেডিয়েন্ট বের করতে পারি, এবং মডেল ট্রেন করতে পারি।

---

### **2️⃣ Training Data**

```python
x = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)
```

* ইনপুট `x` এবং আউটপুট `y` ডেটাসেট।
* এখানে সম্পর্ক: $y = 2 \times x$

---

### **3️⃣ Initialize Weight**

```python
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
```

* `w` হলো মডেলের প্যারামিটার যা ট্রেনিংয়ের সময় শেখা হবে।
* শুরুতে 0.0।
* `requires_grad=True` দিলে PyTorch এই ভ্যারিয়েবলটির গ্রেডিয়েন্ট ক্যালকুলেট করবে।

---

### **4️⃣ Forward Function**

```python
def forward(x):
    return w * x
```

* প্রেডিকশন ফর্মুলা:

$$
\hat{y} = w \times x
$$

---

### **5️⃣ Loss Function**

```python
def loss(y, y_pred):
    return ((y_pred - y) ** 2).mean()
```

* Mean Squared Error (MSE) ব্যবহার করা হচ্ছে:

$$
\text{Loss} = \frac{1}{N} \sum (y_{\text{pred}} - y)^2
$$

* এটা মডেলের প্রেডিকশন আর আসল ভ্যালুর মধ্যে পার্থক্য মাপছে।

---

### **6️⃣ Initial Prediction**

```python
print(f'Prediction before training: {forward(torch.tensor(5.0)).item():.3f}')
```

* ট্রেনিং শুরুর আগে $x = 5$ দিলে প্রেডিকশন চেক করা।
* শুরুতে $w = 0.0$ হওয়ায় আউটপুট হবে 0.000।

---

### **7️⃣ Hyperparameters**

```python
learning_rate = 0.01
n_iters = 1000
```

* **learning\_rate** → প্রতিবার কতটা করে ওজন (weight) আপডেট হবে।
* **n\_iters** → ট্রেনিং কতবার চলবে।

---

### **8️⃣ Training Loop**

```python
for epoch in range(n_iters):
```

* প্রতিটি লুপে একবার করে **forward pass + backward pass + weight update** হবে।

---

#### **Step 1 — Forward Pass**

```python
y_pred = forward(x)
```

* বর্তমান `w` ব্যবহার করে প্রেডিকশন করা।

---

#### **Step 2 — Compute Loss**

```python
l = loss(y, y_pred)
```

* প্রেডিকশন আর আসল ভ্যালুর পার্থক্য মাপা।

---

#### **Step 3 — Backward Pass**

```python
l.backward()
dw = w.grad.item()
```

* PyTorch loss এর w অনুযায়ী derivative (gradient) বের করে `w.grad` এ রাখে।
* `.item()` দিয়ে এটাকে Python float এ কনভার্ট করা হচ্ছে।

---

#### **Step 4 — Update Weight**

```python
with torch.no_grad():
    w -= learning_rate * dw
```

* Gradient Descent ফর্মুলা:

$$
w_{\text{new}} = w_{\text{old}} - \eta \times \frac{\partial L}{\partial w}
$$

* `torch.no_grad()` → এই আপডেট ধাপে gradient tracking বন্ধ রাখা।

---

#### **Step 5 — Reset Gradients**

```python
w.grad.zero_()
```

* PyTorch ডিফল্টভাবে গ্রেডিয়েন্ট জমিয়ে রাখে, তাই প্রতিবার নতুন লুপের আগে শূন্য করা হয়।

---

#### **Step 6 — Print Progress**

```python
if epoch % 10 == 0:
    print(f'Epoch {epoch + 1}/{n_iters}, Loss: {l.item():.4f}, w: {w.item():.4f}')
```

* প্রতি 10 epoch পর পর লস আর w প্রিন্ট করা হয় যাতে প্রগ্রেস দেখা যায়।

---

### **9️⃣ Final Prediction**

```python
print(f'Prediction after training: {forward(torch.tensor(5.0)).item():.3f}')
```

* ট্রেনিং শেষ হওয়ার পরে $w \approx 2$ হওয়া উচিত।
* তাই $x = 5$ দিলে প্রেডিকশন প্রায় 10 আসবে।

---

## **Summary**

* এই কোড **একটা সিম্পল লিনিয়ার রিগ্রেশন মডেল** ট্রেন করছে PyTorch দিয়ে।
* মূল কনসেপ্ট:

  1. **Forward Pass** → Prediction
  2. **Loss Calculation** → Prediction vs True Value
  3. **Backward Pass** → Gradient Calculation
  4. **Weight Update** → Gradient Descent
  5. **Repeat** until model learns

---

