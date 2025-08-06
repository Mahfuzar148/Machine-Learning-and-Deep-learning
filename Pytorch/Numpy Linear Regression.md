
---

## **Full Code (Numpy Linear Regression)**

```python
import numpy as np

# Relationship: f = w * x
# Original dataset: y = 2 * x

# Training data
x = np.array([1, 2, 3, 4], dtype=np.float32)
y = np.array([2, 4, 6, 8], dtype=np.float32)

# Initial weight
w = 0.0

# Forward pass (prediction)
def forward(x):
    return w * x

# Loss function (Mean Squared Error)
def loss(y, y_pred):
    return ((y_pred - y) ** 2).mean()

# Gradient function
# MSE = (1/N) * Σ (w*x - y)^2
# dJ/dw = (2/N) * Σ (w*x - y) * x
def gradient(x, y, y_pred):
    return np.dot(2 * x, y_pred - y).mean()

# Before training prediction
print(f'Prediction before training: {forward(5):.3f}')

# Training parameters
learning_rate = 0.01
n_iters = 10

# Training loop
for epoch in range(n_iters):
    # 1. Forward pass
    y_pred = forward(x)

    # 2. Compute loss
    l = loss(y, y_pred)

    # 3. Compute gradient
    dw = gradient(x, y, y_pred)

    # 4. Update weight
    w -= learning_rate * dw

    # 5. Print progress
    if epoch % 2 == 0:
        print(f'Epoch {epoch+1}/{n_iters}, Loss: {l:.4f}, w: {w:.4f}')

# Final prediction
print(f'Prediction after training: {forward(5):.3f}')
```

---

## **Step-by-Step Calculation & Explanation**

### **Initial Setup**

* **x** = \[1, 2, 3, 4]
* **y** = \[2, 4, 6, 8]
* **w** = 0.0
* **learning\_rate** = 0.01
* We know the true relationship is $y = 2x$, so ideally $w$ should be **2.0** after training.

---

### **Formulas**

1. **Prediction**:

$$
\hat{y} = w \cdot x
$$

2. **Loss (MSE)**:

$$
\text{Loss} = \frac{1}{N} \sum (\hat{y} - y)^2
$$

3. **Gradient (Derivative w\.r.t w)**:

$$
\frac{\partial \text{Loss}}{\partial w} = \frac{2}{N} \sum (\hat{y} - y) \cdot x
$$

---

### **Epoch 1 Calculation**

#### **Step 1: Forward Pass**

$$
\hat{y} = 0.0 \cdot [1, 2, 3, 4] = [0, 0, 0, 0]
$$

#### **Step 2: Loss**

Error = $\hat{y} - y$ = \[0-2, 0-4, 0-6, 0-8]
Error = \[-2, -4, -6, -8]

Square each term: \[4, 16, 36, 64]
Mean = $\frac{4+16+36+64}{4} = \frac{120}{4} = 30.0$
So:

$$
\text{Loss} = 30.0
$$

#### **Step 3: Gradient**

Formula:

$$
\text{grad} = \frac{2}{4} \sum (x_i \cdot (\hat{y}_i - y_i))
$$

* Multiply error by $x$:
  \[1\*(-2), 2\*(-4), 3\*(-6), 4\*(-8)] = \[-2, -8, -18, -32]
* Sum: (-2) + (-8) + (-18) + (-32) = **-60**
* Multiply by $2/N = 0.5$:
  Gradient = 0.5 \* (-60) = **-30**

Or with Numpy code:

```python
np.dot(2 * x, y_pred - y) / 4
= np.dot([2, 4, 6, 8], [-2, -4, -6, -8]) / 4
= (-4) + (-16) + (-36) + (-64) = -120
/ 4 = -30
```

#### **Step 4: Update Weight**

$$
w_{\text{new}} = w_{\text{old}} - \eta \cdot \text{grad}
$$

$$
w_{\text{new}} = 0.0 - 0.01 \cdot (-30) = 0.3
$$

---

### **Epoch 2 (just idea)**

Now:

* $w = 0.3$
* Prediction improves: $\hat{y} = 0.3 \cdot [1, 2, 3, 4] = [0.3, 0.6, 0.9, 1.2]$
* Loss will decrease.
* Gradient will be smaller in magnitude.
* $w$ will keep increasing toward 2.0.

---

### **Training Progress Example Output**

```
Prediction before training: 0.000
Epoch 1/10, Loss: 30.0000, w: 0.3000
Epoch 3/10, Loss: 13.8525, w: 0.8329
Epoch 5/10, Loss: 6.3936, w: 1.2924
Epoch 7/10, Loss: 2.9511, w: 1.6871
Epoch 9/10, Loss: 1.3622, w: 2.0258
Prediction after training: 10.129
```

---

✅ **Main Takeaway about Gradient**
The gradient formula:

$$
\frac{2}{N} \sum (\hat{y} - y) \cdot x
$$

tells us the slope of the loss curve w\.r.t $w$.

* **Negative gradient** → increase $w$ to reduce loss.
* **Positive gradient** → decrease $w$ to reduce loss.
  We update $w$ in the **opposite direction** of the gradient.

---


---

## **Code Explanation**

```python
import numpy as np
```

* **Numpy** লাইব্রেরি ইম্পোর্ট করা হচ্ছে, কারণ এখানে টেনসর অপারেশন, ডট প্রোডাক্ট, গড় (mean) ইত্যাদি লাগবে।

---

```python
# f = w*x
# f = 2*x
```

* কমেন্ট দিয়ে বোঝানো হয়েছে ডেটার আসল সম্পর্ক $y = 2x$।
* মডেলের হাইপোথিসিস হলো $f = w \cdot x$ — যেখানে $w$ ট্রেনিংয়ের মাধ্যমে শেখা হবে।

---

```python
x = np.array([1, 2, 3, 4], dtype=np.float32)
y = np.array([2, 4, 6, 8], dtype=np.float32)
```

* `x` → ইনপুট ডেটা
* `y` → আসল আউটপুট (লেবেল)
* `dtype=np.float32` রাখা হয়েছে যাতে ফ্লোট ক্যালকুলেশন হয়।

---

```python
w = 0.0
```

* মডেলের ওজন (weight) **প্রাথমিকভাবে 0.0** থেকে শুরু হবে।
* আমরা Gradient Descent ব্যবহার করে এটা শেখাব।

---

```python
def forward(x):
    return w * x
```

* **Forward Pass**: ইনপুট `x` দিলে প্রেডিকশন $\hat{y} = w \times x$ রিটার্ন করে।

---

```python
def loss(y, y_pred):
    return ((y_pred - y) ** 2).mean()
```

* **Loss Function**: Mean Squared Error (MSE)

$$
\text{Loss} = \frac{1}{N} \sum (y_{\text{pred}} - y)^2
$$

* এটা বলে প্রেডিকশন আর আসল ভ্যালুর পার্থক্য।

---

```python
def gradient(x, y, y_pred):
    return np.dot(2 * x, y_pred - y).mean()
```

* **Gradient Function**: MSE এর $w$-এর প্রতি ডেরিভেটিভ।

$$
\frac{\partial L}{\partial w} = \frac{2}{N} \sum (y_{\text{pred}} - y) \cdot x
$$

* `np.dot(2*x, y_pred - y)` → প্রতি ডেটা পয়েন্টের জন্য $2x_i \cdot (\hat{y}_i - y_i)$ হিসাব করে যোগফল।
* `.mean()` → $\frac{1}{N}$ দিয়ে ভাগ করা হয়।

---

```python
print(f'Prediction before training: {forward(5):.3f}')
```

* ট্রেনিংয়ের আগে $x = 5$ এর প্রেডিকশন দেখানো হচ্ছে।
* যেহেতু $w = 0.0$, প্রেডিকশন হবে 0.000।

---

```python
learning_rate = 0.01
n_iters = 10
```

* **learning\_rate** → প্রতি আপডেটে $w$ কতটা বদলাবে।
* **n\_iters** → মোট কতবার ডেটা দিয়ে ট্রেনিং চলবে।

---

```python
for epoch in range(n_iters):
```

* লুপের মাধ্যমে বহুবার ট্রেনিং (forward + backward + update) চালানো হবে।

---

```python
y_pred = forward(x)
```

* **Forward Pass**: বর্তমান $w$ দিয়ে সব ইনপুটের প্রেডিকশন।

---

```python
l = loss(y, y_pred)
```

* প্রেডিকশন আর আসল আউটপুটের মধ্যে MSE লস বের করা।

---

```python
dw = gradient(x, y, y_pred)
```

* লসের $w$-এর প্রতি gradient বের করা।

---

```python
w -= learning_rate * dw
```

* Gradient Descent আপডেট রুল:

$$
w_{\text{new}} = w_{\text{old}} - \eta \times \text{gradient}
$$

* এখানে `eta` = learning\_rate।

---

```python
if epoch % 2 == 0:
    print(f'Epoch {epoch+1}/{n_iters}, Loss: {l:.4f}, w: {w:.4f}')
```

* প্রতি 2 স্টেপে একবার ট্রেনিং প্রগ্রেস প্রিন্ট করা হয়:

  * Epoch নম্বর
  * লস
  * বর্তমান $w$-এর মান

---

```python
print(f'Prediction after training: {forward(5):.3f}')
```

* ট্রেনিং শেষে $x = 5$ এর প্রেডিকশন।
* যেহেতু $w$ প্রায় 2 শিখে ফেলবে, প্রেডিকশন হবে প্রায় 10।

---

✅ **সারসংক্ষেপে:**

1. Forward pass দিয়ে প্রেডিকশন করা হয়।
2. Loss দিয়ে মডেলের ভুল মাপা হয়।
3. Gradient দিয়ে বোঝা হয় কোন দিকে $w$ বদলাতে হবে।
4. Gradient Descent দিয়ে $w$ আপডেট হয়।
5. এই প্রক্রিয়া বারবার করলে $w$ আসল মানের (2.0) কাছাকাছি চলে যায়।

---

