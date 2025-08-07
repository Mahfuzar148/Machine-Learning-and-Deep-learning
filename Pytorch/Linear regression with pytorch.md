
---

# ✅ Full Code: Linear Regression with PyTorch

```python
# 1) Design model (input, output, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop:
#    forward pass -> compute prediction
#    backward pass -> compute gradients
#    update weights

import torch
import torch.nn as nn
import torch.optim as optim

# Our dataset: y = 2 * x
x = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

# Test input (to check predictions)
x_test = torch.tensor([[5]], dtype=torch.float32)

# Get dataset dimensions
n_samples, n_features = x.shape
print(f'Number of samples: {n_samples}, Number of features: {n_features}')

# Model parameters
input_size = n_features
output_size = 1  # We want a single output

# Define a simple linear model: y = w*x + b
model = nn.Linear(input_size, output_size)

# Check prediction before training
print(f'Prediction before training: {model(x_test).item():.3f}')

# Hyperparameters
learning_rate = 0.01
n_iters = 10000

# Loss function: Mean Squared Error
loss = nn.MSELoss()

# Optimizer: Stochastic Gradient Descent
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(n_iters):
    # Forward pass: compute predictions
    y_pred = model(x)
    
    # Compute loss
    l = loss(y_pred, y)
    
    # Backward pass: compute gradients
    l.backward()
    
    # Update weights
    optimizer.step()
    
    # Zero the gradients (otherwise they accumulate)
    optimizer.zero_grad()
    
    # Log progress every 100 epochs
    if epoch % 100 == 0:
        [w, b] = model.parameters()
        print(f'Epoch {epoch+1}/{n_iters}, Loss: {l.item():.4f}, w: {w[0][0].item():.4f}, b: {b.item():.4f}')

# Final prediction after training
print(f'Prediction after training: {model(x_test).item():.3f}')

```

---

# 🧠 Detailed Explanation

---

## 🔹 Step 1: Input & Output Data (Training Set)

```python
x = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)
```

* `x`: input features, shaped (4, 1) → 4 samples, each with 1 feature.
* `y`: corresponding labels → linearly dependent: $y = 2x$

> ✅ Always use 2D shape: (batch\_size, features)

---

## 🔹 Step 2: Test Data

```python
x_test = torch.tensor([[5]], dtype=torch.float32)
```

* Used to evaluate the model's prediction **before and after training**.
* Also shaped (1, 1) → one test sample with 1 feature.

---

## 🔹 Step 3: Model Definition

```python
model = nn.Linear(input_size, output_size)
```

* `nn.Linear(in_features, out_features)` defines a layer with:

  * **Weight**: shape (out\_features, in\_features)
  * **Bias**: shape (out\_features,)
* It models the equation:

  $$
  \hat{y} = x \cdot w^T + b
  $$

> 🔍 Since `input_size = 1`, `output_size = 1` → `weight` shape = (1,1)

---

## 🔹 Prediction before Training

```python
print(f'Prediction before training: {model(x_test).item():.3f}')
```

* Before training, weights are random, so prediction will be inaccurate.
* `.item()` gets scalar from single-element tensor.

---

## 🔹 Step 4: Training Configuration

```python
learning_rate = 0.01
n_iters = 10000
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
```

### 🔸 Loss Function: `nn.MSELoss()`



---

**Loss Function:** `nn.MSELoss()`

**Mean Squared Error:**

$\text{Loss} = \frac{1}{n} \sum (y - \hat{y})^2$'

Measures how far predicted values are from actual labels.

---


### 🔸 Optimizer: `SGD`

* Stochastic Gradient Descent updates model parameters using gradients.
* `model.parameters()` returns weights and bias for the optimizer.

---

## 🔹 Step 5: Training Loop

```python
for epoch in range(n_iters):
```

### ✅ Forward Pass

```python
y_pred = model(x)
```

* Input `x` is passed through the model to compute predictions.

---

### ✅ Loss Computation

```python
loss = loss_fn(y_pred, y)
```

* Loss measures difference between predictions and actual values.

---

### ✅ Backward Pass (Backpropagation)

```python
loss.backward()
```

* Calculates gradients of loss w\.r.t model parameters using autograd.
* Builds computational graph and propagates gradients backward.

---

### ✅ Update Weights

```python
optimizer.step()
```

* Applies gradients to update weights and bias in the opposite direction of the gradient.

---

### ✅ Reset Gradients

```python
optimizer.zero_grad()
```

* PyTorch accumulates gradients, so we reset them after each update.

---

### ✅ Logging

```python
if epoch % 1000 == 0:
    w = model.weight
    b = model.bias
    print(f'Epoch {epoch+1}/{n_iters}, Loss: {loss.item():.4f}, w: {w.item():.4f}, b: {b.item():.4f}')
```

* Every 1000 iterations, we print:

  * Epoch
  * Current loss
  * Weight value
  * Bias value

> `.item()` only works because weight and bias have single values (shape = (1,1) and (1,))

---

## 🔹 Final Prediction

```python
print(f'Prediction after training: {model(x_test).item():.3f}')
```

* Model should now predict **close to 10** for input 5
  Because the true function is: $y = 2x$

---

# 📊 Summary Table

| Component      | Purpose                                 |
| -------------- | --------------------------------------- |
| `nn.Linear`    | Defines the model equation $y = wx + b$ |
| `MSELoss`      | Measures prediction error               |
| `SGD`          | Optimizer that updates weights          |
| `.backward()`  | Calculates gradients via autograd       |
| `.step()`      | Updates model parameters                |
| `.zero_grad()` | Clears accumulated gradients            |
| `.item()`      | Extracts float from 1-element tensor    |

---

### ✅ Final Notes:

* Always keep input 2D in shape: `(batch_size, features)`
* `w` is always 2D: shape = `(out_features, in_features)`
* `b` is always 1D: shape = `(out_features,)`
* Loss goes down as training progresses if learning rate is appropriate

---



---

## ✅ Code: Simple Linear Regression with PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim

# ========== Step 1: Create Dataset ==========

# Features (input): x
# Labels (target output): y = 2 * x
x = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)  # shape: (4, 1)
y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)  # shape: (4, 1)

# Test input (for prediction)
x_test = torch.tensor([[5]], dtype=torch.float32)  # shape: (1, 1)

# ========== Step 2: Define Model ==========

# Model input size and output size
input_size = 1
output_size = 1

# Linear model: y = w * x + b
model = nn.Linear(input_size, output_size)

# ========== Step 3: Define Loss Function ==========

# Mean Squared Error Loss
loss = nn.MSELoss()

# ✳️ MSE Loss Formula:
# Loss = (1/n) * Σ(y - ŷ)^2
# Measures how far predicted values are from actual labels

# ========== Step 4: Define Optimizer ==========

# Stochastic Gradient Descent optimizer
learning_rate = 0.01
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# ========== Step 5: Training Loop ==========

n_iters = 10000

for epoch in range(n_iters):
    # Forward pass: compute prediction
    y_pred = model(x)

    # Compute loss
    l = loss(y_pred, y)

    # Backward pass: compute gradients
    l.backward()

    # Update weights using optimizer
    optimizer.step()

    # Reset gradients to zero before next step
    optimizer.zero_grad()

    # Log every 100 epochs
    if epoch % 100 == 0:
        [w, b] = model.parameters()
        print(f'Epoch {epoch+1}/{n_iters}, Loss: {l.item():.4f}, w: {w[0][0].item():.4f}, b: {b.item():.4f}')

# ========== Step 6: Prediction After Training ==========

print(f'Prediction for input 5: {model(x_test).item():.3f}')
```

---

## ✅ Explanation (Line by Line)

---

### 🔹 Step 1: Dataset Creation

```python
x = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)
```

* `x` and `y` are **2D tensors** (shape: `[n_samples, 1]`)
* Required shape for `nn.Linear` which expects `(batch_size, input_features)`
* Each row is a separate training example.

---

### 🔹 Step 2: Model

```python
model = nn.Linear(input_size, output_size)
```

* `nn.Linear(1, 1)` creates a simple linear model with:

  * One weight `w` (shape: `[1, 1]`)
  * One bias `b` (shape: `[1]`)
* It implements:

  $$
  \hat{y} = x \cdot w^T + b
  $$

---

### 🔹 Step 3: Loss Function

```python
loss = nn.MSELoss()
```

* **Mean Squared Error Loss**
* Formula:

  $$
  \text{Loss} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
  $$
* Measures how far predicted values (`ŷ`) are from actual values (`y`).
* Lower loss means better predictions.

---

### 🔹 Step 4: Optimizer

```python
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
```

* **SGD (Stochastic Gradient Descent)** optimizer updates the model parameters:

  $$
  \theta = \theta - \alpha \cdot \nabla_\theta \text{Loss}
  $$

  * `θ` = model weights and bias
  * `α` = learning rate

---

### 🔹 Step 5: Training Loop

```python
for epoch in range(n_iters):
```

* Loop runs 10,000 times to allow the model to learn the correct weight and bias.

#### 🔸 Forward Pass

```python
y_pred = model(x)
```

* Model makes predictions for all inputs in `x`.

#### 🔸 Compute Loss

```python
l = loss(y_pred, y)
```

* Calculates how far predictions are from true outputs.

#### 🔸 Backward Pass

```python
l.backward()
```

* Computes gradients of loss w\.r.t. all model parameters (`w` and `b`) using backpropagation.

#### 🔸 Update Weights

```python
optimizer.step()
```

* Updates model parameters using the gradients computed in `.backward()`.

#### 🔸 Clear Gradients

```python
optimizer.zero_grad()
```

* Gradients accumulate in PyTorch, so we reset them to zero before the next update.

#### 🔸 Logging

```python
[w, b] = model.parameters()
print(...)
```

* Extract weight `w` and bias `b` values from the model and log progress.

---

### 🔹 Step 6: Final Prediction

```python
print(f'Prediction for input 5: {model(x_test).item():.3f}')
```

* After training, test the model by predicting the output for `x = 5`.
* Expected output: close to `10`, since `y = 2 * x`

---

## 🔍 Bonus: Weight & Bias Shapes

```python
model.weight.shape  # torch.Size([1, 1])
model.bias.shape    # torch.Size([1])
```

* Weight is always 2D in `nn.Linear`
* Bias is always 1D

---



