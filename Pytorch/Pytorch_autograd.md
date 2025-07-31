
---

# 🔧 What is Autograd in PyTorch?

**Autograd** is PyTorch's automatic differentiation system. It automatically calculates **gradients** (partial derivatives) for **tensor operations**, making it easy to implement and train neural networks.

---

## 📌 Why Is Autograd Important?

In training deep learning models, you:

1. Compute a **loss function** from the output.
2. Use **gradients** to update the model parameters via **gradient descent**.
3. PyTorch's `autograd` automates this gradient calculation step.

---

## ✅ Key Concepts

| Term            | Meaning                                                                     |
| --------------- | --------------------------------------------------------------------------- |
| `requires_grad` | Tells PyTorch to track operations on this tensor.                           |
| `grad`          | Stores the gradient (derivative) of a tensor after `.backward()` is called. |
| `backward()`    | Computes all gradients for the computation graph.                           |
| `grad_fn`       | A function that tracks how the tensor was created.                          |

---

# 🧪 Step-by-Step Example

### 🔹 Step 1: Import and Create Tensor

```python
import torch

x = torch.tensor([2.0, 3.0], requires_grad=True)
print(x)
```

* `requires_grad=True`: PyTorch will track this tensor for gradient computation.

---

### 🔹 Step 2: Perform Operations

```python
y = x * x + 2
print(y)
```

* This creates a new tensor using `x`, and PyTorch builds a **computation graph**.
* `y` is now a function of `x`.

---

### 🔹 Step 3: Get a Scalar and Call `.backward()`

```python
z = y.sum()  # Convert vector to scalar (required for backward)
z.backward()
```

* PyTorch computes gradients:
  ∂z/∂x = ∂(x² + 2)/∂x = 2x
  So:

  ```
  x = [2, 3]
  grad = [4, 6]
  ```

---

### 🔹 Step 4: Check Gradients

```python
print(x.grad)  # tensor([4., 6.])
```

* You just computed:
  `dz/dx1 = 4`, `dz/dx2 = 6`

---

# 🔁 Behind the Scenes: Computation Graph

Every operation on a tensor creates a **node in a graph**. For example:

```
x → x*x → x*x + 2 → sum → z
```

Calling `.backward()` **traverses** this graph in reverse and computes the gradient.

---

# ✅ Autograd Example with Chain Rule

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3         # y = x³
z = 2 * y + 1      # z = 2x³ + 1
z.backward()       # Compute dz/dx

print(x.grad)      # Should be 6x² = 6 * 4 = 24
```

✔️ Output:

```
tensor(24.)
```

---

## 📌 Notes on `.backward()`

| Situation              | Notes                                 |
| ---------------------- | ------------------------------------- |
| If `z` is a **scalar** | `.backward()` works directly          |
| If `z` is a **vector** | You must provide `gradient=` argument |

---

### 🔸 Vector Output Example

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x * x
y.backward(torch.tensor([1.0, 1.0, 1.0]))  # Equivalent to summing gradients
print(x.grad)  # tensor([2.0, 4.0, 6.0])
```

---

## 🔁 Zeroing Gradients

Gradients accumulate by default. You must clear them manually during training.

```python
x.grad.zero_()  # reset gradients to zero
```

---

## 🔐 Disable Gradient Tracking (for inference)

Use `torch.no_grad()` during evaluation or inference to save memory and computation.

```python
with torch.no_grad():
    y = x * 2
```

Or:

```python
x = torch.tensor([1.0], requires_grad=True)
print(x.requires_grad)  # True

x_detached = x.detach()
print(x_detached.requires_grad)  # False
```

---

# 🧠 Summary: What to Remember

| Concept              | What It Does                               |
| -------------------- | ------------------------------------------ |
| `requires_grad=True` | Tracks operations for gradient             |
| `.backward()`        | Computes gradients                         |
| `.grad`              | Stores computed gradients                  |
| `torch.no_grad()`    | Disables tracking for performance          |
| `.detach()`          | Returns a new tensor without grad tracking |
| `.zero_()`           | Clears gradients                           |

---

## 🧪 Real Example: Linear Regression with Autograd

```python
# y = wx + b (manual implementation)

x = torch.tensor([1.0, 2.0, 3.0])
y_true = torch.tensor([2.0, 4.0, 6.0])

w = torch.tensor(0.0, requires_grad=True)

for epoch in range(10):
    y_pred = w * x
    loss = ((y_pred - y_true) ** 2).mean()

    loss.backward()
    print(f"Epoch {epoch}: w = {w.item()}, loss = {loss.item()}")

    # Update
    with torch.no_grad():
        w -= 0.1 * w.grad
        w.grad.zero_()
```


---
Absolutely! Here's a **clear, complete documentation** of your PyTorch autograd example, with code, explanation, and full gradient calculation.

---

# 📘 PyTorch Autograd Tutorial — Full Documentation

This tutorial walks through:

* Tensor creation with `requires_grad`
* Computation graph
* How PyTorch tracks operations
* Manual gradient calculation (step-by-step)

---

## ✅ Code Overview

```python
import torch

x = torch.randn(3, requires_grad=True)
print("x =", x)

y = x + 2
print("y =", y)

z = y * y * 3
z = z.mean()
print("z =", z)

z.backward()
print("x.grad =", x.grad)
```

---

## 🔍 Explanation of Each Line

---

### 🔹 `x = torch.randn(3, requires_grad=True)`

* Creates a **tensor of shape (3,)** with random values.
* `requires_grad=True` means PyTorch will **track all operations** on `x` to compute gradients later.

> 💡 Use this for any model parameters you want to update via gradient descent.

---

### 🔹 `y = x + 2`

* Element-wise addition.
* Creates a **new tensor** `y`, and PyTorch links it to `x` in the **computation graph**.

> This operation is now tracked in the autograd system.

---

### 🔹 `z = y * y * 3`

* Element-wise:

  ```python
  z_i = 3 * y_i^2
  ```

* Still tracked in computation graph.

---

### 🔹 `z = z.mean()`

* Calculates the average of all elements in `z`.

> At this point, `z` is a **scalar**, and you can call `z.backward()`.

---

### 🔹 `z.backward()`

* **Backpropagation step**: Computes the gradients of `z` with respect to `x`.

> Stores the result in `x.grad`.

---

### 🔹 `print(x.grad)`

* Prints the gradient ∂z/∂x.

---

## 🧠 Computation Graph Overview

```
x → y = x + 2 → z = mean(3 * y²)
```

Autograd builds this graph dynamically as operations occur.

---

## 🧮 Manual Gradient Derivation

We want to compute:

$$
\frac{dz}{dx_i}
$$

Let’s break it down:

### Step 1: Let

$$
y_i = x_i + 2
$$

### Step 2:

$$
z_i = 3 * y_i^2
$$

### Step 3:

$$
z = \frac{1}{3} (z_0 + z_1 + z_2) = \text{mean}
$$

---

### ✅ Apply Chain Rule:

$$
\frac{dz}{dx_i} = \frac{dz}{dy_i} \cdot \frac{dy_i}{dx_i}
$$

Where:

* $$
  $$

\frac{dz}{dy\_i} = \frac{1}{3} \cdot \frac{d(3y\_i^2)}{dy\_i} = \frac{1}{3} \cdot 6y\_i = 2y\_i
]

* $$
  $$

\frac{dy\_i}{dx\_i} = 1
]

Therefore:

$$
\frac{dz}{dx_i} = 2y_i = 2(x_i + 2)
$$

---

## 🔢 Numerical Example

Suppose:

```python
x = tensor([1.0, 2.0, 3.0], requires_grad=True)
```

Then:

* `y = x + 2 = [3.0, 4.0, 5.0]`
* `z = 3 * y^2 = [27.0, 48.0, 75.0]`
* `mean(z) = (27 + 48 + 75)/3 = 150 / 3 = 50.0`

Now:

$$
x.grad = 2 * (x + 2) = 2 * [3, 4, 5] = [6.0, 8.0, 10.0]
$$

✔️ Final gradient:

```python
x.grad = tensor([6., 8., 10.])
```

---

## 🧾 Summary

| Step                 | Description                                   |
| -------------------- | --------------------------------------------- |
| `requires_grad=True` | Enables gradient tracking                     |
| `x + 2`              | Adds a constant (tracked operation)           |
| `* y * 3`            | Element-wise operation                        |
| `.mean()`            | Reduces to scalar (needed for `.backward()`)  |
| `.backward()`        | Triggers automatic gradient calculation       |
| `x.grad`             | Contains gradient of final output w\.r.t. `x` |

---

## ✅ Output Example

```python
x = tensor([1., 2., 3.], requires_grad=True)
y = tensor([3., 4., 5.])
z = tensor(50.0, grad_fn=<MeanBackward0>)
x.grad = tensor([6., 8., 10.])
```

---

