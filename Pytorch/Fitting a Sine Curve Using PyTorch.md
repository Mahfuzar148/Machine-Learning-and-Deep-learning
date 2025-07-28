
---

# 📄 **Documentation: Fitting a Sine Curve Using PyTorch (Polynomial Regression)**

This example demonstrates how to use **PyTorch tensors**, **autograd**, and **manual gradient descent** to approximate the sine function (`sin(x)`) using a **cubic polynomial** of the form:

$$
y_{\text{pred}} = a + bx + cx^2 + dx^3
$$

---

## 🔢 1. **Import Required Libraries**

```python
import torch
import math
import matplotlib.pyplot as plt
```

* `torch`: PyTorch core library for tensors and autograd.
* `math`: For mathematical constants like π.
* `matplotlib.pyplot`: For plotting the sine curve and the predicted curve.

---

## 📊 2. **Generate Input and Target Data**

```python
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)
```

* `x`: 2000 evenly spaced values between `-π` and `π`.
* `y`: Ground truth, `sin(x)` values — this is what the model should learn to predict.

---

## 🔧 3. **Initialize Learnable Parameters**

```python
a = torch.randn((), requires_grad=True)
b = torch.randn((), requires_grad=True)
c = torch.randn((), requires_grad=True)
d = torch.randn((), requires_grad=True)
```

* Four scalar parameters initialized randomly.
* `requires_grad=True` tells PyTorch to track operations on these tensors so gradients can be calculated.

---

## 🧪 4. **Set Learning Rate**

```python
learning_rate = 1e-6
```

* Controls how much the parameters are updated per iteration.

---

## 🔁 5. **Training Loop (2000 Iterations)**

```python
for t in range(2000):
    ...
```

### ➤ a. **Forward Pass**

```python
y_pred = a + b * x + c * x**2 + d * x**3
```

* Predicts y using the polynomial equation.

### ➤ b. **Loss Calculation**

```python
loss = (y_pred - y).pow(2).sum()
```

* Uses **sum of squared errors** to measure the difference between predicted and actual values.

### ➤ c. **Print Loss Every 200 Steps**

```python
if t % 200 == 0:
    print(f"Step {t}, Loss = {loss.item():.4f}")
```

### ➤ d. **Backward Pass (Compute Gradients)**

```python
loss.backward()
```

* Calculates the gradients of `loss` with respect to `a, b, c, d`.

### ➤ e. **Manually Update Parameters**

```python
with torch.no_grad():
    a -= learning_rate * a.grad
    b -= learning_rate * b.grad
    c -= learning_rate * c.grad
    d -= learning_rate * d.grad

    a.grad = None
    b.grad = None
    c.grad = None
    d.grad = None
```

* `with torch.no_grad()` disables gradient tracking during updates.
* Parameters are updated using **gradient descent**.
* `.grad = None` clears old gradients.

---

## 📣 6. **Print Final Parameters**

```python
print(f"\nLearned parameters:\na = {a.item():.4f}, b = {b.item():.4f}, c = {c.item():.4f}, d = {d.item():.4f}")
```

* Displays the final learned values of `a`, `b`, `c`, and `d`.

---

## 📈 7. **Plot Results**

```python
plt.plot(x.numpy(), y.numpy(), label='True sin(x)')
plt.plot(x.numpy(), y_pred.detach().numpy(), label='Learned Curve', linestyle='--')
plt.legend()
plt.title("Polynomial Fit to sin(x)")
plt.grid(True)
plt.show()
```

* Plots both the **original sine wave** and the **learned polynomial curve**.
* `y_pred.detach()` detaches prediction from the computation graph for plotting.

---

## ✅ Summary

| Step                    | Purpose                                            |
| ----------------------- | -------------------------------------------------- |
| Generate `x`, `y`       | Create the sine dataset                            |
| Initialize `a, b, c, d` | Learnable polynomial parameters                    |
| `loss.backward()`       | Compute gradients for training                     |
| Manual update           | Perform gradient descent without optimizer         |
| Plot                    | Visualize how well the model learned the sine wave |

---

---

# ✅ Full Code: Fit `sin(x)` using a Cubic Polynomial

```python
# Step 1: Import Libraries
import torch
import math
import matplotlib.pyplot as plt

# Step 2: Create input (x) and target output (y)
x = torch.linspace(-math.pi, math.pi, 2000)  # 2000 points between -π and π
y = torch.sin(x)                             # Ground truth: sin(x)

# Step 3: Initialize parameters a, b, c, d (learnable scalars)
a = torch.randn((), requires_grad=True)
b = torch.randn((), requires_grad=True)
c = torch.randn((), requires_grad=True)
d = torch.randn((), requires_grad=True)

# Step 4: Set learning rate
learning_rate = 1e-6  # Small value to control update speed

# Step 5: Training loop - try 2000 times to get better
for t in range(2000):
    # Step 5.1: Compute the predicted y (forward pass)
    y_pred = a + b * x + c * x**2 + d * x**3

    # Step 5.2: Calculate the loss (difference between prediction and actual y)
    loss = (y_pred - y).pow(2).sum()

    # Step 5.3: Print the loss every 200 steps
    if t % 200 == 0:
        print(f"Step {t}, Loss = {loss.item():.4f}")

    # Step 5.4: Compute gradients using backward()
    loss.backward()

    # Step 5.5: Manually update parameters using gradients
    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad

        # Step 5.6: Zero out the gradients for the next step
        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None

# Step 6: Print final learned parameters
print(f"\nLearned parameters:")
print(f"a = {a.item():.4f}")
print(f"b = {b.item():.4f}")
print(f"c = {c.item():.4f}")
print(f"d = {d.item():.4f}")

# Step 7: Plot true sin(x) vs learned polynomial curve
plt.plot(x.numpy(), y.numpy(), label='True sin(x)', color='blue')
plt.plot(x.numpy(), y_pred.detach().numpy(), label='Learned Polynomial', color='orange', linestyle='--')
plt.title("Polynomial Approximation of sin(x)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
```

---

# 📘 Explanation of All Steps

## 🧩 Step 1: Import Libraries

* **torch** – for tensor operations and automatic differentiation.
* **math** – for using mathematical constants like `π`.
* **matplotlib.pyplot** – for plotting graphs.

---

## 📊 Step 2: Create Input & Target Data

```python
x = torch.linspace(-π, π, 2000) → Input (x-axis values)

y = torch.sin(x)                → Ground truth (target y values)
```

---

## 🔢 Step 3: Initialize Parameters

We create 4 **scalar parameters** `a`, `b`, `c`, `d` with:

```python
requires_grad=True
```

This allows PyTorch to compute their **gradients** during training.

---

## ⚙️ Step 4: Learning Rate

A small value (`1e-6`) that controls how **fast** we update the parameters. If it's too big, training might fail; if too small, training will be slow.

---

## 🔁 Step 5: Training Loop (2000 Iterations)

We update the parameters to reduce the difference between predicted `y` and actual `y`.

### ✅ Step-by-step inside the loop:

| Sub-Step | Description                                   |
| -------- | --------------------------------------------- |
| 5.1      | Use current `a, b, c, d` to predict `y`       |
| 5.2      | Measure how wrong the prediction is (loss)    |
| 5.3      | Print the loss to see training progress       |
| 5.4      | `loss.backward()` computes gradients          |
| 5.5      | Manually update each parameter using gradient |
| 5.6      | Clear gradients before the next iteration     |

---

## 🧾 Step 6: Print Final Parameters

Shows the values of `a, b, c, d` after training. These define the polynomial that best fits the sine curve.

---

## 📈 Step 7: Plot the Results

We visualize both:

* The **true sine wave** (`y = sin(x)`)
* The **polynomial curve** predicted by the learned model

---

## 🧠 What You Learn from This

| Concept          | What You See                  |
| ---------------- | ----------------------------- |
| Tensors          | `x`, `y`, parameters          |
| Autograd         | `.backward()` and `.grad`     |
| Gradient Descent | Manual parameter updates      |
| Curve Fitting    | Polynomial approximating sine |
| Visualization    | Plotting results              |

---

