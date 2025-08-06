
---

## **Full Code**

```python
import torch

# Step 1: Initialize input, true output, and weight
x = torch.tensor(1.0)                         # Input value
y = torch.tensor(2.0)                         # True value
w = torch.tensor(1.0, requires_grad=True)     # Weight (trainable parameter)

# Step 2: Forward pass (prediction)
y_hat = w * x                                 # Predicted output: ŷ = w * x

# Step 3: Compute loss (Squared Error)
loss = (y_hat - y) ** 2                       # Loss = (ŷ - y)²

# Step 4: Backward pass (compute gradient)
loss.backward()                               # Computes dLoss/dw

# Step 5: Print results
print(f"Prediction (y_hat): {y_hat.item()}")
print(f"Loss: {loss.item()}")
print(f"Gradient (dLoss/dw): {w.grad.item()}")

# Step 6: Update weight manually (Gradient Descent)
learning_rate = 0.1
with torch.no_grad():                         # Disable gradient tracking for update
    w -= learning_rate * w.grad               # w_new = w - η * gradient

# Step 7: Zero the gradients for next iteration
w.grad.zero_()

print(f"Updated weight: {w.item()}")
```

---

## **Explanation**

### **1️⃣ Import Library**

```python
import torch
```

Loads PyTorch, which handles tensors, automatic differentiation, and neural network operations.

---

### **2️⃣ Initialize variables**

```python
x = torch.tensor(1.0)  
y = torch.tensor(2.0)
w = torch.tensor(1.0, requires_grad=True)
```

* **`x`**: Input feature.
* **`y`**: True/actual value (label).
* **`w`**: Model’s weight parameter.

  * `requires_grad=True` tells PyTorch to track operations for gradient computation.

---

### **3️⃣ Forward pass**

```python
y_hat = w * x
```

Prediction:

$$
\hat{y} = w \cdot x
$$

If $w = 1.0$ and $x = 1.0$, then $\hat{y} = 1.0$.

---

### **4️⃣ Compute loss**

```python
loss = (y_hat - y) ** 2
```

Squared error:

$$
\text{Loss} = (\hat{y} - y)^2
$$

If $\hat{y} = 1.0$ and $y = 2.0$:

$$
\text{Loss} = (1 - 2)^2 = (-1)^2 = 1
$$

---

### **5️⃣ Backward pass**

```python
loss.backward()
```

* Computes derivative:

$$
\frac{\partial \text{Loss}}{\partial w} = 2 \cdot (w \cdot x - y) \cdot x
$$

Substitute $w = 1.0$, $x = 1.0$, $y = 2.0$:

$$
\frac{\partial \text{Loss}}{\partial w} = 2 \cdot (1 - 2) \cdot 1 = -2.0
$$

---

### **6️⃣ Print values**

```python
print(f"Prediction (y_hat): {y_hat.item()}")
print(f"Loss: {loss.item()}")
print(f"Gradient (dLoss/dw): {w.grad.item()}")
```

Shows:

```
Prediction (y_hat): 1.0
Loss: 1.0
Gradient (dLoss/dw): -2.0
```

---

### **7️⃣ Weight update (Gradient Descent)**

```python
learning_rate = 0.1
with torch.no_grad():
    w -= learning_rate * w.grad
```

Gradient descent formula:

$$
w_{\text{new}} = w_{\text{old}} - \eta \cdot \frac{\partial \text{Loss}}{\partial w}
$$

Here:

$$
w_{\text{new}} = 1.0 - 0.1 \cdot (-2.0) = 1.2
$$

---

### **8️⃣ Reset gradients**

```python
w.grad.zero_()
```

PyTorch accumulates gradients by default, so we clear them before the next iteration.

---

