
---

# 📘 Documentation: Computation Graph and Backpropagation

---

## 🔹 1. Overview

This documentation explains how a **computation graph** helps visualize operations in a model, and how **backpropagation** computes gradients used for training.

We use a simple **linear regression model** with:

* Input: `x`
* Weight (parameter): `w`
* True output: `y`
* Prediction: `ŷ = w * x`
* Loss function: Squared Error → `Loss = (ŷ - y)²`

---

## 🔹 2. Computation Graph

A **computation graph** breaks the full function into smaller operations:

```
x → [ * ] → ŷ → [ - ] → s → [ ² ] → L
w ↗          y ↗
```

* `[ * ]`: Multiply `w * x` → ŷ (prediction)
* `[ - ]`: Subtract `ŷ - y` → s (error)
* `[ ² ]`: Square the error → L (loss)

---

## 🔹 3. Forward Pass (Prediction and Loss)

We compute the model's output and the loss.

### Example Values:

* `x = 1`
* `y = 2`
* `w = 1`

### Step-by-step:

1. **Prediction**:
   ŷ = w × x = 1 × 1 = 1

2. **Error**:
   s = ŷ - y = 1 - 2 = -1

3. **Loss**:
   L = s² = (-1)² = 1

---

## 🔹 4. Local Gradients (Each Operation’s Derivatives)

To use backpropagation, we need partial derivatives from each operation.

### Multiplication (ŷ = w × x):

* ∂ŷ/∂w = x = 1
* ∂ŷ/∂x = w = 1

### Subtraction (s = ŷ - y):

* ∂s/∂ŷ = 1
* ∂s/∂y = -1

### Squaring (L = s²):

* ∂L/∂s = 2s = -2

---

## 🔹 5. Backward Pass (Using Chain Rule)

We now compute how loss L changes with respect to each input (w, x, y).

### Step 1: ∂L/∂s

$$
\frac{∂L}{∂s} = 2s = 2 × (-1) = -2
$$

### Step 2: ∂L/∂ŷ

$$
\frac{∂L}{∂ŷ} = \frac{∂L}{∂s} × \frac{∂s}{∂ŷ} = -2 × 1 = -2
$$

### Step 3: ∂L/∂w

$$
\frac{∂L}{∂w} = \frac{∂L}{∂ŷ} × \frac{∂ŷ}{∂w} = -2 × 1 = -2
$$

### Step 4: ∂L/∂x

$$
\frac{∂L}{∂x} = \frac{∂L}{∂ŷ} × \frac{∂ŷ}{∂x} = -2 × 1 = -2
$$

### Step 5: ∂L/∂y

$$
\frac{∂L}{∂y} = \frac{∂L}{∂s} × \frac{∂s}{∂y} = -2 × (-1) = 2
$$

---

## 🔹 6. Final Gradients Summary

| Variable | Gradient (∂L/∂Variable) |
| -------- | ----------------------- |
| w        | -2                      |
| x        | -2                      |
| y        | +2                      |

---

## 🔹 7. General Gradient Formulas

For any x, y, w:

* Prediction:

  $$
  \hat{y} = w × x
  $$

* Loss:

  $$
  L = (\hat{y} - y)^2
  $$

* Gradients:

  * $$
    ∂L/∂w = 2(w×x - y) × x
    $$
  * $$
    ∂L/∂x = 2(w×x - y) × w
    $$
  * $$
    ∂L/∂y = -2(w×x - y)
    $$

---

## 🔹 8. Second Example (Perfect Prediction)

### Input:

* x = 1, y = 2, w = 2

### Forward:

* ŷ = 2 × 1 = 2
* Loss = (2 - 2)² = 0

### Backward:

* ∂L/∂w = 0
* (No update needed because the model is already correct.)

---

## 🔹 9. Weight Update (Gradient Descent)

Using learning rate η, the weight update rule is:

$$
w_{new} = w - η × ∂L/∂w
$$

Example:
If η = 0.1 and ∂L/∂w = -2,
Then:

$$
w = 1 \Rightarrow w_{new} = 1 - 0.1 × (-2) = 1.2
$$

---

## 🔹 10. Summary

| Step            | Purpose                                |
| --------------- | -------------------------------------- |
| Forward Pass    | Compute prediction and loss            |
| Local Gradients | Derivatives of each operation          |
| Backward Pass   | Use chain rule to find total gradients |
| Weight Update   | Adjust weights to reduce loss          |

Backpropagation is how neural networks **learn**. It tells us **how much each weight contributes to the error**, so we can adjust it and improve the model.

---
Absolutely! Let’s build on your understanding by working through **medium** and **complex** examples of computation graphs and backpropagation using step-by-step calculations.

---

# 🔍 More Examples: Computation Graph & Backpropagation

We’ll move from a **medium example** (still 1 layer, but with a bias term), to a **complex example** (2-layer neural network).

---

## 🟦 Example 1: Medium — Linear Regression with Bias

### Model:

$$
\hat{y} = w \cdot x + b
$$

$$
\text{Loss} = (\hat{y} - y)^2
$$

### Given:

* $x = 2$
* $y = 5$
* $w = 1$
* $b = 0$

---

### ✅ Forward Pass:

1. **Prediction**:

$$
\hat{y} = w \cdot x + b = 1 \cdot 2 + 0 = 2
$$

2. **Error**:

$$
s = \hat{y} - y = 2 - 5 = -3
$$

3. **Loss**:

$$
L = s^2 = (-3)^2 = 9
$$

---

### 🔁 Backward Pass (Gradients):

Let’s compute gradients using the **chain rule**.

#### Step 1:

$$
\frac{∂L}{∂s} = 2s = 2 × (-3) = -6
$$

#### Step 2:

$$
\frac{∂s}{∂\hat{y}} = 1
\Rightarrow \frac{∂L}{∂\hat{y}} = -6
$$

#### Step 3:

* $\frac{∂\hat{y}}{∂w} = x = 2$
* $\frac{∂\hat{y}}{∂b} = 1$

Then:

* $\frac{∂L}{∂w} = \frac{∂L}{∂\hat{y}} \cdot \frac{∂\hat{y}}{∂w} = -6 × 2 = -12$
* $\frac{∂L}{∂b} = \frac{∂L}{∂\hat{y}} \cdot \frac{∂\hat{y}}{∂b} = -6 × 1 = -6$

---

### 📌 Final Gradients:

| Variable | Gradient (∂L/∂var) |
| -------- | ------------------ |
| w        | -12                |
| b        | -6                 |

> These tell us: Increase both `w` and `b` to reduce the loss.

---

## 🟨 Example 2: Complex — 2-Layer Neural Network

### Model:

$$
\begin{aligned}
z_1 &= w_1 \cdot x + b_1 \quad &&\text{(Linear)}\\
a_1 &= \text{ReLU}(z_1) \quad &&\text{(Activation)} \\
z_2 &= w_2 \cdot a_1 + b_2 \quad &&\text{(Linear)}\\
\hat{y} &= z_2 \\
L &= (\hat{y} - y)^2 \quad &&\text{(Loss)}
\end{aligned}
$$

---

### Given:

* $x = 1$
* $y = 4$
* $w_1 = 1$, $b_1 = 0$
* $w_2 = 2$, $b_2 = 0$

---

### ✅ Forward Pass:

1. $z_1 = w_1 \cdot x + b_1 = 1 \cdot 1 + 0 = 1$
2. $a_1 = \text{ReLU}(z_1) = \max(0, 1) = 1$
3. $z_2 = w_2 \cdot a_1 + b_2 = 2 \cdot 1 + 0 = 2$
4. $\hat{y} = z_2 = 2$
5. $s = \hat{y} - y = 2 - 4 = -2$
6. $L = s^2 = 4$

---

### 🔁 Backward Pass:

We go in reverse using the **chain rule**:

#### Step 1:

$$
\frac{∂L}{∂s} = 2s = -4
\quad \text{and} \quad \frac{∂L}{∂\hat{y}} = -4
$$

#### Step 2:

$$
\hat{y} = z_2 \Rightarrow \frac{∂L}{∂z_2} = -4
$$

#### Step 3:

$$
z_2 = w_2 \cdot a_1 + b_2
$$

* $\frac{∂z_2}{∂w_2} = a_1 = 1$
* $\frac{∂z_2}{∂a_1} = w_2 = 2$
* $\frac{∂z_2}{∂b_2} = 1$

Then:

* $\frac{∂L}{∂w_2} = -4 \cdot 1 = -4$
* $\frac{∂L}{∂b_2} = -4 \cdot 1 = -4$
* $\frac{∂L}{∂a_1} = -4 \cdot 2 = -8$

---

### Step 4: Backprop through ReLU

ReLU:

$$
a_1 = \text{ReLU}(z_1) = \max(0, z_1)
\Rightarrow \frac{da_1}{dz_1} = 1 \text{ (since } z_1 = 1 > 0\text{)}
$$

So:

$$
\frac{∂L}{∂z_1} = \frac{∂L}{∂a_1} \cdot \frac{da_1}{dz_1} = -8 × 1 = -8
$$

---

### Step 5: Backprop to w₁ and b₁

$$
z_1 = w_1 \cdot x + b_1
\Rightarrow \frac{∂z_1}{∂w_1} = x = 1,\quad \frac{∂z_1}{∂b_1} = 1
$$

Then:

* $\frac{∂L}{∂w_1} = -8 \cdot 1 = -8$
* $\frac{∂L}{∂b_1} = -8 \cdot 1 = -8$

---

### 📌 Final Gradients:

| Variable | Gradient (∂L/∂var) |
| -------- | ------------------ |
| w₁       | -8                 |
| b₁       | -8                 |
| w₂       | -4                 |
| b₂       | -4                 |

---

## 🧠 Summary of Insights:

| Concept                  | Key Point                                                              |
| ------------------------ | ---------------------------------------------------------------------- |
| Computation Graph        | Breaks model into nodes to track gradients easily                      |
| Forward Pass             | Calculates prediction and loss                                         |
| Backward Pass (Backprop) | Uses chain rule to compute gradients layer-by-layer                    |
| ReLU                     | Gradient is 0 when input < 0, and 1 when input > 0                     |
| Gradient Descent         | Uses gradients to update parameters in the direction of loss reduction |

---


