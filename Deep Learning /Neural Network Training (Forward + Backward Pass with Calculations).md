Absolutely! Here's a **complete, in-depth documentation** on:

---

# 🧠 Deep Learning: Neural Network Training

**(Forward Pass + Backward Pass with Calculations)**
Includes math, examples, explanations, and illustrations.

---

## 📚 Table of Contents

1. [🔹 Introduction](#introduction)
2. [🔹 Neural Network Architecture](#architecture)
3. [🔹 Forward Propagation](#forward)
4. [🔹 Loss Function](#loss)
5. [🔹 Backward Propagation](#backward)
6. [🔹 Parameter Update (Gradient Descent)](#gradient)
7. [🔹 Numerical Example (with values)](#example)
8. [🔹 Summary](#summary)

---

<a name="introduction"></a>

## 🔹 1. Introduction

A **neural network** learns by minimizing error between predictions and actual outputs using **forward propagation**, **loss calculation**, and **backpropagation**.

---

<a name="architecture"></a>

## 🔹 2. Neural Network Architecture

We'll use a **simple feedforward neural network**:

* **Input Layer (x)** → 2 features
* **1 Hidden Layer (y)** → 2 neurons
* **Output Layer (z)** → 1 neuron

### Diagram:

```
x1     x2
 |      |
 |     [w11, w12]      (Hidden Layer)
  \    /
   [y1, y2] ---> Activation ---> Output Layer
        \
        [w2]   →   z
```

---

<a name="forward"></a>

## 🔹 3. Forward Propagation

### ➤ Input:

Let

\[
\vec{x} = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}, \quad
W^{[1]} = \begin{bmatrix}
w_{11} & w_{12} \\
w_{21} & w_{22}
\end{bmatrix}, \quad
\vec{b}^{[1]} = \begin{bmatrix} b_1 \\ b_2 \end{bmatrix}
\]


### ➤ Hidden Layer (linear + activation):

$$
\vec{z^{[1]}} = \vec{W^{[1]}} \cdot \vec{x} + \vec{b^{[1]}} \quad \text{(Linear)}
$$

$$
\vec{a^{[1]}} = \sigma(\vec{z^{[1]}}) \quad \text{(Activation, e.g., Sigmoid)}
$$

### ➤ Output Layer:

Let output weights:

$$
\vec{W^{[2]}} = \begin{bmatrix} w_{31} & w_{32} \end{bmatrix}, \quad b^{[2]} = b_3
$$

Then,

$$
z^{[2]} = \vec{W^{[2]}} \cdot \vec{a^{[1]}} + b^{[2]}
$$

$$
\hat{y} = \sigma(z^{[2]}) \quad \text{(Prediction)}
$$

---

<a name="loss"></a>

## 🔹 4. Loss Function

We use the **Binary Cross-Entropy** loss for binary classification:

$$
\mathcal{L}(\hat{y}, y) = - \left[ y \cdot \log(\hat{y}) + (1 - y) \cdot \log(1 - \hat{y}) \right]
$$

---

<a name="backward"></a>

## 🔹 5. Backward Propagation (Step-by-Step)

We apply the **chain rule** of calculus to compute gradients of loss w\.r.t each weight.

### ➤ Output Layer:

1. $\frac{\partial \mathcal{L}}{\partial \hat{y}} = \frac{\hat{y} - y}{\hat{y}(1 - \hat{y})}$
2. $\frac{\partial \hat{y}}{\partial z^{[2]}} = \hat{y}(1 - \hat{y})$
   ⟹

$$
\frac{\partial \mathcal{L}}{\partial z^{[2]}} = \hat{y} - y
$$

3. $\frac{\partial z^{[2]}}{\partial W^{[2]}} = a^{[1]}$
   ⟹

$$
\frac{\partial \mathcal{L}}{\partial W^{[2]}} = ( \hat{y} - y ) \cdot a^{[1]}
$$

---

### ➤ Hidden Layer:

Let $\delta^{[2]} = \hat{y} - y$

1. $\frac{\partial \mathcal{L}}{\partial a^{[1]}} = \delta^{[2]} \cdot W^{[2]}$
2. $\frac{\partial a^{[1]}}{\partial z^{[1]}} = a^{[1]} \cdot (1 - a^{[1]})$

⟹

$$
\delta^{[1]} = \left( \delta^{[2]} \cdot W^{[2]} \right) \cdot a^{[1]} \cdot (1 - a^{[1]})
$$

3. $\frac{\partial \mathcal{L}}{\partial W^{[1]}} = \delta^{[1]} \cdot x^\top$

---

<a name="gradient"></a>

## 🔹 6. Gradient Descent: Parameter Update

Let learning rate $\eta$:

$$
W = W - \eta \cdot \frac{\partial \mathcal{L}}{\partial W}
$$

Apply to all:

* $W^{[2]}$, $b^{[2]}$
* $W^{[1]}$, $b^{[1]}$

---

<a name="example"></a>

## 🔹 7. Numerical Example

Let’s take one input sample:

### ➤ Given:

$$
x = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad
W^{[1]} = \begin{bmatrix} 0.1 & 0.2 \\ 0.3 & 0.4 \end{bmatrix}, \quad
b^{[1]} = \begin{bmatrix} 0.1 \\ 0.1 \end{bmatrix}
$$

$$
W^{[2]} = \begin{bmatrix} 0.2 & 0.3 \end{bmatrix}, \quad
b^{[2]} = 0.1, \quad y = 1
$$

---

### ➤ Forward Pass:

1. Hidden linear:

$$
z^{[1]} = W^{[1]} \cdot x + b^{[1]} = \begin{bmatrix} 0.1 \\ 0.3 \end{bmatrix} + \begin{bmatrix} 0.1 \\ 0.1 \end{bmatrix} = \begin{bmatrix} 0.2 \\ 0.4 \end{bmatrix}
$$

2. Activation (Sigmoid):

$$
a^{[1]} = \sigma(z^{[1]}) = \begin{bmatrix} \sigma(0.2) \\ \sigma(0.4) \end{bmatrix} ≈ \begin{bmatrix} 0.55 \\ 0.60 \end{bmatrix}
$$

3. Output layer:

$$
z^{[2]} = W^{[2]} \cdot a^{[1]} + b^{[2]} = 0.2×0.55 + 0.3×0.60 + 0.1 = 0.11 + 0.18 + 0.1 = 0.39
$$

4. Final output:

$$
\hat{y} = \sigma(0.39) ≈ 0.596
$$

---

### ➤ Loss:

$$
\mathcal{L} = -\left[1 \cdot \log(0.596) + 0 \cdot \log(1 - 0.596)\right] ≈ 0.517
$$

---

### ➤ Backward Pass:

1. Output error:

$$
\delta^{[2]} = \hat{y} - y = 0.596 - 1 = -0.404
$$

2. Gradients for $W^{[2]}$:

$$
\frac{\partial \mathcal{L}}{\partial W^{[2]}} = \delta^{[2]} \cdot a^{[1]} = -0.404 × [0.55, 0.60] = [-0.222, -0.243]
$$

3. Backprop to hidden layer:

$$
\delta^{[1]} = \delta^{[2]} \cdot W^{[2]} \cdot a^{[1]} \cdot (1 - a^{[1]})
$$

$$
= -0.404 \cdot [0.2, 0.3] \cdot [0.55(1-0.55), 0.60(1-0.60)] ≈ [-0.020, -0.029]
$$

4. Gradients for $W^{[1]}$:

$$
\frac{\partial \mathcal{L}}{\partial W^{[1]}} = \delta^{[1]} \cdot x^\top = 
\begin{bmatrix} -0.020 \\ -0.029 \end{bmatrix} \cdot \begin{bmatrix} 1 & 0 \end{bmatrix} =
\begin{bmatrix} -0.020 & 0 \\ -0.029 & 0 \end{bmatrix}
$$

---

<a name="summary"></a>

## 🔹 8. Summary

| Step              | Description                             |
| ----------------- | --------------------------------------- |
| **Forward Pass**  | Compute output $\hat{y}$ from input $x$ |
| **Loss**          | Compare $\hat{y}$ to ground truth $y$   |
| **Backward Pass** | Calculate gradients using chain rule    |
| **Update**        | Adjust weights via gradient descent     |

---

Would you like me to export this as a **PDF** or **Word** document for download?
