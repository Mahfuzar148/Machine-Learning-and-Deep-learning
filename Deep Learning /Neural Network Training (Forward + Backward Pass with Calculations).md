
---


# 🧠 Deep Learning: Neural Network Training  
**(Forward Pass + Backward Pass with Calculations)**

---

## 📚 Table of Contents

1. [🔹 Introduction](#1-introduction)  
2. [🔹 Neural Network Architecture](#2-neural-network-architecture)  
3. [🔹 Forward Propagation](#3-forward-propagation)  
4. [🔹 Loss Function](#4-loss-function)  
5. [🔹 Backward Propagation](#5-backward-propagation)  
6. [🔹 Parameter Update (Gradient Descent)](#6-gradient-descent)  
7. [🔹 Numerical Example](#7-numerical-example)  
8. [🔹 Summary](#8-summary)

---

## 🔹 1. Introduction

A **neural network** learns by minimizing error between predictions and actual outputs using:

- Forward propagation  
- Loss calculation  
- Backpropagation

---

## 🔹 2. Neural Network Architecture

We'll use a **simple feedforward neural network**:

- Input Layer → 2 features  
- 1 Hidden Layer → 2 neurons  
- Output Layer → 1 neuron

### Structure:

```

x1     x2
\|      |
\|     \[w11, w12]      (Hidden Layer)
\    /
\[y1, y2] ---> Activation ---> Output Layer

\[w2]   →   z

```

---

## 🔹 3. Forward Propagation

### Input:

Let:

```

x = \[x1, x2]^T
W1 = \[\[w11, w12],
\[w21, w22]]
b1 = \[b1, b2]^T

```

### Hidden Layer:

```

z1 = W1 \* x + b1
a1 = σ(z1)        (e.g., sigmoid)

```

### Output Layer:

```

W2 = \[w31, w32], b2 = b3
z2 = W2 \* a1 + b2
ŷ  = σ(z2)         (final output)

```

---

## 🔹 4. Loss Function

For binary classification, use **binary cross-entropy**:

```

L(ŷ, y) = -\[y \* log(ŷ) + (1 - y) \* log(1 - ŷ)]

```

---

## 🔹 5. Backward Propagation

Using the **chain rule** to compute gradients.

### Output Layer:

```

dL/dŷ = (ŷ - y) / \[ŷ(1 - ŷ)]
dŷ/dz2 = ŷ(1 - ŷ)
→ dL/dz2 = ŷ - y

dL/dW2 = (ŷ - y) \* a1
dL/db2 = ŷ - y

```

### Hidden Layer:

Let δ2 = ŷ - y

```

δ1 = (δ2 \* W2) \* a1 \* (1 - a1)
dL/dW1 = δ1 \* x^T
dL/db1 = δ1

```

---

## 🔹 6. Gradient Descent

Using learning rate η:

```

W = W - η \* dL/dW
b = b - η \* dL/db

```

---

## 🔹 7. Numerical Example

### Given:

```

x = \[1, 0]^T
W1 = \[\[0.1, 0.2], \[0.3, 0.4]]
b1 = \[0.1, 0.1]^T
W2 = \[0.2, 0.3]
b2 = 0.1
y = 1

```

### Forward Pass:

```

z1 = W1 \* x + b1 = \[0.2, 0.4]
a1 = σ(z1) ≈ \[0.55, 0.60]
z2 = W2 \* a1 + b2 = 0.39
ŷ = σ(0.39) ≈ 0.596

```

### Loss:

```

L = -\[log(0.596)] ≈ 0.517

```

### Backward Pass:

```

δ2 = ŷ - y = -0.404
dW2 = δ2 \* a1 = \[-0.222, -0.243]

δ1 = δ2 \* W2 \* a1 \* (1 - a1) ≈ \[-0.020, -0.029]
dW1 = δ1 \* x^T = \[\[-0.020, 0], \[-0.029, 0]]

```

---

## 🔹 8. Summary

| Step              | Description                          |
|-------------------|--------------------------------------|
| Forward Pass      | Calculate output from input          |
| Loss              | Compare prediction with target       |
| Backward Pass     | Compute gradients via chain rule     |
| Update Parameters | Adjust weights using gradient descent|

---
```

---


