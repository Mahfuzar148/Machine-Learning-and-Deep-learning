
---

# 🧠 Deep Learning: Neural Network Training (Forward + Backward Pass with Calculations)

---

## 🔹 1. Understanding Neural Network Layers

![Layer Diagram](attachment\:file-UqQPjYqXHPDyAABhNhjr4g)

### 📌 Layer Types:

| Layer      | Symbol | Description                                         |
| ---------- | ------ | --------------------------------------------------- |
| **Input**  | X      | Accepts raw features like $x_1, x_2, x_3$           |
| **Hidden** | Y      | Performs computation with weights, bias, activation |
| **Output** | Z      | Final predictions like $z_1, z_2$                   |

### Weight and Bias Notation:

* $W^1, b^1$: weights and bias from input to hidden layer
* $W^2, b^2$: weights and bias from hidden to output layer

---

## 🔹 2. Mathematical Flow Across Layers

![Matrix Flow](attachment\:file-9JNpQdD5EGLtm13LehGmUE)

### 🧮 Chaining Layers:

1. First hidden layer:

   $$
   \vec{y} = \sigma(W^1 \cdot \vec{x} + b^1)
   $$

2. Output layer:

   $$
   \vec{z} = \sigma(W^2 \cdot \vec{y} + b^2)
   $$

3. Combined form:

   $$
   \vec{z} = \sigma\left(W^2 \cdot \sigma(W^1 \cdot \vec{x} + b^1) + b^2\right)
   $$

### 🔑 Notes:

* Each layer has its own **weights and bias**
* Activation function $\sigma$ (like sigmoid or ReLU) is applied **after every linear transformation**

---

## 🔹 3. Multi-layer Neural Network View

![Deep ANN](attachment\:file-GPNX4TtscPfFiXat77e9Km)

* This figure shows a deep neural network with:

  * 1 input layer (green)
  * 2 hidden layers (blue)
  * 1 output layer (red)
* Each neuron connects to **all neurons** in the next layer → called a **fully connected layer**

---

## 🔹 4. Forward Pass: Input to Output

![Forward Pass](attachment\:file-JYvbtNX1NM3uuj1E4jpKGr)

### Input Values:

$$
x_1 = 1.2,\quad x_2 = 1.4,\quad x_3 = 1.3
$$

### Output Values:

$$
y_1 = 2.9,\quad y_2 = 0.2
$$

### 📌 How it works:

* Multiply each input by its respective weight
* Add biases
* Apply activation
* Get output

➡️ This process is called the **forward pass**.

---

## 🔹 5. Forward Pass Description

![Forward Summary](attachment\:file-HETdV6sxtGwcgQa2WgHZ5M)

### ✔️ Key Steps:

1. Inputs are fed into the input layer.
2. Each hidden neuron performs:

   $$
   y_i = \sigma(\sum w_{ij} \cdot x_j + b_i)
   $$
3. Hidden layer output goes to the output layer.
4. Final outputs: $y_1 = 2.9$, $y_2 = 0.2$

---

## 🔹 6. Backward Pass: Learning From Error

![Backward Summary](attachment\:file-Qn4Jh1McPxKwnT85ySeYJn)

### Target Values:

$$
\text{Target for } y_1 = 3.2,\quad \text{Target for } y_2 = 0.2
$$

### Error Calculation:

$$
\begin{align*}
\text{Error}_1 &= y_1 - \text{target}_1 = 2.9 - 3.2 = -0.3 \\
\text{Error}_2 &= y_2 - \text{target}_2 = 0.2 - 0.2 = 0
\end{align*}
$$

### 🔁 What happens next:

* These errors are used to compute gradients (via calculus)
* Gradients show **how weights should change**
* Weights are updated using **Gradient Descent** to minimize error

---

## 🔹 7. Loss Function

![Loss Function](attachment\:file-HbdoHPLGEwrZw32iWbkjBB)

### ❓ Why Do We Need a Loss Function?

It compares:

* Predicted output: $\hat{y}$
* Actual/Target output: $y$

It quantifies **how far off** the predictions are.

---

### 📉 Two Common Loss Functions

| Task Type      | Loss Function                | Formula |
| -------------- | ---------------------------- | ------- |
| **Regression** | **Mean Squared Error (MSE)** |         |

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

\| **Classification** | **Cross Entropy (CE)** |

$$
\text{CE} = - \sum_{i} y_i \cdot \log(\hat{y}_i)
$$

---

## ✅ Summary Table

| Step                 | Description                                                             |
| -------------------- | ----------------------------------------------------------------------- |
| **Forward Pass**     | Computes output using inputs, weights, bias, activation.                |
| **Loss Function**    | Measures the error between predicted and true output.                   |
| **Backward Pass**    | Calculates gradient of error w\.r.t. weights using **backpropagation**. |
| **Gradient Descent** | Updates weights to minimize the loss.                                   |

---

## 🚀 Full Flow of Training a Neural Network

1. Input $x_1, x_2, x_3$ is passed (forward pass).
2. Prediction is made $y_1 = 2.9, y_2 = 0.2$.
3. Compare with true labels $3.2, 0.2$ → calculate error.
4. Compute gradients via backward pass.
5. Update weights using gradient descent.
6. Repeat until **loss is minimized** and predictions are accurate.

---


