---

# 🧠 Deep Learning: Neural Network Training (Forward + Backward Pass with Calculations)

---

## 🔹 1. Understanding Neural Network Layers

### 📌 Layer Types:

| Layer      | Symbol      | Description                                                   |
| ---------- | ----------- | ------------------------------------------------------------- |
| **Input**  | \$\vec{x}\$ | Accepts raw features like \$\vec{x} = \[x\_1,\ x\_2,\ x\_3]\$ |
| **Hidden** | \$\vec{y}\$ | Performs computation with weights, bias, activation           |
| **Output** | \$\vec{z}\$ | Final predictions like \$\vec{z} = \[z\_1,\ z\_2]\$           |

### Weight and Bias Notation:

* \$W^{\[1]},\ b^{\[1]}\$: weights and bias from input to hidden layer
* \$W^{\[2]},\ b^{\[2]}\$: weights and bias from hidden to output layer

---

## 🔹 2. Mathematical Flow Across Layers

### 🧮 Chaining Layers:

1. First hidden layer:

   $$
   \vec{y} = \sigma(W^{[1]} \cdot \vec{x} + b^{[1]})
   $$

2. Output layer:

   $$
   \vec{z} = \sigma(W^{[2]} \cdot \vec{y} + b^{[2]})
   $$

3. Combined form:

   $$
   \vec{z} = \sigma\left(W^{[2]} \cdot \sigma(W^{[1]} \cdot \vec{x} + b^{[1]}) + b^{[2]}\right)
   $$

### 🔑 Notes:

* Each layer has its own **weights and bias**
* Activation function \$\sigma\$ (like sigmoid or ReLU) is applied **after every linear transformation**

---

## 🔹 3. Multi-layer Neural Network View

* This figure shows a deep neural network with:

  * 1 input layer (green)
  * 2 hidden layers (blue)
  * 1 output layer (red)
* Each neuron connects to **all neurons** in the next layer → called a **fully connected layer**

---

## 🔹 4. Forward Pass: Input to Output

### Input Vector:

$$
\vec{x} = [1.2,\ 1.4,\ 1.3]
$$

### Output Vector:

$$
\vec{y} = [2.9,\ 0.2]
$$

### 📌 How it works:

* Multiply each input by its respective weight
* Add biases
* Apply activation
* Get output

➡️ This process is called the **forward pass**.

---

## 🔹 5. Forward Pass Description

### ✔️ Key Steps:

1. Inputs are fed into the input layer.

2. Each hidden neuron performs:

   $$
   y_i = \sigma\left(\sum_j w_{ij} \cdot x_j + b_i\right)
   $$

3. Hidden layer output goes to the output layer.

4. Final outputs: \$y\_1 = 2.9\$, \$y\_2 = 0.2\$

---

## 🔹 6. Backward Pass: Learning From Error

### Target Vector:

$$
\vec{t} = [3.2,\ 0.2]
$$

### Error Calculation:

$$
\begin{align*}
\text{Error}_1 &= y_1 - t_1 = 2.9 - 3.2 = -0.3 \\
\text{Error}_2 &= y_2 - t_2 = 0.2 - 0.2 = 0
\end{align*}
$$

### 🔁 What happens next:

* These errors are used to compute gradients (via calculus)
* Gradients show **how weights should change**
* Weights are updated using **Gradient Descent** to minimize error

---

## 🔹 7. Loss Function

### ❓ Why Do We Need a Loss Function?

It compares:

* Predicted output: \$\hat{y}\$
* Actual/Target output: \$y\$

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

1. Input \$\vec{x} = \[x\_1,\ x\_2,\ x\_3]\$ is passed (forward pass).
2. Prediction is made \$\vec{y} = \[2.9,\ 0.2]\$.
3. Compare with true labels \$\vec{t} = \[3.2,\ 0.2]\$ → calculate error.
4. Compute gradients via backward pass.
5. Update weights using gradient descent.
6. Repeat until **loss is minimized** and predictions are accurate.

