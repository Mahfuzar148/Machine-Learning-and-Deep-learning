
---

## 🔧 What Is an Activation Function?

An **activation function** is a mathematical function used in neural networks to decide whether a neuron should be activated (i.e., contribute to the output). It applies to the **weighted sum of inputs + bias**, and its primary role is to introduce **non-linearity** into the model.

---

## 🔍 Why Are Activation Functions Important?

| Purpose             | Explanation                                                                                 |
| ------------------- | ------------------------------------------------------------------------------------------- |
| **Non-Linearity**   | Without it, a neural network acts like a simple linear model, unable to learn complex data. |
| **Signal Control**  | Helps determine if a neuron should activate or not.                                         |
| **Learn Patterns**  | Enables deep networks to learn from complicated datasets using layers and non-linearities.  |
| **Backpropagation** | Their derivatives are used during backpropagation to adjust weights and minimize loss.      |

---

## 📚 Types of Activation Functions

### 1. **Sigmoid (Logistic Function)**

**Formula**:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

**Properties**:

* Output range: (0, 1)
* Used for binary classification
* Problem: vanishing gradient for large/small inputs

### 2. **Tanh (Hyperbolic Tangent)**

**Formula**:

$$
\sigma(x) = \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

**Properties**:

* Output range: (−1, 1)
* Zero-centered (better for optimization)
* Still can suffer from vanishing gradients

### 3. **ReLU (Rectified Linear Unit)**

**Formula**:

$$
\sigma(x) = \max(0, x)
$$

**Properties**:

* Most widely used in deep learning
* Output: 0 for x < 0, and x for x ≥ 0
* Solves vanishing gradient to some extent
* Problem: “dying ReLU” (neurons stuck at 0)

### 4. **Softmax**

**Formula**:

$$
\sigma(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}
$$

**Properties**:

* Used in output layer for multi-class classification
* Outputs a **probability distribution** (sums to 1)

---

## ✅ Summary Table

| Function | Output Range  | Common Use Case            | Notes                                    |
| -------- | ------------- | -------------------------- | ---------------------------------------- |
| Sigmoid  | (0, 1)        | Binary classification      | Can squash gradients                     |
| Tanh     | (−1, 1)       | Hidden layers (earlier NN) | Zero-centered, better than sigmoid       |
| ReLU     | \[0, ∞)       | Most hidden layers         | Fast and effective; watch for dead units |
| Softmax  | (0, 1), sum=1 | Output layer (multi-class) | Gives probabilities for each class       |

---


---

## 1️⃣ ReLU (Rectified Linear Unit)

**Formula:**

$$
f(x) = \max(0, x)
$$

### 📦 Used in:

* Hidden layers of CNNs and feedforward networks
* Very common in deep networks due to simplicity

### ✅ Advantages:

* Fast & easy to compute
* Avoids vanishing gradient (for x > 0)
* Sparse activation → improves efficiency

### ⚠️ Disadvantages:

* “Dying ReLU” problem: if input is ≤ 0, gradient becomes zero and neuron may stop learning

---

## 2️⃣ Leaky ReLU

**Formula:**

$$
f(x) = 
\begin{cases}
x & x \ge 0 \\
\epsilon x & x < 0
\end{cases}
\quad (\epsilon \text{ is a small value like 0.01})
$$

### 📦 Used in:

* Hidden layers to fix dying ReLU

### ✅ Advantages:

* Allows small gradient for negative inputs
* Fixes dead neuron issue

### ⚠️ Disadvantages:

* Still not entirely zero-centered
* Value of ε must be chosen carefully

---

## 3️⃣ Parametric ReLU (PReLU)

**Like Leaky ReLU, but ε is learnable during training**

### 📦 Used in:

* Deeper architectures where learning flexibility helps (e.g., ResNet variants)

### ✅ Advantages:

* More flexible
* Learns optimal leakiness

### ⚠️ Disadvantages:

* Risk of overfitting with small datasets

---

## 4️⃣ ELU (Exponential Linear Unit)

**Formula:**

$$
f(x) = 
\begin{cases}
x & x \ge 0 \\
\alpha (e^x - 1) & x < 0
\end{cases}
$$

### 📦 Used in:

* Deep networks (especially where faster convergence is desired)

### ✅ Advantages:

* Smooth gradient for all x
* Avoids dead neuron problem
* Negative values help mean activations closer to zero (faster training)

### ⚠️ Disadvantages:

* More computationally expensive than ReLU

---

## 5️⃣ Swish

**Formula:**

$$
f(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
$$

### 📦 Used in:

* Advanced models (e.g., EfficientNet)

### ✅ Advantages:

* Smooth function, good for gradient flow
* Outperforms ReLU in deeper models

### ⚠️ Disadvantages:

* Slower to compute
* Newer → less universally supported

---

## 6️⃣ Sigmoid (Logistic Function)

**Formula:**

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

### 📦 Used in:

* **Output layer** for **binary classification**

### ✅ Advantages:

* Output in range (0, 1), useful for probabilities

### ⚠️ Disadvantages:

* Vanishing gradient
* Not zero-centered
* Expensive computation

---

## 7️⃣ Tanh (Hyperbolic Tangent)

**Formula:**

$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

### 📦 Used in:

* Hidden layers (especially in RNNs, GRU, LSTM)

### ✅ Advantages:

* Zero-centered
* Better than sigmoid for hidden layers

### ⚠️ Disadvantages:

* Still has vanishing gradient
* More expensive than ReLU

---

## 8️⃣ Softmax

**Formula (for class $i$):**

$$
\sigma(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

### 📦 Used in:

* **Output layer** for **multi-class classification**

### ✅ Advantages:

* Outputs probabilities for each class
* Normalized (sum = 1)

### ⚠️ Disadvantages:

* Not good for regression
* Sensitive to large input values (can saturate)

---

### 🧠 Quick Review Table

| Activation | Use Layer | Use Case                   | Advantage         | Disadvantage    |
| ---------- | --------- | -------------------------- | ----------------- | --------------- |
| ReLU       | Hidden    | CNNs, FFNs                 | Fast, Sparse      | Dying ReLU      |
| Leaky ReLU | Hidden    | CNNs, RNNs                 | Fixes dying ReLU  | ε manual        |
| PReLU      | Hidden    | Deep learning              | Learnable ε       | Overfit         |
| ELU        | Hidden    | Deep nets                  | Smooth & fast     | Costly          |
| Swish      | Hidden    | Deep nets (new)            | Smooth & flexible | Expensive       |
| Sigmoid    | Output    | Binary classification      | Probabilities     | Vanishing grad  |
| Tanh       | Hidden    | RNNs                       | Centered          | Still vanishing |
| Softmax    | Output    | Multi-class classification | Class probs       | Can saturate    |

---


---

### 1. **ReLU (Rectified Linear Unit)**

* **Formula**: `f(x) = max(0, x)`
* **Use in Layer**: Hidden layers (most commonly)
* **Use Case**: General deep learning models — fast and simple.
* **Advantages**:

  * Fast to compute
  * Avoids vanishing gradient (mostly)
* **Disadvantages**:

  * "Dying ReLU" problem: neurons can become inactive
* **Code**:

  ```python
  def relu(x):
      return np.maximum(0, x)
  ```

---

### 2. **Leaky ReLU**

* **Formula**: `f(x) = x if x > 0 else αx`, commonly α = 0.01
* **Use in Layer**: Hidden layers
* **Use Case**: When ReLU causes dead neurons
* **Advantages**:

  * Fixes dying ReLU by allowing small gradient for x < 0
* **Disadvantages**:

  * Slightly more complex, α must be tuned
* **Code**:

  ```python
  def leaky_relu(x, alpha=0.01):
      return np.where(x > 0, x, x * alpha)
  ```

---

### 3. **ELU (Exponential Linear Unit)**

* **Formula**:

  ```
  f(x) = x if x ≥ 0 else α * (exp(x) - 1)
  ```
* **Use in Layer**: Hidden layers
* **Use Case**: Deep networks, when smooth negative values help
* **Advantages**:

  * Non-zero gradient for x < 0
  * Helps with vanishing gradients
* **Disadvantages**:

  * Computationally slower (uses `exp`)
* **Code**:

  ```python
  def elu(x, alpha=1.0):
      return np.where(x > 0, x, alpha * (np.exp(x) - 1))
  ```

---

### 4. **Sigmoid**

* **Formula**: `σ(x) = 1 / (1 + exp(-x))`
* **Use in Layer**: Output layer for **binary classification**
* **Advantages**:

  * Maps input to (0, 1), like probability
* **Disadvantages**:

  * Vanishing gradient
  * Not zero-centered
  * Slower to compute
* **Code**:

  ```python
  def sigmoid(x):
      return 1 / (1 + np.exp(-x))
  ```

---

### 5. **Softmax**

* **Formula**:

  $$
  \text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
  $$
* **Use in Layer**: Output layer for **multiclass classification**
* **Advantages**:

  * Outputs probabilities for multiple classes
* **Disadvantages**:

  * Sensitive to outliers (exponentiation)
* **Code**:

  ```python
  def softmax(x):
      e_x = np.exp(x - np.max(x))  # for numerical stability
      return e_x / e_x.sum(axis=0)
  ```

---

### 6. **Swish**

* **Formula**: `f(x) = x * sigmoid(x)`
* **Use in Layer**: Hidden layers
* **Use Case**: Advanced models (e.g., EfficientNet, deeper CNNs)
* **Advantages**:

  * Can outperform ReLU
  * Smooth and non-monotonic
* **Disadvantages**:

  * Slower due to sigmoid
* **Code**:

  ```python
  def swish(x):
      return x * sigmoid(x)
  ```

---

### 7. **Tanh**

* **Formula**:

  $$
  \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
  $$
* **Use in Layer**: Hidden layers (especially in RNNs, LSTM, GRU)
* **Advantages**:

  * Zero-centered
  * Maps to (-1, 1)
* **Disadvantages**:

  * Still has vanishing gradient
* **Code**:

  ```python
  def tanh(x):
      return np.tanh(x)
  ```

---

### 8. **Linear**

* **Formula**: `f(x) = x`
* **Use in Layer**: Output layer for regression problems
* **Advantages**:

  * Best for regression (keeps full range)
* **Disadvantages**:

  * No non-linearity — not useful in hidden layers
* **Code**:

  ```python
  def linear(x):
      return x
  ```

---



