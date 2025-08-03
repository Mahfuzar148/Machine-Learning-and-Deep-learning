
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

