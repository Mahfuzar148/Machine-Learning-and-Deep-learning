
---

## 🖼️ Reference Image

Here's the reference image that visually explains the neural network structure and process (as seen in your upload):
**![Neural Network Diagram](https://github.com/Mahfuzar148/Machine-Learning-and-Deep-learning/raw/main/Deep%20Learning/Backward%20propagation.png)**

---

## 📌 Goal

We will train a **single-layer neural network** for binary classification using **gradient descent and sigmoid activation**, following these steps:

### ✅ Dataset Row:

```
x₁ = 3000  
x₂ = 6  
x₃ = 1  
y = 0
```

---

## ⚙️ Step-by-Step Full Calculation

### 🔢 Initial Parameters:

```
w₁ = 0.001  
w₂ = 0.01  
w₃ = -0.005  
b = 0.1  
η (Learning Rate) = 0.00001
```

---

### **Step 1: Forward Pass – Linear Combination**

$$
z = x₁w₁ + x₂w₂ + x₃w₃ + b
$$

$$
z = (3000 \cdot 0.001) + (6 \cdot 0.01) + (1 \cdot -0.005) + 0.1
$$

$$
z = 3.0 + 0.06 - 0.005 + 0.1 = \boxed{3.155}
$$

---

### **Step 2: Apply Sigmoid Activation**

$$
\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}} = \frac{1}{1 + e^{-3.155}} \approx \boxed{0.9591}
$$

So, prediction = **0.9591**

---

### **Step 3: Loss Calculation (MSE)**

$$
\text{Loss} = (y - \hat{y})^2 = (0 - 0.9591)^2 = \boxed{0.91988}
$$

---

### **Step 4: Backpropagation – Gradient Calculation**

#### 4A:

$$
\frac{\partial l}{\partial \hat{y}} = -2(y - \hat{y}) = -2(0 - 0.9591) = \boxed{1.9182}
$$

#### 4B:

$$
\frac{\partial \hat{y}}{\partial z} = \hat{y}(1 - \hat{y}) = 0.9591 \cdot (1 - 0.9591) = \boxed{0.03932}
$$

#### 4C: Chain Rule

$$
\frac{\partial l}{\partial z} = \frac{\partial l}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z}
= 1.9182 \cdot 0.03932 = \boxed{0.0755}
$$

#### 4D: Gradients of weights and bias

$$
\frac{\partial l}{\partial w₁} = \frac{\partial l}{\partial z} \cdot \frac{\partial z}{\partial w₁} = 0.0755 \cdot 3000 = \boxed{226.5}
$$

$$
\frac{\partial l}{\partial w₂} = 0.0755 \cdot 6 = \boxed{0.453}
$$

$$
\frac{\partial l}{\partial w₃} = 0.0755 \cdot 1 = \boxed{0.0755}
$$

$$
\frac{\partial l}{\partial b} = 0.0755 \cdot 1 = \boxed{0.0755}
$$

---

### **Step 5: Update Weights & Bias**

$$
w₁ = 0.001 - 0.00001 \cdot 226.5 = \boxed{-0.001265}
$$

$$
w₂ = 0.01 - 0.00001 \cdot 0.453 = \boxed{0.0099955}
$$

$$
w₃ = -0.005 - 0.00001 \cdot 0.0755 = \boxed{-0.005000755}
$$

$$
b = 0.1 - 0.00001 \cdot 0.0755 = \boxed{0.09999924}
$$

---

## 📋 Final Summary Table

| **Parameter** | **Old Value** | **Gradient** | **New Value** |
| ------------- | ------------- | ------------ | ------------- |
| w₁            | 0.001         | 226.5        | -0.001265     |
| w₂            | 0.01          | 0.453        | 0.0099955     |
| w₃            | -0.005        | 0.0755       | -0.005000755  |
| b             | 0.1           | 0.0755       | 0.09999924    |

---

