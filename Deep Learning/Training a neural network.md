
---

## 🧾 Dataset Sample

We’ll use the first row:
```
Size = 3000, Bedroom = 6, AC = 1, Buy = 0
Features → x₁ = 3000, x₂ = 6, x₃ = 1  
Label → y = 0
```

---

## ⚙️ Initial Parameters

```
w₁ = 0.001  
w₂ = 0.01  
w₃ = -0.005  
b = 0.1  
η (Learning Rate) = 0.00001
```

---

## 🔢 Step-by-Step Calculation

### **Step 1: Forward Pass – Net Input Calculation**

We compute:

\[
z = x_1w_1 + x_2w_2 + x_3w_3 + b
= (3000 \cdot 0.001) + (6 \cdot 0.01) + (1 \cdot -0.005) + 0.1
= 3 + 0.06 - 0.005 + 0.1 = \boxed{3.155}
\]

---

### **Step 2: Apply Sigmoid Activation**

\[
\hat{y} = \frac{1}{1 + e^{-z}} = \frac{1}{1 + e^{-3.155}} ≈ \boxed{0.959}
\]

এই \( \hat{y} \) হলো আমাদের predicted value.

---

### **Step 3: Compute Loss**

We use MSE (Mean Squared Error):

\[
l = (y - \hat{y})^2 = (0 - 0.959)^2 = \boxed{0.919681}
\]

---

### **Step 4: Backpropagation – Gradient Calculation**

We want \( \frac{\partial l}{\partial w_1}, \frac{\partial l}{\partial w_2}, \frac{\partial l}{\partial w_3}, \frac{\partial l}{\partial b} \)

#### Breakdown:

**Step A:**  
\[
\frac{\partial l}{\partial \hat{y}} = -2(y - \hat{y}) = -2(0 - 0.959) = \boxed{1.918}
\]

**Step B:**  
\[
\frac{\partial \hat{y}}{\partial z} = \hat{y}(1 - \hat{y}) = 0.959 \cdot (1 - 0.959) = 0.959 \cdot 0.041 = \boxed{0.039319}
\]

**Step C:** Compute gradients

\[
\frac{\partial z}{\partial w_1} = x_1 = 3000  
\Rightarrow \frac{\partial l}{\partial w_1} = 1.918 \cdot 0.039319 \cdot 3000 = \boxed{226.2}
\]

\[
\frac{\partial z}{\partial w_2} = x_2 = 6  
\Rightarrow \frac{\partial l}{\partial w_2} = 1.918 \cdot 0.039319 \cdot 6 = \boxed{0.453}
\]

\[
\frac{\partial z}{\partial w_3} = x_3 = 1  
\Rightarrow \frac{\partial l}{\partial w_3} = 1.918 \cdot 0.039319 \cdot 1 = \boxed{0.0755}
\]

\[
\frac{\partial z}{\partial b} = 1  
\Rightarrow \frac{\partial l}{\partial b} = 1.918 \cdot 0.039319 \cdot 1 = \boxed{0.0755}
\]

---

### **Step 5: Weight & Bias Update**

\[
w_1 = w_1 - \eta \cdot \frac{\partial l}{\partial w_1} = 0.001 - 0.00001 \cdot 226.2 = \boxed{-0.001262}
\]

\[
w_2 = 0.01 - 0.00001 \cdot 0.453 = \boxed{0.0099955}
\]

\[
w_3 = -0.005 - 0.00001 \cdot 0.0755 = \boxed{-0.005000755}
\]

\[
b = 0.1 - 0.00001 \cdot 0.0755 = \boxed{0.09999924}
\]

---

## ✅ Final Updated Parameters

| Parameter | Old Value | Gradient | New Value     |
|-----------|-----------|----------|---------------|
| w₁        | 0.001     | 226.2    | -0.001262     |
| w₂        | 0.01      | 0.453    | 0.0099955     |
| w₃        | -0.005    | 0.0755   | -0.005000755  |
| b         | 0.1       | 0.0755   | 0.09999924    |

---

