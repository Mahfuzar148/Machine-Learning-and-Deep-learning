**Manual Gradient Descent: 5 Function Types with Full Calculations**

---

### ✅ Example 1: Two-variable Linear Function

**Function:** $f(w, b) = (wx + b - y)^2$

**Data:**

* $x = 2$, $y = 5$
* Initial: $w = 1$, $b = 0$
* Learning rate: $\eta = 0.1$

#### Step-by-step Iterations:

**Step 1:**

* $\hat{y} = 1 \cdot 2 + 0 = 2$
* Loss: $(2 - 5)^2 = 9$
* Gradients:

  * $\frac{\partial L}{\partial w} = 2(-3) \cdot 2 = -12$
  * $\frac{\partial L}{\partial b} = 2(-3) = -6$
* Updates:

  * $w = 1 + 1.2 = 2.2$
  * $b = 0 + 0.6 = 0.6$

**Step 2:**

* $\hat{y} = 2.2 \cdot 2 + 0.6 = 4.4 + 0.6 = 5$
* Loss: $(5 - 5)^2 = 0$
* Training converged.

---

### ✅ Example 2: Three-variable Linear Function

**Function:** $f(w_1, w_2, b) = (w_1x_1 + w_2x_2 + b - y)^2$

**Data:**

* $x_1 = 1, x_2 = 2, y = 10$
* Initial: $w_1 = 1, w_2 = 1, b = 0$
* $\eta = 0.1$

**Step 1:**

* $\hat{y} = 1 + 2 + 0 = 3$
* Loss: $(3 - 10)^2 = 49$
* Gradients:

  * $\partial w_1 = 2(-7) \cdot 1 = -14$
  * $\partial w_2 = 2(-7) \cdot 2 = -28$
  * $\partial b = 2(-7) = -14$
* Updates:

  * $w_1 = 2.4, w_2 = 3.8, b = 1.4$

**Step 2:**

* $\hat{y} = 2.4 \cdot 1 + 3.8 \cdot 2 + 1.4 = 2.4 + 7.6 + 1.4 = 11.4$
* Loss: $(11.4 - 10)^2 = 1.96$
* Gradients:

  * $\partial w_1 = 2(1.4) \cdot 1 = 2.8$
  * $\partial w_2 = 2(1.4) \cdot 2 = 5.6$
  * $\partial b = 2(1.4) = 2.8$
* Updates:

  * $w_1 = 2.12, w_2 = 3.24, b = 1.12$

**Step 3:**

* Continue until loss \~ 0

---

### ✅ Example 3: Polynomial Function

**Function:** $f(w) = (w^2 - y)^2$

**Data:**

* $y = 9, w = 1, \eta = 0.1$

**Step 1:**

* $\hat{y} = 1^2 = 1$
* Loss = $(1 - 9)^2 = 64$
* Gradient:

  * $\frac{dL}{dw} = 4w(w^2 - y) = 4 \cdot 1 \cdot (1 - 9) = -32$
* Update:

  * $w = 1 + 3.2 = 4.2$

**Step 2:**

* $\hat{y} = 4.2^2 = 17.64$
* Loss = $(17.64 - 9)^2 \approx 74.5$
* Gradient:

  * $4 \cdot 4.2 \cdot (17.64 - 9) \approx 145$
* Update:

  * $w = 4.2 - 14.5 = -10.3$

**Step 3+:** Continue until convergence.

---

### ✅ Example 4: Trigonometric Function

**Function:** $f(w) = (\sin(w) - y)^2$

**Data:**

* $y = 0.5, w = 0, \eta = 0.1$

**Step 1:**

* $\hat{y} = \sin(0) = 0$, Loss = $0.25$
* Gradient: $2(\sin(w) - y) \cdot \cos(w) = -1$
* Update: $w = 0 + 0.1 = 0.1$

**Step 2:**

* $\hat{y} = \sin(0.1) \approx 0.0998$, Error = $-0.4002$
* Gradient: $2 \cdot -0.4002 \cdot \cos(0.1) \approx -0.796$
* Update: $w = 0.1796$

Continue till loss \~ 0.

---

### ✅ Example 5: Logarithmic Function

**Function:** $f(w) = (\log(w) - y)^2$

**Data:**

* $y = 1, w = 2, \eta = 0.1$

**Step 1:**

* $\log(2) \approx 0.693$, Loss = $(0.693 - 1)^2 = 0.094$
* Gradient: $2(\log(w) - y)/w = -0.1535$
* Update: $w = 2.015$

**Step 2:**

* $\log(2.015) \approx 0.700$, Loss = $0.09$, Gradient \~ -0.149
* Continue updating until loss \~ 0

---

**Conclusion:** Each function required:

* Defining prediction ($\hat{y}$)
* Loss computation
* Gradient derivation using chain rule
* Parameter updates using gradient descent

Repeat until convergence.
