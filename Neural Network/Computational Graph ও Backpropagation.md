
---

## 📘 **বিস্তারিত ডকুমেন্টেশন: Computational Graph ও Backpropagation**

---

### 🔍 **পরিচিতি**

**Computational Graph** হলো এক ধরনের গাণিতিক ডেটা-স্ট্রাকচার যা একটি জটিল ফাংশনকে ছোট ছোট গাণিতিক অপারেশনে ভাগ করে একটি গ্রাফের মতো করে উপস্থাপন করে। নিউরাল নেটওয়ার্কে এটি ব্যবহৃত হয় মূলত:

* **Forward Pass**: আউটপুট কীভাবে আসছে, তা নির্ণয় করতে
* **Backward Pass**: Error বা Loss কিভাবে propagate হচ্ছে, তা বোঝাতে

প্রতিটি **Node** বোঝায় একটি অপারেশন (যেমন যোগ, গুণ) অথবা ভেরিয়েবল (যেমন x, y)। আর **Edge** বোঝায় data flow বা অপারেশনের input/output।

---

### 🧮 **উদাহরণ ১: $z = x \times y$**

**Computation Graph:**

```
     x = 3
       \
        * ---> z = x × y = 3 × 4 = 12
       /
     y = 4
```

#### ➤ Forward Pass:

* ইনপুট: x = 3, y = 4
* অপারেশন: z = x × y
* আউটপুট: z = 12

#### ➤ Backward Pass:

আমরা জানতে চাই, **Loss L** এর respect-এ $\frac{\partial L}{\partial x}$ এবং $\frac{\partial L}{\partial y}$ কত হবে।

এখানে, যদি আমরা জানি $\frac{\partial L}{\partial z} = 1$, তাহলে Chain Rule অনুযায়ী:

* $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial x} = 1 \cdot y = 4$
* $\frac{\partial L}{\partial y} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial y} = 1 \cdot x = 3$

---

### 🔁 **Backpropagation এর ধাপসমূহ**

Backpropagation হল এমন একটি অ্যালগরিদম, যা Gradient Descent ব্যবহার করে নিউরাল নেটওয়ার্কে ওজন (weights) হালনাগাদ করে।

#### ✅ ধাপ ১: Forward Pass

ইনপুট থেকে আউটপুট পর্যন্ত ফিড-ফরওয়ার্ড হয়, প্রত্যেকটি লেয়ারে অ্যাক্টিভেশন ও ওজন প্রয়োগ করে।

#### ✅ ধাপ ২: Loss হিসাব

Prediction এর সাথে Actual Label এর পার্থক্য মাপা হয়:

$$
L = \text{Loss}(y_{\text{true}}, y_{\text{pred}})
$$

#### ✅ ধাপ ৩: Backward Pass (Gradient হিসাব)

Chain Rule ব্যবহার করে Loss এর respect-এ প্রতিটি Weight এর Partial Derivative বের করা হয়:

$$
\frac{\partial L}{\partial w_i}
$$

#### ✅ ধাপ ৪: Weight Update (Gradient Descent):

প্রতিটি weight নতুনভাবে নির্ধারণ করা হয়:

$$
w_{\text{new}} = w_{\text{old}} - \eta \cdot \frac{\partial L}{\partial w}
$$

এখানে $\eta$ হলো Learning Rate।

---

### 🔄 **Chain Rule এবং Gradient Flow**

ধরি, একটি নেটওয়ার্ক আছে:

```text
x --> [Layer1] --> [Layer2] --> ... --> Loss
        ↑           ↑
       dw1         dw2        ← Backward direction
```

Chain Rule অনুসারে:

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial y} \cdot \frac{\partial y}{\partial x}
$$

এভাবে Gradient ধাপে ধাপে propagate হয়।

---

### 📊 **একটি বড় Computational Graph এর উদাহরণ**

ধরি:

$$
a = x \times y \\
b = a + z \\
c = \text{ReLU}(b)
$$

**Forward Pass:**

1. $a = x \times y$
2. $b = a + z$
3. $c = \max(0, b)$

**Backward Pass:**

1. $\frac{\partial c}{\partial b} = 1$ if $b > 0$, otherwise 0
2. $\frac{\partial b}{\partial a} = 1$, $\frac{\partial b}{\partial z} = 1$
3. $\frac{\partial a}{\partial x} = y$, $\frac{\partial a}{\partial y} = x$

**Final Gradient (Chain Rule):**

$$
\frac{\partial c}{\partial x} = \frac{\partial c}{\partial b} \cdot \frac{\partial b}{\partial a} \cdot \frac{\partial a}{\partial x} = \delta \cdot 1 \cdot y
$$

---

### 📚 **গুরুত্বপূর্ণ টার্মস**

| টার্ম               | ব্যাখ্যা                                              |
| ------------------- | ----------------------------------------------------- |
| **Node**            | একটি অপারেশন বা মান (যেমন: x, +, log)                 |
| **Edge**            | একটি ভ্যারিয়েবল যেটা দুটি Node এর মধ্যে ফ্লো করে      |
| **Local Gradient**  | নির্দিষ্ট একটি অপারেশন এর gradient                    |
| **Global Gradient** | Loss function respect-এ চূড়ান্ত gradient              |
| **Chain Rule**      | Derivative propagate করার জন্য ব্যবহৃত নিয়ম           |
| **Loss Function**   | Prediction কতটা ভুল হয়েছে তা নির্ণয়কারী ফাংশন         |
| **Learning Rate**   | Weight পরিবর্তনের গতি নিয়ন্ত্রণকারী হাইপারপ্যারামিটার |

---

### 🧠 **Practical ব্যবহার: নিউরাল নেটওয়ার্ক ট্রেনিং**

* **Computational Graph** নিউরাল নেটওয়ার্ক শেখার সময় প্রতিটি ধাপে কোথায় কী অপারেশন হচ্ছে তা বোঝায়।
* **Backpropagation** ব্যবহার করে প্রতিটি Layer এর weight/bias update করে।
* **Auto-differentiation (যেমন PyTorch, TensorFlow)** computational graph ব্যবহার করে gradient স্বয়ংক্রিয়ভাবে বের করে।

---

### ✅ **শেষ কথা**

* Computational Graph হলো নিউরাল নেটওয়ার্কের মস্তিষ্কস্বরূপ, যা প্রতিটি গাণিতিক অপারেশনকে ট্র্যাক করে।
* Backpropagation এই Graph এর উপর ভিত্তি করেই কাজ করে Gradient Flow হিসাব করে।
* এটি Model Train করার efficiency এবং scalability অনেক গুণ বাড়িয়ে দেয়।

---


---
