
---

## 📘 ডকুমেন্টেশন: Computational Graph ও Backpropagation

### 🔍 পরিচিতি

Computational Graph এমন একটি গাণিতিক কাঠামো, যা complex function-কে ছোট ছোট operation-এ ভাগ করে দৃশ্যমানভাবে উপস্থাপন করে। নিউরাল নেটওয়ার্কে এটি forward pass ও backward pass ট্র্যাক করতে ব্যবহার হয়।

---

### 🧩 উদাহরণ: \( z = x \times y \)

একটি সাধারণ computational graph:

```text
    x = 3
      \
       * ---> z = x × y = 3 × 4 = 12
      /
    y = 4
```

**Forward Pass:**
- ইনপুট: x=3, y=4
- অপারেশন: z = x × y → z = 12

**Backward Pass (Gradient Calculation):**
- \( \frac{\partial z}{\partial x} = y = 4 \)
- \( \frac{\partial z}{\partial y} = x = 3 \)

এখানে আমরা local gradient পেয়েছি।

---

### 🔁 Backpropagation এর পদক্ষেপ

নিউরাল নেটওয়ার্কে ব্যাকপ্রোপাগেশনের ৪টি ধাপ:

1. **Forward Pass:**
   - ইনপুট → অপারেশন → আউটপুট

2. **Loss Function হিসাব:**
   - Prediction এর সাথে Actual এর পার্থক্য বের করে

3. **Backward Pass (Gradient হিসাব):**
   - Loss কে respect করে weight ও bias-এর partial derivative নেওয়া হয়

4. **Weight Update (Gradient Descent):**
   - $$ w_{\text{new}} = w_{\text{old}} - \eta \cdot \frac{\partial L}{\partial w} $$  
   - \( \eta \) = learning rate

---

### 📊 Gradient Flow: Layer-wise Visualization

```text
Input --> Layer1 --> Layer2 --> ... --> Loss
             ↑        ↑
           dw1       dw2       ← Gradients flow backward
```

**Gradient Flow back করে:** শেষের layer থেকে প্রথম layer-এর দিকে weight, bias ও activation এর respect এ partial derivatives বের করে।

---

### 🧠 Practical Use: Neural Network Training

- Computational Graph ব্যাকপ্রোপাগেশনের জন্য ভিত্তি প্রদান করে
- Graph-এর প্রতিটি node-এর gradient হিসাব করে model শেখে
- এটি training efficiency এবং scalability বৃদ্ধি করে

---

### 📚 গুরুত্বপূর্ণ টার্মস

| Term | ব্যাখ্যা |
|------|---------|
| Node | একটি অপারেশন বা ভেরিয়েবল |
| Edge | অপারেশনের input-output সংযোগ |
| Local Gradient | একটি node এর gradient |
| Global Gradient | চূড়ান্ত loss respect এ gradient |
| Chain Rule | gradient কে propagate করার নিয়ম |
| Loss Function | prediction accuracy পরিমাপক |

---

### ✅ শেষ কথা

Computational Graph হল নিউরাল নেটওয়ার্ক শেখার জন্য গাণিতিক ম্যাপ, যার মাধ্যমে ব্যাকপ্রোপাগেশন চালিয়ে weight update করা যায়।

