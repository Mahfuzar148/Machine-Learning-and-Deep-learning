
---

# 🧠 Relational Plot কী?

👉 **Relational plot** হলো এমন plot যেটা ব্যবহার করা হয়
➡️ **দুইটা (বা তার বেশি) variable-এর relationship বোঝার জন্য**

📌 Example:

* Height vs Weight
* Time vs Sales
* Study hours vs Marks

👉 অর্থাৎ:

> **একটা variable বাড়লে আরেকটা কীভাবে change করে**

---

# 🎯 কখন ব্যবহার করবো?

তুমি relational plot ব্যবহার করবে যখন:

✔️ 2টা numeric variable এর সম্পর্ক দেখতে চাও
✔️ trend (increase/decrease) বুঝতে চাও
✔️ pattern detect করতে চাও
✔️ outlier খুঁজতে চাও

---

# 🔹 Types of Relational Plot (Seaborn)

### 1️⃣ scatterplot()

👉 Individual data points দেখায়

---

## 📌 Example: Scatter Plot

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

sns.scatterplot(x="total_bill", y="tip", data=tips)
plt.show()
```

---

## 🧠 কী বুঝাবে?

👉 bill বাড়লে tip বাড়ছে কিনা
👉 data spread কেমন

---

# 🔹 Advanced scatterplot

```python
sns.scatterplot(
    x="total_bill",
    y="tip",
    hue="sex",        # color দিয়ে category
    size="size",      # size দিয়ে info
    data=tips
)
plt.show()
```

---

# 2️⃣ lineplot()

👉 Trend (সময় বা ordered data) দেখায়

---

## 📌 Example: Line Plot

```python
fmri = sns.load_dataset("fmri")

sns.lineplot(x="timepoint", y="signal", data=fmri)
plt.show()
```

---

## 🧠 কী বুঝাবে?

👉 সময়ের সাথে signal কেমন change হচ্ছে

---

## 🔹 Advanced lineplot

```python
sns.lineplot(
    x="timepoint",
    y="signal",
    hue="event",
    style="region",
    data=fmri
)
plt.show()
```

---

# 3️⃣ relplot() (High-level function)

👉 এটা **scatterplot + lineplot wrapper**

👉 grid + multiple plots support করে

---

## 📌 Example: relplot (scatter)

```python
sns.relplot(
    x="total_bill",
    y="tip",
    hue="sex",
    col="time",
    data=tips
)
plt.show()
```

---

## 📌 Example: relplot (line)

```python
sns.relplot(
    x="timepoint",
    y="signal",
    kind="line",
    data=fmri
)
plt.show()
```

---

# 🧠 relplot কেন use করবো?

✔️ multiple subplot (faceting)
✔️ clean visualization
✔️ high-level control

---

# 🔥 Summary Table

| Function      | Use case                     |
| ------------- | ---------------------------- |
| scatterplot() | relation between 2 variables |
| lineplot()    | trend over time              |
| relplot()     | advanced / multiple plots    |

---

# 🎯 Real Life Example

👉 Student data:

* hours vs marks → scatterplot
* sales over months → lineplot
* different regions compare → relplot

---

# 🧾 Final Concept (Most Important)

👉 Relational plot answer করে:

> **“একটা variable change হলে অন্যটা কীভাবে react করে?”**

---

