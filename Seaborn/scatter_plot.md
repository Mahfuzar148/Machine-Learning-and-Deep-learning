
---

# 📊 Scatter Plot কী?

A **scatter plot** হলো এমন একটা graph যেখানে:

* দুইটা numerical variable দেখানো হয়
* প্রতিটা data point = একটা dot

👉 Example:

* X-axis → house size
* Y-axis → price

➡️ প্রতিটা dot = একটা house

---

## 🖼️ Example Visualization

![Image](https://images.openai.com/static-rsc-4/ATDJr6LDrzbADgAmgwEbQ-tzASQrBYD2xaZS4NsEkapEerdTe9ywFbNZq9ECL5aI2Oei-lMqsL_Oxs0sfH-_Cnqa2Gih36cnBUEYKZrnhzCi1zwQo7tPiei5f2kgYYF4XLs7I5EMwlh645W6eD4c2pXdYjqfNnHYRd_5Bzq9vQ81IShhbWWdL0Qy3vNAxBxJ?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/ZzVYFbiAxOq2zt8jlQte1T6dKjNOBwm1K2W7XNbOUYeXqrZDlHzDuvJclrOp-HA1Keb4L0LoIkg-yxikByLaK53gNEFQAWB38SNo4DM6LtL8Qstb7yWAXGLxd7aC_19_QbGJkA3gVl96ouMMYLdM7CBEmbjkUW7ziFSmfxvvuUrhl7HO2PPE_DpEXRD0SZsS?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/CKeVH8x6FJkGiEnHA3Di1bbPXc5x2QLJfmi4R7kToANjwQoz9VZwWk81vtXp0-fV-y55cw8FRSuy9-PzaLZAlc1zmUpSWLC5oHE1J1AcnQzakCAqZ8HfhYepS7IqhW8rwf7xwrvfLA-kkZaOQcWwmktDTz2bqvCQC-0-dEzhhz-dtAoM7wnbqSZ0ZmUrgmtS?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/klUbjO1oCqhlWgRNR8ggT6brR2LQt7xKDfH3pMhb5qK-veLssBuDRmIJUvIUl-Ul1qO9Nr0zie-2JXfW9ZHrobH1-AEo4YU4FKrOJY-Lvv622xj17E4zNxD4Lu0l5d5-q5WrbKuS6ohaBfNRHRnj7W1taAYdRM1FI2YuQbUZgS-PhZewwvB1tYWR4F79dnBE?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/3Xy0iZleZWM6vghQuzPPD89XS27kQPMsN4oxV0HldW_bTksNa-XACo-LFn2-v0oWMkXFIazKId0LPalgkwLWAYQtfu_G-XjtYurd8iPUWxKVGO7wzUNxYqJxI3LpLQw6EBGazi_u1w9zFNhGsdAa_dn7KC-HlAIwzWzQg4deiFNu0Bxsdynx7hiGcgkQG47G?purpose=fullsize)

---

# 🧠 Scatter Plot কেন ব্যবহার করি?

👉 মূল কাজ:

* Relationship (relation) বোঝা
* Pattern detect করা
* Outlier খুঁজে বের করা

---

# 📌 কখন ব্যবহার করবো?

Use scatter plot when:

* দুইটা **numerical feature** আছে
* তুমি relation দেখতে চাও

### Example:

* Height vs Weight
* Study time vs Marks
* Area vs Price

---

# 🔍 Relationship Types

Scatter plot দিয়ে তুমি বুঝতে পারো:

* 📈 Positive correlation → X বাড়লে Y বাড়ে
* 📉 Negative correlation → X বাড়লে Y কমে
* ❌ No correlation → কোন relation নেই

---

# ⚙️ Seaborn `scatterplot()` Syntax

```python
sns.scatterplot(x=, y=, data=)
```

---

## 🔥 Full Syntax (Advanced)

```python
sns.scatterplot(
    x="column1",
    y="column2",
    data=df,
    hue="category",      # color based on category
    style="category",    # different marker styles
    size="column",       # size variation
)
```

---

# 💻 Code Example (Basic)

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = sns.load_dataset("tips")

# Create scatter plot
sns.scatterplot(x="total_bill", y="tip", data=df)

# Show plot
plt.show()
```

👉 Meaning:

* X = total bill
* Y = tip

---

# 💻 Code Example (Advanced)

```python
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset("tips")

sns.scatterplot(
    x="total_bill",
    y="tip",
    hue="sex",       # male vs female color
    style="smoker",  # different marker
    size="size",     # size of group
    data=df
)

plt.title("Total Bill vs Tip")
plt.show()
```

---

# 🧠 Real Life Example (ML)

👉 House Price Prediction:

```python
sns.scatterplot(x="GrLivArea", y="SalePrice", data=df)
```

➡️ বুঝতে পারবে:

* বাড়ির area বাড়লে price বাড়ে কিনা

---

# ⚠️ Common Mistakes

❌ categorical data use করা
❌ too many overlapping points
❌ scaling না করা (ML case)

---

# 🧩 Tips (Important)

✔ Large dataset → use alpha (transparency):

```python
sns.scatterplot(x="x", y="y", data=df, alpha=0.5)
```

✔ Overlapping fix:

* smaller size
* transparency

---

# 🧠 Short Summary

👉 Scatter plot:

* Two numeric variables
* Dot-based graph
* Relationship বুঝতে use হয়

---


