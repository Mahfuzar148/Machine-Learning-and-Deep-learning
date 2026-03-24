
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

---

# 🔹 Step 1: Basic Scatter Plot

```python
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset("tips")

sns.scatterplot(x="total_bill", y="tip", data=df)

plt.show()
```

### 🧠 Explanation:

* X-axis → total bill
* Y-axis → tip
* প্রতিটা dot = ১টা customer

---

# 🔹 Step 2: `hue` (Color by category)

```python
sns.scatterplot(x="total_bill", y="tip", hue="sex", data=df)
```

### 🧠 `hue` কী?

👉 Data কে color দিয়ে আলাদা করে

### 📌 Possible values:

* `sex` (Male/Female)
* `smoker` (Yes/No)
* `day` (Thur, Fri, Sat, Sun)
* `time` (Lunch/Dinner)

### 🎯 Use:

👉 category-wise pattern দেখতে

---

# 🔹 Step 3: `style` (Marker style)

```python
sns.scatterplot(x="total_bill", y="tip", hue="sex", style="smoker", data=df)
```

### 🧠 `style` কী?

👉 Different shape (circle, cross, etc.)

### 📌 Possible values:

* `smoker`
* `sex`
* `day`

### 🎯 Use:

👉 visually differentiate categories

---

# 🔹 Step 4: `size` (Dot size)

```python
sns.scatterplot(x="total_bill", y="tip", size="size", data=df)
```

### 🧠 `size` কী?

👉 Dot-এর size change করে

### 📌 Possible values:

* `size` (number of people)
* any numeric column

### 🎯 Use:

👉 3rd dimension দেখাতে

---

# 🔹 Step 5: `sizes` (Size range control)

```python
sns.scatterplot(x="total_bill", y="tip", size="size", sizes=(20, 200), data=df)
```

### 🧠 `sizes` কী?

👉 Minimum → Maximum size control

### 📌 Format:

* `(min, max)` tuple

---

# 🔹 Step 6: `palette` (Color theme)

```python
sns.scatterplot(x="total_bill", y="tip", hue="day", palette="Set2", data=df)
```

### 🧠 `palette` কী?

👉 Color style control

### 📌 Popular values:

* `"Set1"`, `"Set2"`, `"cool"`, `"viridis"`, `"deep"`

---

# 🔹 Step 7: `alpha` (Transparency)

```python
sns.scatterplot(x="total_bill", y="tip", alpha=0.5, data=df)
```

### 🧠 `alpha` কী?

👉 Dot transparency

### 📌 Range:

* 0 → invisible
* 1 → fully visible

---

# 🔹 Step 8: `markers` (Custom shapes)

```python
sns.scatterplot(
    x="total_bill",
    y="tip",
    style="smoker",
    markers=["o", "X"],
    data=df
)
```

### 🧠 `markers` কী?

👉 Shape manually set করা

### 📌 Examples:

* `"o"` → circle
* `"X"` → cross
* `"s"` → square

---

# 🔹 Step 9: `legend`

```python
sns.scatterplot(x="total_bill", y="tip", hue="sex", legend="full", data=df)
```

### 🧠 `legend` কী?

👉 Label show/hide control

### 📌 Values:

* `"auto"`
* `"brief"`
* `"full"`
* `False`

---

# 🔥 Final Combined Example

```python
sns.scatterplot(
    x="total_bill",
    y="tip",
    hue="day",
    style="smoker",
    size="size",
    sizes=(20, 200),
    palette="Set2",
    alpha=0.7,
    data=df
)

plt.title("Advanced Scatter Plot")
plt.show()
```

---

# 🧠 Final Understanding

👉 Scatter plot = 2D relation
👉 Extra parameters = more dimensions

| Parameter | কাজ            |
| --------- | -------------- |
| hue       | color grouping |
| style     | shape change   |
| size      | bubble size    |
| palette   | color theme    |
| alpha     | transparency   |

---

# 🚀 Pro Tip

👉 Machine Learning EDA-তে:

* scatterplot + hue = **gold combo** ⭐

---

---

# 📘 `seaborn.scatterplot()` – Official Style Documentation (Simplified)

```python
sns.scatterplot(
    data=None,
    x=None, y=None,
    hue=None,
    style=None,
    size=None,
    palette=None,
    sizes=None,
    markers=True,
    alpha=None,
    legend='auto'
)
```

---

# 🔹 1. `hue` — Color Encoding

## 📌 Definition:

Maps data values to **different colors**

## 📥 Input:

* Column name (categorical বা numerical)

## 📊 Behavior:

### ✔ Categorical:

* প্রতিটা class → আলাদা color

Example:

```python
hue="sex"
```

Output:

* Male → blue
* Female → orange

---

### ✔ Numerical:

* gradient color (low → high)

Example:

```python
hue="total_bill"
```

---

## 🎯 Use Case:

* Category comparison
* Group visualization

---

# 🔹 2. `style` — Marker Shape Encoding

## 📌 Definition:

Maps data values to **different marker shapes**

## 📥 Input:

* Categorical column

## 📊 Behavior:

Example:

```python
style="smoker"
```

Output:

* Yes → circle
* No → cross

---

## 🎯 Use Case:

* Extra grouping (color + shape combo)

---

# 🔹 3. `size` — Marker Size Encoding

## 📌 Definition:

Maps values to **marker size (bubble plot)**

## 📥 Input:

* Numeric column (recommended)

## 📊 Behavior:

Example:

```python
size="size"
```

Output:

* small value → small dot
* large value → big dot

---

## 🔹 `sizes` (optional)

```python
sizes=(min, max)
```

Example:

```python
sizes=(20, 200)
```

👉 controls min & max size

---

## 🎯 Use Case:

* Third dimension visualization

---

# 🔹 4. `palette` — Color Mapping

## 📌 Definition:

Controls **color theme for `hue`**

## 📥 Input:

* String (palette name)
* list of colors
* dict mapping

---

## 🎨 Common Palettes:

### ✔ Categorical:

* `"deep"` (default)
* `"muted"`
* `"pastel"`
* `"bright"`
* `"dark"`
* `"Set1"`, `"Set2"`, `"Set3"`

---

### ✔ Continuous:

* `"viridis"` ⭐
* `"plasma"`
* `"coolwarm"`

---

## 📊 Example:

```python
palette="Set2"
```

---

## 🎯 Use Case:

* Better visualization
* Presentation quality plots

---

# 🔹 5. `markers`

## 📌 Definition:

Custom marker shapes

## 📥 Input:

* list or dict

Example:

```python
markers=["o", "X"]
```

---

# 🔹 6. `alpha` — Transparency

## 📌 Definition:

Controls opacity

## 📥 Range:

* 0 → transparent
* 1 → solid

Example:

```python
alpha=0.5
```

---

# 🔹 7. `legend`

## 📌 Definition:

Controls legend display

## 📥 Values:

* `"auto"`
* `"brief"`
* `"full"`
* `False`

---

# 💻 Full Example (All Combined)

```python
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset("tips")

sns.scatterplot(
    x="total_bill",
    y="tip",
    hue="day",
    style="smoker",
    size="size",
    sizes=(20, 200),
    palette="Set2",
    markers=True,
    alpha=0.7,
    legend="full",
    data=df
)

plt.title("Seaborn Scatterplot Full Example")
plt.show()
```

---

# 🧠 Conceptual Visualization

---

# 🧠 Final Summary Table

| Parameter | কাজ            | Data Type             |
| --------- | -------------- | --------------------- |
| hue       | color mapping  | categorical / numeric |
| style     | marker shape   | categorical           |
| size      | marker size    | numeric               |
| palette   | color theme    | string/list           |
| sizes     | size range     | tuple                 |
| alpha     | transparency   | float                 |
| legend    | legend control | string/bool           |

---

# 🚀 Pro Insight

👉 `hue + style + size` একসাথে use করলে:
➡️ 4D visualization possible 😎

---

