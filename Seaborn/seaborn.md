`Seaborn` হলো Python-এর একটা **statistical data visualization library**, যা খুব সুন্দর graph বানাতে use হয় (especially ML project-এ 📊).

আমি নিচে **Seaborn-এর main functionsগুলো category-wise full list** দিচ্ছি 👇

---

# 🧠 Seaborn Functions List (Complete Guide)

---

# 🔹 1. Relational Plots (Relationship দেখার জন্য)

👉 দুইটা variable-এর relation বুঝতে

### Functions:

* `scatterplot()` → scatter plot
* `lineplot()` → line graph
* `relplot()` → high-level interface

---








# 🔹 2. Distribution Plots (Data distribution)

👉 data কিভাবে ছড়ানো আছে

### Functions:

* `histplot()` → histogram
* `kdeplot()` → density curve
* `displot()` → advanced distribution plot
* `ecdfplot()` → cumulative distribution

---

# 🔹 3. Categorical Plots (Category-based analysis)

👉 categorical vs numeric data

### Functions:

* `barplot()`
* `countplot()`
* `boxplot()`
* `violinplot()`
* `stripplot()`
* `swarmplot()`
* `catplot()` → high-level

---

# 🔹 4. Matrix Plots (Matrix visualization)

👉 correlation বা matrix data

### Functions:

* `heatmap()` ⭐ (very important)
* `clustermap()`

---

# 🔹 5. Regression Plots (ML relation)

👉 regression analysis

### Functions:

* `regplot()`
* `lmplot()`
* `residplot()`

---

# 🔹 6. Multi-Plot Grids

👉 একাধিক plot একসাথে

### Functions:

* `FacetGrid`
* `PairGrid`
* `pairplot()` ⭐ (very popular)
* `jointplot()`

---

# 🔹 7. Style & Theme

👉 plot সুন্দর করা

### Functions:

* `set_style()`
* `set_context()`
* `set_palette()`
* `set_theme()`

---

# 🔹 8. Color & Palette

👉 color control

### Functions:

* `color_palette()`
* `cubehelix_palette()`
* `light_palette()`
* `dark_palette()`

---

# 🔹 9. Utility Functions

👉 extra support

### Functions:

* `load_dataset()` → built-in dataset
* `despine()` → border remove
* `move_legend()`
* `set()`

---

# 🔥 Most Important Functions (Must Know)

👉 এগুলো 90% project-এ use হয়:

* `scatterplot()`
* `lineplot()`
* `histplot()`
* `countplot()`
* `boxplot()`
* `heatmap()` ⭐
* `pairplot()` ⭐
* `barplot()`

---

# 🧩 Example Code

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = sns.load_dataset("tips")

# Scatter plot
sns.scatterplot(x="total_bill", y="tip", data=df)

# Heatmap
sns.heatmap(df.corr(), annot=True)

plt.show()
```

---

# ⚠️ Important Note

👉 Seaborn internally uses:

* `matplotlib`

👉 So always use:

```python
import matplotlib.pyplot as plt
```

---

# 🧠 Short Summary

✔ Seaborn = Beautiful + easy visualization
✔ 9 major categories
✔ ML project-এ must-use tool

---

