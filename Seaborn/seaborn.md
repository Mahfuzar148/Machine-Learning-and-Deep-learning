`Seaborn` হলো Python-এর একটা **statistical data visualization library**, যা খুব সুন্দর graph বানাতে use হয় (especially ML project-এ 📊).

আমি নিচে **Seaborn-এর main functionsগুলো category-wise full list** দিচ্ছি 👇

---

# 🧠 Seaborn Functions List (Complete Guide)




# 📊 Visual Idea

![Image](https://images.openai.com/static-rsc-4/CxNBK6bbHYX53-6rptfDYlkCdBQNnnJA0e532VUENgubBiXZygEXMocJ08cAJ20L66Gu7mJ36SIAq_KQusbEJH4h2xUyMzC40MZI1ZwPlqL29ZUoWjR6PkGTzCOK1uKxyOI_m94EYS6IuFgH3WMPFHlbd5zn__VIHuf5JFcMTR6M1KNAbhBoraJt9zvg-7Gp?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/hSCOjlQNDhMUMH3pkbw61NlM4veSfO3DaLoV_Gkij2olohe3zO1sWM4QNITh12dsj3bqKIGo_K2jKirLHhM5sXyGsD-Grti1n1BaNbFQ865udTwp4LqqCRBRV1jafeUhBSJ54bMjWL8_axk2EmKyr_aeKpD9SkmurQap8QJzEIgSVhWgbbCQfZzkavQ1pKvK?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/O5Llw0GMQavzfRyKFcuURiVz3fLdDzE14vIgBvG9wACmeJRzoy1sv30_Q18vFRw503s6fYAGH4Tx5PEkdkNyJHOh76-mKImVG-0fVPMmUFjGY853k4LDA9wNLsl70In1oAC8T7LPvTroUnmNGsVjZoVkqY6vzDvLCztficCdmuPFGgvKivJRUW3ef1t4Y07W?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/vJbKBY0RH5Tr0z6K8W3zQD_pzHG_Iadp4vclhnhWB999O4prSvLBWjhbhRnIG_XUdxGJbzHEeK7g39rvwFuTDTMWEo6GuklsZy2gaAi3AvATs-Hv-cTRjR2wAWP9G_xdDrXDlgocX2PPFtAftWV_FQDxhhkMe4M54F6YzfvldXLqOB4Gy8kwXSdwMDd2uNf8?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/_dqrh8qFAV1Ykx22Fzbxy1FGKrgesYDQlA4zepE3hBxbY1SAoDyDB3OAQjaDKk96C7pX_DkVtRqtUTqgnyE5FThBiJdjHO-l3TTj2_-txenbFB4pX2ybK33XnWZahY3--xh65hhEuN_TnSgwHM-7NoX42P3yWmxP_CNMJTw2Acwm7MlMobbcTvE3AimRyDCG?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/yQNzDITARDBQMWPu0ojMZr26IYUddpbJH8yobZAwUTgGmyiw3E9Qc57DVC4EPLLJ6gL6ce-YTdpU33dkyNjThI5DAJoJqMmh5DDq_GRrmX5iesVgfE_WKxSGaVdvUdOrUGuSIs6EykaLNB2hxsGlxIZgV-jnnwSAV3w2t_g6JedX5kxRpztAwbR6oYQvz6cU?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/pJ4Ped75NPeVX2EXQZygLQRr4r_9_pnM4J9AnZhpqlMMYUEuIVj-JXCRJx3KSvIUcbSBTdMUQroFJiZ7m6J9RoZO9ZXo56eonJuhz_XJSyzqdCkCCx2WPh8KKmWtX7TzZ8SDfBwphn9XjG6aUgvhOvQ6AEwcQsG1grt5_cSdfb9PngCRoQZEMFXuP4a9-p8j?purpose=fullsize)

---

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

