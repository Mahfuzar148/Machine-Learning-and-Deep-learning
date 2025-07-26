
---

## 📌 সবচেয়ে বেশি ব্যবহৃত Seaborn Functions

এই ফাংশনগুলো **প্রতিদিনের ডেটা এনালাইসিস**, **EDA (Exploratory Data Analysis)** এবং **reporting** এ সবথেকে বেশি ব্যবহৃত হয়।

---

### 1. 🔹 `sns.barplot()`

✅ **কাজ:**
একটি ক্যাটাগরি অনুযায়ী গড়/summary মান দেখায়। Error bar দিয়ে variation বোঝায়।

✅ **সিনট্যাক্স:**

```python
sns.barplot(data=tips, x=\"day\", y=\"total_bill\", ci=\"sd\")
```

✅ **প্যারামিটার ও মান:**

| Parameter   | মান                    | কাজ                      |
| ----------- | ---------------------- | ------------------------ |
| `x`, `y`    | column name            | category এবং numeric মান |
| `hue`       | group column           | ভেতরের subgroup          |
| `ci`        | `'sd'`, `95`           | error bar type           |
| `estimator` | `np.mean`, `np.median` | কিসের গড় নেবে            |

---

### 2. 🔹 `sns.boxplot()`

✅ **কাজ:**
5-number summary দেখায় (min, Q1, median, Q3, max) + outlier।

✅ **সিনট্যাক্স:**

```python
sns.boxplot(data=tips, x=\"day\", y=\"total_bill\")
```

| Parameter       | মান        | কাজ                        |
| --------------- | ---------- | -------------------------- |
| `x`, `y`, `hue` | column     | category-wise distribution |
| `notch`         | True/False | median notch               |
| `showmeans`     | True       | mean দেখাবে                |

---

### 3. 🔹 `sns.violinplot()`

✅ **কাজ:**
Distribution + box summary + KDE একসাথে দেখায়।

✅ **সিনট্যাক্স:**

```python
sns.violinplot(data=tips, x=\"day\", y=\"total_bill\", inner='box')
```

| Parameter | মান                           | কাজ                    |
| --------- | ----------------------------- | ---------------------- |
| `inner`   | `'box'`, `'stick'`, `'point'` | ভিতরের স্ট্যাট         |
| `split`   | True                          | hue-based split violin |

---

### 4. 🔹 `sns.histplot()`

✅ **কাজ:**
Numeric ভেরিয়েবলের histogram (frequency plot) তৈরি করে।

✅ **সিনট্যাক্স:**

```python
sns.histplot(data=tips, x=\"total_bill\", bins=20, kde=True)
```

| Parameter | মান  | কাজ                 |
| --------- | ---- | ------------------- |
| `bins`    | int  | histogram bin count |
| `kde`     | True | KDE যোগ করে         |

---

### 5. 🔹 `sns.scatterplot()`

✅ **কাজ:**
দুই numeric ভেরিয়েবলের মধ্যে সম্পর্ক দেখায় (scatter plot)।

✅ **সিনট্যাক্স:**

```python
sns.scatterplot(data=tips, x=\"total_bill\", y=\"tip\", hue=\"sex\")
```

| Parameter | মান      | কাজ                |
| --------- | -------- | ------------------ |
| `hue`     | category | রঙে আলাদা গ্রুপ    |
| `style`   | marker   | মার্কার ভিন্নতা    |
| `size`    | column   | point size control |

---

### 6. 🔹 `sns.lineplot()`

✅ **কাজ:**
একটি trend বা continuous change দেখায়।

✅ **সিনট্যাক্স:**

```python
sns.lineplot(data=tips, x=\"size\", y=\"tip\", hue=\"sex\")
```

| Parameter | মান      | কাজ             |
| --------- | -------- | --------------- |
| `markers` | True     | point marker    |
| `ci`      | 95, 'sd' | confidence band |

---

### 7. 🔹 `sns.heatmap()`

✅ **কাজ:**
2D Matrix-এর মধ্যে **রঙের সাহায্যে মান** দেখায়।

✅ **সিনট্যাক্স:**

```python
sns.heatmap(data=corr_matrix, annot=True, cmap=\"coolwarm\")
```

| Parameter | মান       | কাজ                |
| --------- | --------- | ------------------ |
| `annot`   | True      | প্রতিটি cell-এ মান |
| `cmap`    | color map | রঙের থিম           |

---

### 8. 🔹 `sns.pairplot()`

✅ **কাজ:**
সব numeric column-এর মধ্যে pairwise সম্পর্ক দেখায় (scatter + histogram combo)।

✅ **সিনট্যাক্স:**

```python
sns.pairplot(data=tips, hue=\"sex\")
```

---

### 9. 🔹 `sns.lmplot()`

✅ **কাজ:**
Linear regression + scatter plot, subplot-সহ।

✅ **সিনট্যাক্স:**

```python
sns.lmplot(data=tips, x=\"total_bill\", y=\"tip\", hue=\"sex\", col=\"time\")
```

---

## 🌟 **সবচেয়ে Versatile Function: `sns.catplot()`**

✅ **কারণ:**
এটি হলো **Figure-level** wrapper function যেটা দিয়ে bar, box, violin, strip, swarm — সব ক্যাটাগরিকাল প্লট বানানো যায়।

✅ **সিনট্যাক্স:**

```python
sns.catplot(data=tips, x=\"day\", y=\"total_bill\", kind=\"box\", hue=\"sex\", col=\"time\")
```

✅ **Task Coverage:**

* Faceting (col/row)
* Hue-based comparison
* Kind: `'box'`, `'violin'`, `'bar'`, `'strip'`, `'swarm'`, `'point'`, `'count'`

📌 **প্যারামিটার:**
Same as above + `kind`, `col`, `row`, `height`, `aspect`

---

## ✅ উপসংহার: কোনটা কবে ব্যবহার করবো?

| Function        | কাজ                       | সবচেয়ে উপযোগী ক্ষেত্রে |
| --------------- | ------------------------- | ----------------------- |
| `catplot()`     | All categorical plots     | ক্যাটাগরির data EDA     |
| `scatterplot()` | দুই ভেরিয়েবলের সম্পর্ক    | correlation check       |
| `boxplot()`     | outlier সহ distribution   | clean + visual summary  |
| `violinplot()`  | shape + spread একসাথে     | KDE + summary           |
| `barplot()`     | গড় মান ও error bar        | grouped summary         |
| `lineplot()`    | trend/time series         | continuous var tracking |
| `heatmap()`     | matrix (e.g. correlation) | grid-based analysis     |
| `pairplot()`    | সব numeric var একসাথে     | full data overview      |

---


```python
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pandas as pd

st.title("🔥 Most Used Seaborn Functions - Demo Dashboard")

# Load dataset
tips = sns.load_dataset("tips")
st.write("Using the famous 'tips' dataset")

# 1. barplot
st.header("1. sns.barplot()")
fig1, ax1 = plt.subplots()
sns.barplot(data=tips, x="day", y="total_bill", ci="sd", ax=ax1)
st.pyplot(fig1)

# 2. boxplot
st.header("2. sns.boxplot()")
fig2, ax2 = plt.subplots()
sns.boxplot(data=tips, x="day", y="total_bill", notch=True, showmeans=True, ax=ax2)
st.pyplot(fig2)

# 3. violinplot
st.header("3. sns.violinplot()")
fig3, ax3 = plt.subplots()
sns.violinplot(data=tips, x="day", y="total_bill", inner="box", ax=ax3)
st.pyplot(fig3)

# 4. histplot
st.header("4. sns.histplot()")
fig4, ax4 = plt.subplots()
sns.histplot(data=tips, x="total_bill", bins=20, kde=True, ax=ax4)
st.pyplot(fig4)

# 5. scatterplot
st.header("5. sns.scatterplot()")
fig5, ax5 = plt.subplots()
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="sex", style="time", size="size", ax=ax5)
st.pyplot(fig5)

# 6. lineplot
st.header("6. sns.lineplot()")
fig6, ax6 = plt.subplots()
sns.lineplot(data=tips, x="size", y="tip", hue="sex", markers=True, ci="sd", ax=ax6)
st.pyplot(fig6)

# 7. heatmap
st.header("7. sns.heatmap()")
corr_matrix = tips.corr(numeric_only=True)
fig7, ax7 = plt.subplots()
sns.heatmap(data=corr_matrix, annot=True, cmap="coolwarm", ax=ax7)
st.pyplot(fig7)

# 8. pairplot
st.header("8. sns.pairplot()")
fig8 = sns.pairplot(data=tips, hue="sex")
st.pyplot(fig8.figure)

# 9. lmplot
st.header("9. sns.lmplot()")
fig9 = sns.lmplot(data=tips, x="total_bill", y="tip", hue="sex", col="time")
st.pyplot(fig9.figure)

# 10. catplot (most versatile)
st.header("10. sns.catplot()")
fig10 = sns.catplot(data=tips, x="day", y="total_bill", kind="box", hue="sex", col="time")
st.pyplot(fig10.figure)
```

### ✅ কী কী আছে:

1. `sns.barplot()` – গড় মান ও ভ্যারিয়েশন
2. `sns.boxplot()` – outlier সহ distribution
3. `sns.violinplot()` – KDE + summary
4. `sns.histplot()` – histogram + KDE
5. `sns.scatterplot()` – numeric var relationship
6. `sns.lineplot()` – trend দেখানোর জন্য
7. `sns.heatmap()` – correlation matrix
8. `sns.pairplot()` – সব numeric ভেরিয়েবলের মধ্যে সম্পর্ক
9. `sns.lmplot()` – regression analysis
10. `sns.catplot()` – সর্বাধিক flexible categorical plot


