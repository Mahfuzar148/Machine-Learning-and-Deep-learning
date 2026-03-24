চলো একটা **complete example** করি যেখানে:

✅ Pandas দিয়ে **data inspection**
✅ Seaborn দিয়ে **scatter plot visualization**

সব একসাথে থাকবে + clean structured code 👇

---

# 🧠 Full Code: Data Inspection + Scatter Plot

```python
# ================================
# 1. Import Libraries
# ================================
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ================================
# 2. Load Dataset
# ================================
# Seaborn built-in dataset (tips dataset)
df = sns.load_dataset("tips")

# ================================
# 3. Data Inspection (Pandas)
# ================================

# First 5 rows
print("---- HEAD ----")
print(df.head())

# Last 5 rows
print("---- TAIL ----")
print(df.tail())

# Shape (rows, columns)
print("---- SHAPE ----")
print(df.shape)

# Column names
print("---- COLUMNS ----")
print(df.columns)

# Data types
print("---- DATA TYPES ----")
print(df.dtypes)

# Dataset info (null values, types)
print("---- INFO ----")
print(df.info())

# Statistical summary
print("---- DESCRIBE ----")
print(df.describe())

# Mean values
print("---- MEAN ----")
print(df.mean(numeric_only=True))

# Max values
print("---- MAX ----")
print(df.max(numeric_only=True))

# Missing values check
print("---- MISSING VALUES ----")
print(df.isnull().sum())

# ================================
# 4. Scatter Plot (Seaborn)
# ================================

# Basic scatter plot
sns.scatterplot(x="total_bill", y="tip", data=df)

# Add title
plt.title("Total Bill vs Tip")

# Show plot
plt.show()

# ================================
# 5. Advanced Scatter Plot
# ================================

sns.scatterplot(
    x="total_bill",
    y="tip",
    hue="sex",       # color by gender
    style="smoker",  # marker style
    size="size",     # bubble size
    sizes=(20, 200),
    palette="Set2",
    alpha=0.7,
    data=df
)

plt.title("Advanced Scatter Plot")
plt.show()
```

---

# 🧠 Code Explanation (Short)

## 🔍 Data Inspection Part:

* `head()` → প্রথম data
* `shape` → কত row/column
* `info()` → null + datatype
* `describe()` → mean, max, min

---

## 📊 Visualization Part:

* `scatterplot()` → relation দেখায়
* `hue` → color grouping
* `style` → shape change
* `size` → bubble size

---

# 🖼️ Scatter Plot Idea

![Image](https://images.openai.com/static-rsc-4/qxz3lTLla_5c7QgRVMgc0pvfajvRp5t2D3k9lXzeHU0zgQv1QGrqa1o863VtbYL3w7JiqhEq4eXs3mU4V-cr0_Qp7Mfo3Pnown4IaoFCB2OBn5CMounpjFhKc8U4iL-lBuLZZjEELceJ6y2jOllrbEBzD_bQs9_6w9HOU-jc49n0YUsucr_1oa_c1i3FDXNz?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/hHTKopNZLS3jIFpcx8neUnqdAJjB_OhYIuX60Zl5c0CP5plq463Wmh4fPDLylWv-kejzBIye9wzPrjy4naQji6_3rf4Hr8dr5HpjRS9IrIqb6i95HGo9ovO5IMLuGIjV4xPOrUBe244oBqHydvn-RfXF81NYAP1V-NvOaBv3aSt76zaCluwCTm-PvUJCfxnh?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/quOtCYnIP_u_6BdW13ExEI-0gq4GXJaSj1XZeAWyWfVv4qesJbmlq20PoKPu29yoj3n45vvFCp2phJbiDB9P5tT7jUE-75Mc3hRmdsAUNsJYu4gGZu_ogVqG9-B7nx6mmN81-bQ751lPBLbZI_8GhbkEHkLb3J3j7OOeHMfSG7X7DCADS1sJ_cBDXH6-ShIe?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/JyyFpeVn4x9K6pZQok2RVkE3Opn8NyYOt_CcfB6-12BtkWlift7sShVw6jgA3AOngeQqcIu11iJCBTrdiE8LTIpU0xqaVvxno0FKISBvVjHhyOBOzoggFLkA41OKdirQPq-G30xdllou7ZV9NayRk2Z4hajYN8aMMStYAJTFtt4fmrGqQkXNJNDzkekXn28V?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/4diB6uUaV_LdcsXlUIt0qyunkgFn3YBc4o9Ehukk_7YgpnfFKIoPCHAzpokfr9avOTKUvYGq26y40h6Pl5NNCOng1Hs4tLgS32q53kj095pv1aoxsdUfvzCgXxkI-QJulqZ2RGN-AkbH9rXIWf4ZX1ks1rXA_2UzwHTg0FnDfYnx5b5TxE719uQTxuX5A0UD?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/4H9Yt_tagUfuQctRnZED1zH1Fbo1JF-9y1nZUA_wFYjsj-7alg5iI5vXTUW8fDQIMBzVzlzkAeYFd4OOrqzW-ya7-d3gNPJq2QqYGO47u1ITHAoqIoBASV5JT1mVIAIxIvrIu1Lpxom7xRcUAR6YgpNd8eOvrT27PFtSgEtw2CxyLgfhSf5dqlaRz-hD0MDp?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/y0lJpM-WvzYVmsYyg6GeMBvSneoRsQlnB_-5w57ENzWemBNPpQoKWtpMTEM65ItW7byVko6lkSmiFEFrRs2F8Jhpa_eefZLmGm2wn-zpWyVreyc3sehEKVdelo6gsio-57dJMX9Hbng0VO7SSI-dNoKOST3OL_3YH7Vv7dju6Oc6IZ-3lbmiFRGLT_krXyC1?purpose=fullsize)

---

# 🚀 What You Learn From This

👉 Data structure কেমন
👉 Missing value আছে কিনা
👉 Relation: bill vs tip
👉 Gender/smoker effect

---

# 🧠 Final Workflow

1. Load data
2. Inspect (pandas)
3. Visualize (seaborn)

---

