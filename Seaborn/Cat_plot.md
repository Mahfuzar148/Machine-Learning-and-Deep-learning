
---

# 📘 `sns.catplot()` — Full Documentation

## 🧠 What is `catplot()`?

👉 `catplot()` = **Categorical data visualization-এর universal function**
👉 internally use করে:

* barplot
* boxplot
* violinplot
* stripplot
* swarmplot
* countplot
* pointplot

---

# 🔹 Basic Syntax

```python
sns.catplot(
    data=None,
    x=None, y=None,
    hue=None,
    kind="strip",
    col=None,
    row=None,
    palette=None,
    height=5,
    aspect=1
)
```

---

# 🔥 Supported Plot Types (`kind`)

## 📌 `kind` parameter → সবচেয়ে important

| kind       | plot type                   |
| ---------- | --------------------------- |
| `"strip"`  | scatter (categorical)       |
| `"swarm"`  | better scatter (no overlap) |
| `"box"`    | boxplot                     |
| `"violin"` | violin plot                 |
| `"boxen"`  | advanced boxplot            |
| `"bar"`    | barplot (mean)              |
| `"count"`  | countplot                   |
| `"point"`  | pointplot                   |

---

# 🔍 1. `kind="strip"`

```python
sns.catplot(x="day", y="total_bill", kind="strip", data=df)
```

👉 raw data points দেখায়

---

# 🔍 2. `kind="swarm"`

```python
sns.catplot(x="day", y="total_bill", kind="swarm", data=df)
```

👉 overlap avoid করে

---

# 🔍 3. `kind="box"`

```python
sns.catplot(x="day", y="total_bill", kind="box", data=df)
```

👉 median, quartiles দেখায়

---

# 🔍 4. `kind="violin"`

```python
sns.catplot(x="day", y="total_bill", kind="violin", data=df)
```

👉 distribution + density

---

# 🔍 5. `kind="bar"`

```python
sns.catplot(x="day", y="total_bill", kind="bar", data=df)
```

👉 mean value দেখায়

---

# 🔍 6. `kind="count"`

```python
sns.catplot(x="day", kind="count", data=df)
```

👉 count of rows

---

# 🔍 7. `kind="point"`

```python
sns.catplot(x="day", y="total_bill", kind="point", data=df)
```

👉 mean + confidence interval

---

# 🔍 8. `kind="boxen"`

```python
sns.catplot(x="day", y="total_bill", kind="boxen", data=df)
```

👉 large dataset-এর জন্য better boxplot

---

# 🧠 Important Parameters

---

## 🔹 1. `hue`

👉 color grouping

```python
hue="sex"
```

---

## 🔹 2. `col`

👉 multiple plots (column-wise)

```python
col="time"
```

---

## 🔹 3. `row`

👉 row-wise split

```python
row="smoker"
```

---

## 🔹 4. `palette`

👉 color theme

```python
palette="Set2"
```

---

## 🔹 5. `height` & `aspect`

👉 plot size control

```python
height=5, aspect=1.5
```

---

# 💻 Full Example

```python
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset("tips")

sns.catplot(
    x="day",
    y="total_bill",
    hue="sex",
    col="time",
    kind="box",
    palette="Set2",
    height=5,
    aspect=1.2,
    data=df
)

plt.show()
```

---

# 🖼️ Visualization Idea

![Image](https://images.openai.com/static-rsc-4/bQb89HFV_YHI1ZZH-bw8XRDE_1mY20UxcPMWw-HKeGrAJljMYnKEun9JG1pkLE3k9Pwe5NL5Ts_X7tXMHqp6QEmkqJpwLhd-c8BVvBmE4mAeixjgkQMFe4hOqUzgh04vRGGdpmwdhEreysQcC8c13kvFNN_nGzidZYv5ZgYbxHfuljRx8ZTmbJPBiWvS1Nc8?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/hv21GQcNn8mJ_gjiYkNR3MIOls1LAYDPt0CRpu_cqinGouIKXXKLadXePlmFCB5LWniSp-7UY07o3nglG-P8tTqZHnYQX4B-jVyOyyBcpITVpN987eTPO5MaKlZBg3AmWaeyfObuuit1AZij4fMK0vXKt-B5aquVWPTk7NE5rV7Ho4Gkprmfg5w_yy9rN5Yg?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/eN-fd1Z4l99iS1F35W_6svcrCQInlTpvlKG4Fxyb1p14SYeQCE-if-oKHw4JY4ObYrDQUziGUqNwjfOG-mNvu7YyexPTrja0_kPu6bhEsym4is4gKsKI9JnCjrDoUov2tFOq8Bf4OcyXIRkktL7Ef3ilUOJjEJmCRCAJqLGx1JXHTiYP15_ovlweQtgKleVl?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/2QCau7bWiCr5pecfblo_hRLvuMHHznQGkTzaFLbC3nrGe3Jn5jDRei1qT-4oo-yvtYxxkDNsUvcMSBtctmWu66zx_chN0yd6KA8g8I3VMWahW0-wOrunyBcn9zTRDxf-A06HV6EbILwAYQWiwkgEQk1s-v6q_xNjxh_e7Sn6Kmm3d0TWtYerZn9S5GYWI-RC?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/J23R3oKsepxK_Au9WsbLvQq7LWZSiik3wH_lkf7KdBETzOp2eFRCADecF4WSo_zmhub9_JHmWqJ0kaezkHmdNCpKzXiRLQ3ZFRYuuSpHizmIRiRfN39VWEDdPcZjUoZQuPsj_gAT-Su_5W-m094nItFMyHmVQ1Ra1NVkenRJpboaeQBuJNd_tBC1Tal9tsQR?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/6wC3UkBmzjPWuxeDiU_hko_3-cOi9KJkMPyfdHu59EjRPwxCNXW0BP0s3fU5XGInLs6W7CX8pRyJ9EKHDkyl28SqgHStG0yG3P4cG6WirazT5w91nGPkCj31_G74I2GIErV0fUZyFmbyCAImCLOJBd1Zjftbt9XDfXtxXKNb1LqK3mW5ESFQrZnrqF1j-kq_?purpose=fullsize)

---

# 🧠 catplot vs others

| Function | Use                     |
| -------- | ----------------------- |
| catplot  | categorical data        |
| relplot  | relation (scatter/line) |
| displot  | distribution            |

---

# 🔥 When to Use `catplot()`

👉 যখন:

* x = categorical
* y = numeric
* group comparison করতে চাও

---

# ⚠️ Important Notes

✔ এটা figure-level function
✔ multiple plots support করে
✔ internally axes-level functions use করে

---

# 🧠 Final Summary

👉 `catplot()` = categorical plots-এর master function ⭐
👉 `kind` change করে plot change হয়
👉 hue/col দিয়ে advanced visualization

---

# 🚀 Pro Tip

👉 Interview answer:

❓ “Which function handles most categorical plots?”
👉 Answer: **`sns.catplot()`**

---
