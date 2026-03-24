
---

# 📘 `sns.displot()` — Full Documentation

---

# 🧠 What is `displot()`?

👉 `displot()` = **distribution plots-এর universal function**

👉 এটা internally use করে:

* histogram
* KDE (density plot)
* ECDF

---

# 🔹 Basic Syntax

```python
sns.displot(
    data=None,
    x=None, y=None,
    kind="hist",
    hue=None,
    col=None,
    row=None,
    bins=None,
    kde=False,
    palette=None,
    height=5,
    aspect=1
)
```

---

# 🔥 Supported Plot Types (`kind`)

## 📌 `kind` parameter → সবচেয়ে important

| kind     | plot type               |
| -------- | ----------------------- |
| `"hist"` | Histogram               |
| `"kde"`  | Density plot            |
| `"ecdf"` | Cumulative distribution |

---

# 🔍 1. `kind="hist"` (Default)

```python
sns.displot(x="total_bill", kind="hist", data=df)
```

👉 frequency (count) দেখায়
👉 data distribution বুঝতে best

---

# 🔍 2. `kind="kde"`

```python
sns.displot(x="total_bill", kind="kde", data=df)
```

👉 smooth curve
👉 probability density দেখায়

---

# 🔍 3. `kind="ecdf"`

```python
sns.displot(x="total_bill", kind="ecdf", data=df)
```

👉 cumulative distribution
👉 % of data ≤ value

---

# 🧠 Important Parameters

---

## 🔹 1. `hue` — Color grouping

```python
sns.displot(x="total_bill", hue="sex", data=df)
```

👉 Male vs Female distribution compare

---

## 🔹 2. `bins` — Histogram bars

```python
sns.displot(x="total_bill", bins=20, data=df)
```

👉 bar count control

---

## 🔹 3. `kde=True` (hist-এর সাথে)

```python
sns.displot(x="total_bill", kde=True, data=df)
```

👉 histogram + smooth curve

---

## 🔹 4. `col` — Multiple plots

```python
sns.displot(x="total_bill", col="time", data=df)
```

👉 Lunch & Dinner আলাদা plot

---

## 🔹 5. `row` — Row split

```python
sns.displot(x="total_bill", row="sex", data=df)
```

---

## 🔹 6. `palette` — Color theme

```python
sns.displot(x="total_bill", hue="day", palette="Set2", data=df)
```

---

## 🔹 7. `height` & `aspect`

```python
sns.displot(x="total_bill", height=5, aspect=1.5, data=df)
```

---

# 💻 Full Example

```python
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset("tips")

sns.displot(
    x="total_bill",
    hue="sex",
    kind="hist",
    kde=True,
    col="time",
    palette="Set2",
    bins=20,
    height=5,
    aspect=1.2,
    data=df
)

plt.show()
```

---

# 🖼️ Visualization Idea

![Image](https://images.openai.com/static-rsc-4/vJbKBY0RH5Tr0z6K8W3zQD_pzHG_Iadp4vclhnhWB999O4prSvLBWjhbhRnIG_XUdxGJbzHEeK7g39rvwFuTDTMWEo6GuklsZy2gaAi3AvATs-Hv-cTRjR2wAWP9G_xdDrXDlgocX2PPFtAftWV_FQDxhhkMe4M54F6YzfvldXLqOB4Gy8kwXSdwMDd2uNf8?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/O5Llw0GMQavzfRyKFcuURiVz3fLdDzE14vIgBvG9wACmeJRzoy1sv30_Q18vFRw503s6fYAGH4Tx5PEkdkNyJHOh76-mKImVG-0fVPMmUFjGY853k4LDA9wNLsl70In1oAC8T7LPvTroUnmNGsVjZoVkqY6vzDvLCztficCdmuPFGgvKivJRUW3ef1t4Y07W?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/s9vwKriGp_1HCy3j_UakfL7c_MvbtC01gps3inAWH9mfTz6P6EOnAPsqLBT02K4giQuT_cYOvuCiSh9sPGTKGy44OTap9hEX6AAOayU-53_SHTMDUW5yjADrowJEEMzgdld431h7wJrfXSXziK2nf3COYhA4cWSPBTO40yH0sBUtjCe2K47enf5rsFFCmtM2?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/bCZuacCWojhiF1He7Pc0i3E4iBaU0CfHL9CY8brSqcAcGzOpeW8HzcsV9x3Dp6iWlvvXv2bjOCChtIRR7z0s49n_k-8gg8ql-PMleARrbYJpvGzU6OeecFoo3oX4YiQeG9TbDm20fnxrbyryA5zbIWIB8If1o7Zvv58djbpgPWqw_9_KehIr5xjb3uv_Q3jl?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/gOtR5cRqS8Zd_sGHK36rx8Xg44wg64gbUvG2eW8L_FfGRdlbTDF7MJ0pVO7a0TXAf2Qia--TSwm7zowSs3Nd4hMcWu0SyukzahTLm9pn1pFSVJcYs545xLGn5nsHrLgSUav7RUBg6iZrtArfdj6qnKfxLkW5D_egLrWbQzelYFFyTCbCevxbS-Vio4uXRO4G?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/H8vACV8eTvPuB8Jkaa3jAHGCCowDe0GSRHRrcthPaBwZVJTlhvvKQ3yVSYBr08RH-Y-sXnQq8yn2tBp-8daxc-Tu6wBhegpGIg0VQ6OQ5a3runftj7neJxUtv58j5HVDgpjXQoimnK1KLmWlxjwidRY6VCtxHSl21dNkAk9DDs_PMZYtKUaQ4z9f7biC5l41?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/Ucgd3J81Z4eCrOAuRIcdLGZP99T7EKqQF2z4M3B_7I2Apij64rAtpZUHBt1y3E5cXdXDkWgPNZdtR731qWSjjzLIhrp0hDGbggnwvOSNdcTNRIT0BDT_YDCGXwNZYQX1colsxlHy2Ci6G2J3H6bwQnNdFnukV9kuwsFzmF9VaRBW67Vfm2pEU3HRGVahhD15?purpose=fullsize)

---

# 🧠 displot vs others

| Function | Use          |
| -------- | ------------ |
| displot  | distribution |
| relplot  | relation     |
| catplot  | categorical  |

---

# 🔥 When to Use `displot()`

👉 যখন:

* data distribution বুঝতে চাও
* skewness check করতে চাও
* outlier detect করতে চাও

---

# ⚠️ Important Notes

✔ figure-level function
✔ multiple plots support
✔ large dataset-এ useful

---

# 🧠 Final Summary

👉 `displot()` = distribution master function ⭐
👉 histogram + KDE + ECDF সব করে
👉 hue/col দিয়ে advanced analysis

---

# 🚀 Pro Tip

👉 Interview question:

❓ “Which function is used for distribution plots?”
👉 Answer: **`sns.displot()`**

---


