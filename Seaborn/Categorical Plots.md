
---

# 🧠 Categorical Plots কী?

👉 যখন:

* X = categorical (day, gender, smoker…)
* Y = numeric (price, tip, score…)

👉 তখন এই plots use হয়

---

# 📘 1. `barplot()` — Mean Comparison

## 📌 What it does:

👉 প্রতিটা category-এর **average (mean)** দেখায়

---

## 🔹 Syntax

```python
sns.barplot(data=None, x=None, y=None, hue=None, palette=None)
```

---

## 🔹 Important Params

* `x` → category
* `y` → numeric
* `hue` → grouping
* `palette` → color theme

---

## 💻 Example

```python
sns.barplot(x="day", y="total_bill", data=df)
```

👉 কোন দিনে average bill বেশি?

---

# 📘 2. `countplot()` — Frequency Count

## 📌 What it does:

👉 category-এ কয়টা data আছে count করে

---

## 🔹 Syntax

```python
sns.countplot(data=None, x=None, hue=None)
```

---

## 💻 Example

```python
sns.countplot(x="day", data=df)
```

👉 কোন দিনে কত customer?

---

# 📘 3. `boxplot()` — Statistical Summary ⭐

## 📌 What it does:

👉 median + quartiles + outliers দেখায়

---

## 🔹 Syntax

```python
sns.boxplot(data=None, x=None, y=None, hue=None)
```

---

## 💻 Example

```python
sns.boxplot(x="day", y="total_bill", data=df)
```

👉 distribution + outliers বুঝা যায়

---

# 📘 4. `violinplot()` — Distribution + Shape

## 📌 What it does:

👉 boxplot + KDE (shape) combine

---

## 🔹 Syntax

```python
sns.violinplot(data=None, x=None, y=None, hue=None)
```

---

## 💻 Example

```python
sns.violinplot(x="day", y="total_bill", data=df)
```

---

# 📘 5. `stripplot()` — Raw Data Points

## 📌 What it does:

👉 সব data point scatter করে দেখায়

---

## 🔹 Syntax

```python
sns.stripplot(data=None, x=None, y=None, jitter=True)
```

---

## 💻 Example

```python
sns.stripplot(x="day", y="total_bill", data=df)
```

---

# 📘 6. `swarmplot()` — Non-overlapping Scatter

## 📌 What it does:

👉 stripplot-এর better version (overlap avoid)

---

## 🔹 Syntax

```python
sns.swarmplot(data=None, x=None, y=None)
```

---

## 💻 Example

```python
sns.swarmplot(x="day", y="total_bill", data=df)
```

---

# 📘 7. `catplot()` ⭐ (High-Level Function)

## 📌 What it does:

👉 সব categorical plot এক জায়গা থেকে control করে

---

## 🔹 Syntax

```python
sns.catplot(
    data=None,
    x=None, y=None,
    kind="strip",
    hue=None,
    col=None,
    row=None
)
```

---

## 🔹 Supported `kind`

| kind   | plot         |
| ------ | ------------ |
| strip  | stripplot    |
| swarm  | swarmplot    |
| box    | boxplot      |
| violin | violinplot   |
| boxen  | advanced box |
| bar    | barplot      |
| count  | countplot    |
| point  | pointplot    |

---

## 💻 Example

```python
sns.catplot(
    x="day",
    y="total_bill",
    kind="box",
    hue="sex",
    col="time",
    data=df
)
```

---

# 📊 Visual Idea

![Image](https://images.openai.com/static-rsc-4/gZeTHs45yDr6mAb29v0r3yjn39-lUSOsUgaLk9ufAde2C9aVnkAErDwLULVsGu-HcxjNZoS-dfucC3DkQpd_goEzP1XALME3R4sH_3L4ELhqnJkIDm970IyasBjG5pN8JS0k622tgKqM30SgiZ31CyDYT-TVTa8WuxrPqDCHmHP6Iret9j11bkfCBsVZMyvK?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/FGWYMlurHneJJXfjZkumRF2bGdF4EZwkYy--1YI3IS_PfyVhXJ61kvdhPzsTlj-LIF6vebzqD6TWESr89mlTKI83JmVV4w_2TQV3rgFlbrHwfXluSyD0G9p0HihVwagFYknFWqz2MCqitaoG3jxCk9kzuFUt1tlNoIjghM8-HIlFlYiKfukeqTWdZNYCot7U?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/CxNBK6bbHYX53-6rptfDYlkCdBQNnnJA0e532VUENgubBiXZygEXMocJ08cAJ20L66Gu7mJ36SIAq_KQusbEJH4h2xUyMzC40MZI1ZwPlqL29ZUoWjR6PkGTzCOK1uKxyOI_m94EYS6IuFgH3WMPFHlbd5zn__VIHuf5JFcMTR6M1KNAbhBoraJt9zvg-7Gp?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/AvXjI56LntEvaHJeOfp_NAfwZaMpcVdpzie4FWoaOh1pyI1NGieQgK2QaVKTrdtXshRkY6D1872rc3iqG9-jsNo70YdkCMtLVsk4qcG38kvx3hQuVSuQ-ba9OM-CNepoZhvanZdtE-HLeVE7aOZVHdP-S4zBN-erAwHj_DNGI_m26BxOTWXJER-U9dxOr9mJ?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/bQb89HFV_YHI1ZZH-bw8XRDE_1mY20UxcPMWw-HKeGrAJljMYnKEun9JG1pkLE3k9Pwe5NL5Ts_X7tXMHqp6QEmkqJpwLhd-c8BVvBmE4mAeixjgkQMFe4hOqUzgh04vRGGdpmwdhEreysQcC8c13kvFNN_nGzidZYv5ZgYbxHfuljRx8ZTmbJPBiWvS1Nc8?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/BqjqcSWYXfdD-bubNfF9IMGjfB7fM267F_hS0N0WG1IL7Smpp_eWQSNOKmQgP9_H8GL0hnmuf54AoQ25gQ_CySs7pD0h6oKNrHKwbYQ8m_4uRJV_CR0I4PliSuGl8X1jIo__ZqW4nQOeR-72Jz4T9nLOlE8iRo2B7qL4K61YZE_lOkchGh0RY9_OBa8H6cUD?purpose=fullsize)

---

# 🧠 When to Use Which?

| Plot       | Use                 |
| ---------- | ------------------- |
| barplot    | average compare     |
| countplot  | count/frequency     |
| boxplot    | stats + outliers ⭐  |
| violinplot | distribution shape  |
| stripplot  | raw data            |
| swarmplot  | clean scatter       |
| catplot    | advanced/multiple ⭐ |

---

# 🔥 Common Pattern (Very Important)

👉 90% time:

```python
sns.boxplot(x="category", y="value", data=df)
```

👉 OR

```python
sns.catplot(kind="box")
```

---

# 🧠 Final Summary

✔ categorical plot = category vs numeric
✔ 6 basic + 1 advanced function
✔ `catplot()` = master controller ⭐

---

# 🚀 Pro Tip

👉 Interview answer:

❓ “Best plot for outliers?”
👉 `boxplot()`

❓ “Best for full categorical analysis?”
👉 `catplot()`

---

