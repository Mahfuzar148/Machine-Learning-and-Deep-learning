
---

# 🧠 1. `histplot()` — Histogram

## 📌 What it does:

👉 Data কে **bins (bar)**-এ ভাগ করে frequency/count দেখায়

---

## 🔹 Basic Syntax

```python
sns.histplot(data=None, x=None, bins=None, kde=False, hue=None)
```

---

## 🔹 Important Parameters

| Parameter | কাজ                   |
| --------- | --------------------- |
| x         | কোন column plot হবে   |
| bins      | কতগুলো bar হবে        |
| kde       | smooth curve add করবে |
| hue       | group-wise color      |

---

## 💻 Example

```python
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset("tips")

# Basic histogram
sns.histplot(x="total_bill", data=df)

plt.show()
```

---

## 🔥 Advanced Example

```python
sns.histplot(
    x="total_bill",
    bins=20,
    kde=True,
    hue="sex",
    data=df
)

plt.show()
```

---

## 🧠 When to use:

✔ data distribution বুঝতে
✔ skewness detect করতে
✔ outlier hint পেতে

---

# 🧠 2. `kdeplot()` — Density Plot

## 📌 What it does:

👉 data-এর **smooth probability curve** বানায়

---

## 🔹 Syntax

```python
sns.kdeplot(data=None, x=None, hue=None, fill=False)
```

---

## 🔹 Important Parameters

| Parameter | কাজ              |
| --------- | ---------------- |
| x         | data column      |
| hue       | group comparison |
| fill      | area fill        |

---

## 💻 Example

```python
sns.kdeplot(x="total_bill", data=df)
plt.show()
```

---

## 🔥 Advanced Example

```python
sns.kdeplot(
    x="total_bill",
    hue="sex",
    fill=True,
    data=df
)

plt.show()
```

---

## 🧠 When to use:

✔ smooth distribution দেখতে
✔ histogram-এর alternative

---

# 🧠 3. `displot()` — Advanced Distribution Plot ⭐

## 📌 What it does:

👉 histogram + kde + ecdf → সব এক জায়গায়
👉 multiple plots support করে

---

## 🔹 Syntax

```python
sns.displot(
    data=None,
    x=None,
    kind="hist",
    hue=None,
    col=None,
    row=None
)
```

---

## 🔹 Supported `kind`

| kind | plot       |
| ---- | ---------- |
| hist | histogram  |
| kde  | density    |
| ecdf | cumulative |

---

## 💻 Example

```python
sns.displot(x="total_bill", kind="hist", data=df)
```

---

## 🔥 Advanced Example

```python
sns.displot(
    x="total_bill",
    hue="sex",
    kind="hist",
    kde=True,
    col="time",
    data=df
)
```

---

## 🧠 When to use:

✔ multiple distribution compare
✔ dashboard-style visualization

---

# 🧠 4. `ecdfplot()` — Cumulative Distribution

## 📌 What it does:

👉 cumulative % দেখায়
👉 “এই value পর্যন্ত কত data আছে?”

---

## 🔹 Syntax

```python
sns.ecdfplot(data=None, x=None, hue=None)
```

---

## 💻 Example

```python
sns.ecdfplot(x="total_bill", data=df)
plt.show()
```

---

## 🔥 Advanced Example

```python
sns.ecdfplot(
    x="total_bill",
    hue="sex",
    data=df
)

plt.show()
```

---

## 🧠 When to use:

✔ percentile বুঝতে
✔ distribution compare করতে

---

# 📊 Visual Idea

![Image](https://images.openai.com/static-rsc-4/CxNBK6bbHYX53-6rptfDYlkCdBQNnnJA0e532VUENgubBiXZygEXMocJ08cAJ20L66Gu7mJ36SIAq_KQusbEJH4h2xUyMzC40MZI1ZwPlqL29ZUoWjR6PkGTzCOK1uKxyOI_m94EYS6IuFgH3WMPFHlbd5zn__VIHuf5JFcMTR6M1KNAbhBoraJt9zvg-7Gp?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/hSCOjlQNDhMUMH3pkbw61NlM4veSfO3DaLoV_Gkij2olohe3zO1sWM4QNITh12dsj3bqKIGo_K2jKirLHhM5sXyGsD-Grti1n1BaNbFQ865udTwp4LqqCRBRV1jafeUhBSJ54bMjWL8_axk2EmKyr_aeKpD9SkmurQap8QJzEIgSVhWgbbCQfZzkavQ1pKvK?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/O5Llw0GMQavzfRyKFcuURiVz3fLdDzE14vIgBvG9wACmeJRzoy1sv30_Q18vFRw503s6fYAGH4Tx5PEkdkNyJHOh76-mKImVG-0fVPMmUFjGY853k4LDA9wNLsl70In1oAC8T7LPvTroUnmNGsVjZoVkqY6vzDvLCztficCdmuPFGgvKivJRUW3ef1t4Y07W?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/vJbKBY0RH5Tr0z6K8W3zQD_pzHG_Iadp4vclhnhWB999O4prSvLBWjhbhRnIG_XUdxGJbzHEeK7g39rvwFuTDTMWEo6GuklsZy2gaAi3AvATs-Hv-cTRjR2wAWP9G_xdDrXDlgocX2PPFtAftWV_FQDxhhkMe4M54F6YzfvldXLqOB4Gy8kwXSdwMDd2uNf8?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/_dqrh8qFAV1Ykx22Fzbxy1FGKrgesYDQlA4zepE3hBxbY1SAoDyDB3OAQjaDKk96C7pX_DkVtRqtUTqgnyE5FThBiJdjHO-l3TTj2_-txenbFB4pX2ybK33XnWZahY3--xh65hhEuN_TnSgwHM-7NoX42P3yWmxP_CNMJTw2Acwm7MlMobbcTvE3AimRyDCG?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/yQNzDITARDBQMWPu0ojMZr26IYUddpbJH8yobZAwUTgGmyiw3E9Qc57DVC4EPLLJ6gL6ce-YTdpU33dkyNjThI5DAJoJqMmh5DDq_GRrmX5iesVgfE_WKxSGaVdvUdOrUGuSIs6EykaLNB2hxsGlxIZgV-jnnwSAV3w2t_g6JedX5kxRpztAwbR6oYQvz6cU?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/pJ4Ped75NPeVX2EXQZygLQRr4r_9_pnM4J9AnZhpqlMMYUEuIVj-JXCRJx3KSvIUcbSBTdMUQroFJiZ7m6J9RoZO9ZXo56eonJuhz_XJSyzqdCkCCx2WPh8KKmWtX7TzZ8SDfBwphn9XjG6aUgvhOvQ6AEwcQsG1grt5_cSdfb9PngCRoQZEMFXuP4a9-p8j?purpose=fullsize)

---

# 🧠 Final Comparison

| Function | Type       | Best Use            |
| -------- | ---------- | ------------------- |
| histplot | bars       | frequency           |
| kdeplot  | curve      | smooth distribution |
| displot  | all-in-one | advanced analysis   |
| ecdfplot | cumulative | percentile          |

---

# 🔥 Pro Insight

👉 Real-world use:

* quick view → `histplot()`
* smooth analysis → `kdeplot()`
* multiple plots → `displot()`
* percentile → `ecdfplot()`

---

# 🧠 Final Summary

✔ Distribution plots = data spread বুঝা
✔ ৪টা function = complete coverage

---

