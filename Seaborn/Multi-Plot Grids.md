

---

# 🧠 Multi-Plot Grids কী?

👉 একাধিক plot একসাথে arrange করা (grid format-এ)

➡️ Use when:

* অনেক variable আছে
* relation compare করতে চাও
* dashboard-style visualization দরকার

---

# 📘 1. `FacetGrid`

---

## 📌 What it does:

👉 dataset কে ভাগ করে **multiple plots (grid)** বানায়

---

## 🔹 Syntax

```python
sns.FacetGrid(data, col=None, row=None, hue=None)
```

---

## 🔹 Example

```python
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset("tips")

g = sns.FacetGrid(df, col="time")
g.map(sns.scatterplot, "total_bill", "tip")

plt.show()
```

---

## 🧠 Use case:

👉 category-wise separate plots

---

# 📘 2. `PairGrid`

---

## 📌 What it does:

👉 multiple variables-এর relation manually control করে

---

## 🔹 Syntax

```python
sns.PairGrid(data)
```

---

## 🔹 Example

```python
g = sns.PairGrid(df)
g.map(sns.scatterplot)

plt.show()
```

---

## 🧠 Use case:

👉 custom pairwise visualization

---

# 📘 3. `pairplot()` ⭐

---

## 📌 What it does:

👉 সব numeric column-এর relation একসাথে দেখায়

👉 diagonal = distribution
👉 off-diagonal = scatter plot

---

## 🔹 Syntax

```python
sns.pairplot(data, hue=None)
```

---

## 💻 Example

```python
sns.pairplot(df, hue="sex")
plt.show()
```

---

## 🧠 কেন popular?

👉 এক লাইনে full dataset relation 😎

---

# 📘 4. `jointplot()`

---

## 📌 What it does:

👉 2টা variable-এর:

* relation (center plot)
* distribution (side plot)

---

## 🔹 Syntax

```python
sns.jointplot(data=None, x=None, y=None, kind="scatter")
```

---

## 🔹 Supported `kind`

| kind    | plot      |
| ------- | --------- |
| scatter | scatter   |
| kde     | density   |
| hist    | histogram |
| hex     | hexbin    |

---

## 💻 Example

```python
sns.jointplot(x="total_bill", y="tip", data=df, kind="scatter")
plt.show()
```

---

## 🔥 Advanced Example

```python
sns.jointplot(
    x="total_bill",
    y="tip",
    kind="kde",
    data=df
)
```

---

# 📊 Visual Idea

![Image](https://images.openai.com/static-rsc-4/9pBzZK_ouI9CMk9w6424n4bY80ksfKBIfN0x31680Jgy9bKTNhAWdAnXklN3ImrMroTVP8I-YA3bLeryuLsxNTzYnpbNQmprt87H4ak-BHu9Gq7MHZhKHP85F6Zqf2K6gQUuqxfW7EhMseUig7pILn1Glx8XlWSKJ90N1FNkg9vGZsApXLGYDmenJw0fW292?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/5GMYbXoI_rf_3fvLWse9FknzLNOAv_cBKXLgZZE662h5lkomkHhziIihp2uE5fcLqsenuu3_6YRepcyvmC2V1rSH_aJdwTaRlBDSNbvvVlg2YXW7nqZlVCanjLnCCIKuFhDSR3vMZB0bSvWtlJlhjol-67iCfBn0kgR3KF-DYnwh_k4ePZkpqLpgcsVLXuHW?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/BZCWWrKV1KSm2jUkc3GFRfP0kmERMQspML5712C9Zjubr4lMDguJ0vNrbOnqUynsBIAkOLaVzK4-mZOnMxZdNrODVa1dC4Mdi82zHSTuIan5tIBfzhXDBpUxeR5OkW2Qw95xuoQAYPDJS_oauzC_iXctvQLtXY48F6TMkbr-WKEC7MetJSBXrjt37jH_0ZSR?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/RfD1z1AXOTuwJWtJ-GxkoaH1fFRiYysTYuNh8Al79HRjtMpjNsxuDOF3-gudNYyjGK_MfP6AKYONgnKfBmrHwlC9uL8lea031m0Xpr20JWK-ARhPf6sf08cgRsiIXbN1Zc7jZKkhSixmYl-ZNpOmsnFRNQn_1hTqXXsV0b6v2yAcZOE0W-psrdtIfFb5rrpE?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/mfpxB5ru23cpH7qWs7KJqo2BZXKkElIlH622YWZKwagHfA1SVv1-qp20ty68QtCCr6q4KJr9UdemP3Mm50sejAQVeSW2YikKnwj7F9EvoISlqJP6LOZQjZIcPVx2H4_XxCSlE3_0CKf79Ux9hI91wxlY6M26hpFwyEFSwSpCNn52vl-fyVYva24IdtQWsTqr?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/Z-XrI93bRNAihmAH7dbJD8pyffdmUrSdf4Wyt58SUyo0AirWmWbDX9HajbmpHBtL-WEOZRiwXRU2V5961wLT6pWHoTjdaQ0nFDOejN1xVdd2R2Occ7wzzX6rG57D2rXn02Zp5MNb1Ko0Y2oWVuTfwBDzilQDG5bOSHh2U_3SIe03H-zgNHKAb3RGqy8kINu8?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/vYHyIFU3gWo16bJU61yuAVlKzuQNYkaUT3X6RA95gQxA7EI5raamMTikSHn2gegXti7U1RPw-aTWlf-nFOlSnbpdBO9zhciWIFZBvwBtgwyKAcPiDKwCfcRm_ST4QQkfpeigIWF25d-XzCfcYhIJi2g0-0HdEiaeuQ5Y12TpKL-gRJqiBKl4UIqk9kra271o?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/A-icnHEoA_WyDCYA8JyAtQQDxnG6zytPjEt744FSxw3ZF1TFQXeulnQiOvClCEx_ZfMTDa41ghMKf-WkxEDsruzm7qPz5DNDXUhdgLTpuueaFtuvIWfItbG-lrfS_3reUTJ6ImsHQmPWavLHTqxC7-sw2VKNWc6DaE16ygLLjFAjVOwVrf1svlWAwimm4tx4?purpose=fullsize)

---

# 🧠 Comparison Table

| Function  | কাজ                       |
| --------- | ------------------------- |
| FacetGrid | custom grid               |
| PairGrid  | advanced control          |
| pairplot  | auto relation ⭐           |
| jointplot | 2-variable + distribution |

---

# 🔥 When to Use Which?

👉 Quick guide:

* simple multi-variable → `pairplot()` ⭐
* custom grid → `FacetGrid`
* advanced control → `PairGrid`
* 2 variable deep analysis → `jointplot()`

---

# 🧠 Real ML Use

👉 EDA step:

```python
sns.pairplot(df)
```

➡️ feature relation বুঝা

---

# 🧠 Final Summary

✔ Multi-plot grids = multiple visualization
✔ pairplot = most powerful quick tool ⭐
✔ jointplot = deep 2-variable analysis

---

# 🚀 Pro Tip

👉 Interview:

❓ “How to visualize all feature relationships quickly?”
👉 Answer: **pairplot()**

---

