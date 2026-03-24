---

# 🧠 Matrix Plot কী?

👉 Data যদি **matrix / table (row × column)** আকারে থাকে
👉 তখন color দিয়ে value দেখানো হয়

➡️ Example:

* correlation matrix
* similarity matrix

---

# 📘 1. `heatmap()` ⭐

---

## 📌 `heatmap()` কী?

👉 **matrix-এর প্রতিটা value color দিয়ে দেখায়**

👉 Most common use:

> **Correlation matrix visualize করা**

---

## 🔍 Heatmap কী বোঝায়?

👉 Color intensity = value

* Dark color → high value
* Light color → low value

👉 correlation ক্ষেত্রে:

* +1 → strong positive
* 0 → no relation
* -1 → negative relation

---

## 🔹 Syntax

```python
sns.heatmap(
    data,
    annot=False,
    cmap=None,
    linewidths=0
)
```

---

## 🔹 Important Parameters

| Parameter  | কাজ                |
| ---------- | ------------------ |
| data       | matrix / dataframe |
| annot      | number show করবে   |
| cmap       | color theme        |
| linewidths | grid line          |

---

## 💻 Example (Correlation Heatmap)

```python
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset("tips")

# correlation matrix
corr = df.corr(numeric_only=True)

# heatmap
sns.heatmap(corr, annot=True, cmap="coolwarm")

plt.title("Correlation Heatmap")
plt.show()
```

---

## 🧠 কখন `heatmap()` use করবো?

👉 Use when:

* feature relation বুঝতে (ML-এ খুব important)
* correlation check
* data pattern detect

---

# 📘 2. `clustermap()`

---

## 📌 `clustermap()` কী?

👉 **heatmap + clustering (grouping)**

👉 Similar rows/columns → একসাথে group করে

---

## 🔍 Clustermap কী বোঝায়?

👉 শুধু value না,
👉 data-এর **similarity pattern** দেখায়

➡️ dendrogram (tree) দিয়ে group দেখায়

---

## 🔹 Syntax

```python
sns.clustermap(
    data,
    cmap=None,
    standard_scale=None
)
```

---

## 🔹 Important Parameters

| Parameter      | কাজ       |
| -------------- | --------- |
| data           | matrix    |
| cmap           | color     |
| standard_scale | normalize |

---

## 💻 Example

```python
import seaborn as sns

df = sns.load_dataset("tips")

corr = df.corr(numeric_only=True)

sns.clustermap(corr, cmap="coolwarm")

plt.show()
```

---

## 🧠 কখন `clustermap()` use করবো?

👉 Use when:

* similar features/group খুঁজতে
* clustering analysis
* pattern grouping

---

# 📊 Visual Idea

![Image](https://images.openai.com/static-rsc-4/n4547_92SEzZI4lF4mRmTYjIYoiMr-qcxMRCp5QB4sjYJ4GoG48b0GVkk9hDZWJBP6S1opjOR1hQ6DzbDPeVEpPu9QvJcQZ7oUXfLtTL_jCuzl4y08I64zNnf0ZS1162DGsVb1jTK4SIq6CNkrrANXlGcfU85Nh4yLqoQfMqHNk1Dwfn9IjzbOYtDAQv-KOk?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/hYmkE7fvuTtGne_Da2EgMONRo_njFvAdSLNF58oYDndmSYy7FcqxhXQAh_gsVu0b9AfgV3T0MiaRnt4oifb11YZSU6SDMPiZjRH8VIc6qLoi0Wo6DDpZk6WIGJEsKRaVatE1w1Se6gw4wY_ClxEaoaMNIM960RTbJoI_7-0y-bIAF9r_lojhtPRU8_aAYpv3?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/4AyzSvwd4VXH5a2D2fQInyLSxulxHaOjwbF7WaR3pf7WAIbY52fNhD5cu_WfaRg7BkCCbvlEtln6QG-KdUtjb9zRqQbZrKVa1enrQZJVFGTNUGBoicC50coV3lEgpyfcL8zMwL2Kmce0y1y7owk1RFV_-Vsbzy_8-G52ayIHsVoVpRcy42R3qb3vjRdK24Kl?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/f0WaBuMFt56QqzoSrhLm5byu3kwvCuyieebXFvSkImcS-inlsKxDB4IIhSS7YedBeLfnXH618nM5G4kdJXnsXXKOv3jJNSMBAv4Ve7-dqGlDwTrPwICNH-YDsHxHMik4zMaG2MNr3ktb-aRSPBB35DmtiRkN3wyP9KNn9VxRfLJqm3Awovth3qJiGOkDBfW5?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/9VvkvdrdUu6SA8aZUHJyqxZUcGw3iHHZxk5KII_EwcfIl79AmR0R6zlcE8gsJHrXVz559a8ZypSsv0rxaZJRw1XDa2jNq1vCZhza8RcNCn_AI-bFzcMpcS5ds1AXV8RNHcVjjuMIfEFma5f5c9eoL483y29ITZklHVw1YYZTRKVomPisc8YmROFFKb7JkeGX?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/KdcppZGhRkb6ZCuk3olh_0SWqa3S47Jxhl7HlqJ-zq1TkNqU_dgFAuMrF7ZABVs6ZW6sXuSOPhAU3BspGR9UXAqi227iJolryE9E6Kw7Je8x3Misz0wUeHLNKKm-1YChUvUneo2LkW0bLTxtEVtLSvG6U93lJOsPPS4VoT7y04_i7gyx261CrSknHsnzyo1H?purpose=fullsize)

---

# 🧠 Heatmap vs Clustermap

| Feature        | heatmap     | clustermap |
| -------------- | ----------- | ---------- |
| Shows values   | ✅           | ✅          |
| Shows grouping | ❌           | ✅          |
| Dendrogram     | ❌           | ✅          |
| Use case       | correlation | clustering |

---

# 🔥 Simple Understanding

👉 **heatmap** = “value দেখাও”
👉 **clustermap** = “similar group খুঁজো”

---

# 🧠 Real ML Use Case

👉 Feature selection:

```python
sns.heatmap(df.corr(), annot=True)
```

➡️ highly correlated features remove করা যায়

---

# 🧠 Final Summary

✔ heatmap = correlation visualization ⭐
✔ clustermap = pattern + clustering

---

# 🚀 Pro Tip

👉 Interview question:

❓ “Difference between heatmap and clustermap?”
👉 Answer:

* heatmap → value display
* clustermap → grouping + clustering

---


