

---

# 📘 `sns.relplot()` — Full Documentation

---

# 🧠 What is `relplot()`?

👉 `relplot()` = **relationship (relation) visualize করার universal function**

👉 এটা internally use করে:

* `scatterplot()`
* `lineplot()`

---

# 🔹 Basic Syntax

```python
sns.relplot(
    data=None,
    x=None, y=None,
    kind="scatter",   # or "line"
    hue=None,
    style=None,
    size=None,
    col=None,
    row=None,
    col_wrap=None,
    palette=None,
    height=5,
    aspect=1
)
```

---

# 🔥 Supported Plot Types (`kind`)

## 📌 `kind` parameter → সবচেয়ে important

| kind        | plot type    |
| ----------- | ------------ |
| `"scatter"` | Scatter plot |
| `"line"`    | Line plot    |

---

# 🔍 1. `kind="scatter"` (Default)

```python
sns.relplot(x="total_bill", y="tip", kind="scatter", data=df)
```

👉 2টা numerical variable-এর relation দেখায়

---

# 🔍 2. `kind="line"`

```python
sns.relplot(x="total_bill", y="tip", kind="line", data=df)
```

👉 trend (increase/decrease) দেখায়

---

# 🧠 Important Parameters

---

## 🔹 1. `hue` — Color grouping 🎨

```python
hue="sex"
```

👉 প্রতিটা class → different color

---

## 🔹 2. `style` — Shape / Line style 🔺

```python
style="smoker"
```

👉 different marker shape (scatter)
👉 different line style (line plot)

---

## 🔹 3. `size` — Bubble size 🔵

```python
size="size"
```

👉 value অনুযায়ী dot size change হয়

---

## 🔹 4. `palette` — Color theme 🌈

```python
palette="Set2"
```

👉 hue-এর color style control করে

---

## 🔹 5. `col` — Column-wise split

```python
col="time"
```

👉 multiple plots (Lunch / Dinner আলাদা)

---

## 🔹 6. `row` — Row-wise split

```python
row="sex"
```

---

## 🔹 7. `col_wrap`

```python
col_wrap=2
```

👉 grid wrap control

---

## 🔹 8. `height` & `aspect`

```python
height=5, aspect=1.5
```

👉 plot size control

---

## 🔹 9. `facet_kws` (advanced)

👉 FacetGrid customize করতে

---

# 💻 Full Example (All Features)

```python
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset("tips")

sns.relplot(
    x="total_bill",
    y="tip",
    kind="scatter",
    hue="sex",
    style="smoker",
    size="size",
    col="time",
    row="day",
    palette="Set2",
    height=4,
    aspect=1.2,
    data=df
)

plt.show()
```

---

# 🖼️ Visualization Idea

![Image](https://images.openai.com/static-rsc-4/cGHiFik8rt7idT-pwwvsvDIvpNZmrIg-SKAFcx8KlJbxR4LqFI_VSETlItMAfq1UMCPQxfztiGXxArqtncXxazWsxHyYibqTUd21GRxEXmdWLmmkn04RuOCXCrbnTjQRzLurWHg4CAi61-Fn0Y8No4HUZQXQj-XxDAm8l31cKXp3cNsKUZSo_BgoB9G2AcTp?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/xI6gJ0SMcSfdUB24DwQiDQBoHDn4KekkfA8K_lb7a5JmwWA1yCaOYBcb0Qpo3xwqsKyApov8_iTWuqw_AkGg8WqcE95CliwJZpDgvVHIKV4pXLn1mpuxyCEOxCk5krIDG4H9omtoVLAOf6izO0Kdy0YdC6fO_d83x0Cfna12_hj45YVblnDXCBsIK3-F8G9H?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/uJWpEJvWMM8RlDkWAO7yUS4ZuBNd1zSsfcF2UMahW0U0aueaAvur7nnIaov-nZk5vsQmT8p5WKh_lRUp9yApId__hnnsYvDAo6V304nsLtIYRiTmacOPIFzTHPfYIWSKmkMuEU0RwVmPLx6qkE1jsnbekGfvKM1Uhnm0kNE8eQQPN-FndczwT0kDUqCs2Oim?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/GOFILkma2Tc6cH_LjLT1WPo6ZiV55JI5EnMLGRMvLkCiT57e9IFbBwFDrRMDFUelaVFYeAHOycYERD1ZmyfsTqC3AB9yKjPxdJg9tbcLPtt1NS4pAjeTowtgYX8ChXPFHrzm2iQyjyizxPlUNDIul6v_kl63EfSYziD0QC859BvYVUit9I-NsXvtDjrgPLSZ?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/LIHL0tec8kCLGuHCixWoEIrb1b81ieSw23izaabZvjrW5xTpOVx5y2D-Et4H5PPLMsgyHDJS9d0QZX1AnlaEwcdcuKDYAdi-FO57aPhozFN5I0UrAu1la9-J8Ymw2N7T6tGYgfeC0vRFib4aXprozKHY-HGUpnxJ_O0fdVfGK8RTG8TonyRrhk21x2FB3hFI?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/6J4NbymZyR_npDOzZjeJuMT36QVslurn5I1OCu3g0FGL2ODUGHUK2oxuQKLY2EmU8dpEFZ-fPrdjBieTgoegsEnNqNENq0FOpQG9XWVNcQP15AKDk_8scF99Z3gmq6Xxei089XiHP-JidNpy3JaamluL1s10os8uj5f7t0UIEhzt-emoR0cJukloqDE7OPM7?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/gs9vE_kjyu5H3ogydG3N3k6mKID6k6ZqPE53AFn1XErEpT0kj9q2d4TJpTcF95my4Fjny2YpaBlXxfTgyZSG_lTjXhHVIshZoEOuXJ3OlR6jxor--DYCCj5Qjma3QTPUqby2fDQs1OjvDlzPLMUpzl5umdVfF26uIl4HqjUN_-dHgFBcmzVaiRfAaoitR5SV?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/OKqppA03g3FmO5J3K7WbtoAJHMmzoQI_0MqI4VJJefc4hu0KPy1hsfMLWJwxeYvY1JN1ytvgSECkuSdLjEsPpkxQa-aUJmaOiv0bn_yXa4yFDtajVbGIhMSHjzi5ZA2cSZRqw62Okoq_vFr75mhrYI4Iz-u9lh6_9vc939UK4RZNZZ-KKTVTtMAFvmnG0vmI?purpose=fullsize)

---

# 🧠 relplot vs scatterplot

| Feature        | scatterplot() | relplot()    |
| -------------- | ------------- | ------------ |
| Type           | axes-level    | figure-level |
| Multiple plots | ❌             | ✅            |
| Faceting       | ❌             | ✅            |
| Power          | medium        | high         |

---

# 🔥 When to Use `relplot()`

👉 যখন:

* multiple plots দরকার
* category-wise comparison
* dashboard style visualization

---

# ⚠️ Important Notes

✔ এটা **figure-level function**
✔ internally `FacetGrid` use করে
✔ performance একটু slower হতে পারে

---

# 🧠 Final Summary

👉 `relplot()` = relationship visualization master ⭐
👉 scatter + line দুইটাই করে
👉 multiple plots + grouping support

---

# 🚀 Pro Tip

👉 Quick rule:

* simple → `scatterplot()`
* advanced → `relplot()`

---

