
---

# 🧠 Regression Plot কী?

👉 X (input) → Y (output) এর relation দেখায়
👉 সাথে একটা **best-fit line (regression line)** draw করে

➡️ Example:

* house area → price
* study time → marks

---

# 📘 1. `regplot()` ⭐

---

## 📌 What it does:

👉 scatter plot + regression line (best fit line)

---

## 🔹 Syntax

```python
sns.regplot(data=None, x=None, y=None)
```

---

## 🔹 Important Parameters

| Parameter | কাজ                  |
| --------- | -------------------- |
| x, y      | variables            |
| data      | dataset              |
| scatter   | dots show করবে কিনা  |
| fit_reg   | regression line show |
| ci        | confidence interval  |

---

## 💻 Example

```python
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset("tips")

sns.regplot(x="total_bill", y="tip", data=df)

plt.show()
```

---

## 🔥 Advanced Example

```python
sns.regplot(
    x="total_bill",
    y="tip",
    scatter_kws={"color": "blue"},
    line_kws={"color": "red"},
    ci=95,
    data=df
)

plt.show()
```

---

## 🧠 কখন use করবো?

👉 simple relation দেখতে
👉 regression line check করতে

---

# 📘 2. `lmplot()` ⭐ (Advanced version)

---

## 📌 What it does:

👉 regplot + multiple plots + grouping

👉 এটা figure-level function

---

## 🔹 Syntax

```python
sns.lmplot(
    data=None,
    x=None, y=None,
    hue=None,
    col=None,
    row=None
)
```

---

## 🔹 Important Parameters

| Parameter | কাজ            |
| --------- | -------------- |
| hue       | color grouping |
| col       | multiple plot  |
| row       | row split      |

---

## 💻 Example

```python
sns.lmplot(
    x="total_bill",
    y="tip",
    hue="sex",
    data=df
)
```

---

## 🔥 Advanced Example

```python
sns.lmplot(
    x="total_bill",
    y="tip",
    hue="sex",
    col="time",
    data=df
)
```

---

## 🧠 কখন use করবো?

👉 category-wise regression compare করতে
👉 multiple plots দরকার হলে

---

# 📘 3. `residplot()` ⭐

---

## 📌 What it does:

👉 residuals (error) দেখায়

👉 residual = actual - predicted

---

## 🔹 Syntax

```python
sns.residplot(data=None, x=None, y=None)
```

---

## 💻 Example

```python
sns.residplot(x="total_bill", y="tip", data=df)

plt.show()
```

---

## 🧠 Residual Plot কী বোঝায়?

👉 dots যদি random থাকে → model good
👉 pattern থাকলে → model problem

---

# 📊 Visual Idea

![Image](https://images.openai.com/static-rsc-4/oXLWrHWYId1SixWmgvPEUx5JQ_VPruu0wRCAC0wNKIY7KWbrJWFt0Q-qzILGXfcQuVutsdd5cmyZ8nJpZJU50I6KiLYIONu5IRYhULSHZvMeZp9W0t8wREAG_XRbR1mi-FwO3EvEjQ5hvnrawJQEIsAjjbCibBlCbu03CwWqXaoJpRomQSBLh6HsH_8Zjy_3?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/Ezppq8KL6WaUcc3uDCr6sXn5_PHrHs2aAsKbzm4jkDNpGkeHDzPsjaTJrmNPDx_vrQS2WL_riXEdmjidoJ-KYevX3bGGn1KrLDwyPT09daLMBZH0GhrM9mpi90gDE7IsfXL66ZsCsVc0F0XKo-VWsZoOBpT4XIf9vZOrooA9uB8fYW7er2WbWSqbkOQytMKv?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/HFwnekcsGqUA0BdeLVV5VrNBV7U40fB6Nyo_PdT0gUQEw3zYLsJeeFThvJ6xp-b2pkN_16nf71b2hfudh7oEqWwAzUtmNl4qNDwxk8oce14TAKI5HTLk_Xqiwt7PtvLQNcXeRrmn6QCNe8wlvR01NgqR6yixKEsjc5cKivI1iVgeS_qEA4Zvm2ibIx5OuJNy?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/yMi2wQ2dVzCre_EXvoWZ9n4Dg5tE10HFiBDWu4wMj_kNNIgWiAEZHLpFoL490Opu3fqTdpdnmH5ksNVz04kZXsriX4ZGpnxyBvfDR3Qn9-FY4Y-4B2B3pO2j7M64YJgeP9vLZGJVJ2N-OtCTIfszwGQIF5dWDXhysTcz_UvVqbsYtlp5btWjCHNL0ccIOyPy?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/VoVFyf_l8-tEPwIiw4LWWsY6rC41WkTw_63-eH5EQl7x963MhFImKx53lG9FifF5hJsYYfWy93-mS9cCfs7JKivXh3eQydhhorxn53PbPP4K7nkjwbQUvxqtXcp-txkyRe6S4G4-47RsVLWAsGjYP8TlN1aYbBl13fRIgIXIebXSiC7VRUt8cjJMeyK-QlUS?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/9U6bzbTamk5PgiriQXI1NPHcY6_0OBUEsGGV3vW6t-ZRV5XUL_uG_5IAHeWFTK0hH2ck5itD61mwG8E8VfOhvTk26TrJsOWYuwe5kBwofFtcYLEPr0UDJq27EOzA-fFe3-mk6Kb7LJITdRphmvIW--Ub_-moTqJPAwuiymbL7f0RK5oV1bbTDy80BETFnROp?purpose=fullsize)

---

# 🧠 Comparison

| Function  | কাজ                 |
| --------- | ------------------- |
| regplot   | simple regression   |
| lmplot    | advanced + grouping |
| residplot | error analysis      |

---

# 🔥 Real ML Workflow

👉 Step:

1. `regplot()` → relation check
2. `lmplot()` → group compare
3. `residplot()` → model check

---

# 🧠 Final Summary

✔ regplot → scatter + line
✔ lmplot → multiple regression
✔ residplot → error visualization

---

# 🚀 Pro Tip

👉 Interview:

❓ “How to check regression model quality visually?”
👉 Answer: **residplot()**

---

