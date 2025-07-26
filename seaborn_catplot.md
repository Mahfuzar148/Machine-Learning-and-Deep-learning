
---

## 🐱 `sns.catplot()` – বিস্তারিত বিশ্লেষণ

### ✅ কাজ কী?

`sns.catplot()` হলো Seaborn-এর একটি **Figure-level function** যা দিয়ে **categorical variables** ভিত্তিক প্লট তৈরি করা যায়। এটি একটি wrapper যার `kind` argument দিয়ে তুমি যেকোনো এক ধরণের ক্যাটাগরিকাল প্লট বানাতে পারো।

👉 এটি দিয়ে **barplot, boxplot, violinplot, stripplot, swarmplot, pointplot, countplot** তৈরি করা যায় — সব এক ফাংশনেই!

---

## 🔧 `catplot()` দিয়ে কী কী করা যায়?

| টাস্ক                         | ব্যাখ্যা               |
| ----------------------------- | ---------------------- |
| Category ভিত্তিক গড় বের করা   | যেমন দিনে গড় খরচ       |
| Outlier খুঁজে বের করা         | box/violin plot দিয়ে   |
| Category অনুযায়ী distribution | strip/swarm plot       |
| Group-wise comparison         | hue parameter দিয়ে     |
| Subplot/faceting              | row/col parameter দিয়ে |

---

## 🧾 Full Parameter Table

| Parameter          | মান                                                                                 | কাজ                                |
| ------------------ | ----------------------------------------------------------------------------------- | ---------------------------------- |
| `data`             | DataFrame                                                                           | ডেটাসেট                            |
| `x`, `y`           | column name                                                                         | একটিকে category, অন্যটি numeric    |
| `hue`              | column                                                                              | ভেতরের গ্রুপ রঙে আলাদা             |
| `col`, `row`       | column                                                                              | subplot তৈরি করতে                  |
| `kind`             | `'strip'`, `'swarm'`, `'box'`, `'violin'`, `'boxen'`, `'point'`, `'bar'`, `'count'` | কী টাইপের ক্যাটাগরিকাল প্লট        |
| `height`           | সংখ্যা                                                                              | প্রতিটি subplot-এর উচ্চতা          |
| `aspect`           | সংখ্যা                                                                              | চওড়া/উচ্চতার অনুপাত               |
| `order`            | list                                                                                | x-axis category-এর অর্ডার কন্ট্রোল |
| `hue_order`        | list                                                                                | hue category-এর অর্ডার             |
| `palette`          | `'deep'`, `'muted'`, `'pastel'`, `'bright'`, `'dark'`, `'colorblind'`               | রঙের থিম                           |
| `legend`           | `'brief'`, `'full'`, `False`                                                        | লেজেন্ড দেখাবে কিনা                |
| `margin_titles`    | True/False                                                                          | subplot margin এ title দেবে কিনা   |
| `sharex`, `sharey` | True/False                                                                          | subplot গুলোর scale এক হবে কিনা    |

---

## 🧪 Coding Examples for Each `kind`

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset(\"tips\")

# stripplot (default)
sns.catplot(data=tips, x=\"day\", y=\"total_bill\", kind=\"strip\")
plt.title(\"stripplot\")
plt.show()

# swarmplot
sns.catplot(data=tips, x=\"day\", y=\"total_bill\", kind=\"swarm\")
plt.title(\"swarmplot\")
plt.show()

# boxplot
sns.catplot(data=tips, x=\"day\", y=\"total_bill\", kind=\"box\")
plt.title(\"boxplot\")
plt.show()

# violinplot
sns.catplot(data=tips, x=\"day\", y=\"total_bill\", kind=\"violin\")
plt.title(\"violinplot\")
plt.show()

# boxenplot
sns.catplot(data=tips, x=\"day\", y=\"total_bill\", kind=\"boxen\")
plt.title(\"boxenplot\")
plt.show()

# pointplot
sns.catplot(data=tips, x=\"day\", y=\"total_bill\", kind=\"point\")
plt.title(\"pointplot\")
plt.show()

# barplot
sns.catplot(data=tips, x=\"day\", y=\"total_bill\", kind=\"bar\")
plt.title(\"barplot\")
plt.show()

# countplot (no y required)
sns.catplot(data=tips, x=\"day\", kind=\"count\")
plt.title(\"countplot\")
plt.show()
```

---

## 🖼️ Faceting, Hue, Order, Palette, Size: Examples

```python
# hue parameter
sns.catplot(data=tips, x=\"day\", y=\"total_bill\", hue=\"sex\", kind=\"box\")
plt.title(\"hue example\")
plt.show()

# col and row (faceting)
sns.catplot(data=tips, x=\"day\", y=\"total_bill\", col=\"sex\", row=\"time\", kind=\"violin\")
plt.show()

# height and aspect
sns.catplot(data=tips, x=\"day\", y=\"total_bill\", kind=\"bar\", height=4, aspect=1.5)
plt.title(\"height & aspect\")
plt.show()

# order & hue_order
sns.catplot(data=tips, x=\"day\", y=\"total_bill\", hue=\"sex\", kind=\"point\", 
            order=[\"Sun\", \"Sat\", \"Thur\", \"Fri\"], hue_order=[\"Female\", \"Male\"])
plt.title(\"order & hue_order\")
plt.show()

# palette
sns.catplot(data=tips, x=\"day\", y=\"total_bill\", hue=\"sex\", kind=\"bar\", palette=\"muted\")
plt.title(\"palette example\")
plt.show()

# legend and margin_titles
sns.catplot(data=tips, x=\"day\", y=\"total_bill\", col=\"sex\", kind=\"box\", 
            legend=\"full\", margin_titles=True)
plt.show()
```

---

## ✅ উপসংহার

| Feature          | `catplot()` দিয়ে করা যায়?       |
| ---------------- | ------------------------------- |
| বার প্লট         | ✅ kind='bar'                    |
| বক্সপ্লট         | ✅ kind='box'                    |
| ভায়োলিনপ্লট      | ✅ kind='violin'                 |
| কাস্টম অর্ডার    | ✅ `order`, `hue_order`          |
| গ্রুপিং রঙে      | ✅ `hue`                         |
| সাবপ্লট তৈরি     | ✅ `col`, `row`                  |
| চেহারা নিয়ন্ত্রণ | ✅ `height`, `aspect`, `palette` |

---

🔧


```python
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# Load the dataset
tips = sns.load_dataset("tips")

st.title("🐱 Seaborn catplot() Examples Dashboard")
st.write("Explore all kinds of categorical plots using sns.catplot() with different parameters.")

# kind: strip (default)
st.subheader("kind='strip'")
st.pyplot(sns.catplot(data=tips, x="day", y="total_bill", kind="strip").figure)

# kind: swarm
st.subheader("kind='swarm'")
st.pyplot(sns.catplot(data=tips, x="day", y="total_bill", kind="swarm").figure)

# kind: box
st.subheader("kind='box'")
st.pyplot(sns.catplot(data=tips, x="day", y="total_bill", kind="box").figure)

# kind: violin
st.subheader("kind='violin'")
st.pyplot(sns.catplot(data=tips, x="day", y="total_bill", kind="violin").figure)

# kind: boxen
st.subheader("kind='boxen'")
st.pyplot(sns.catplot(data=tips, x="day", y="total_bill", kind="boxen").figure)

# kind: point
st.subheader("kind='point'")
st.pyplot(sns.catplot(data=tips, x="day", y="total_bill", kind="point").figure)

# kind: bar
st.subheader("kind='bar'")
st.pyplot(sns.catplot(data=tips, x="day", y="total_bill", kind="bar").figure)

# kind: count
st.subheader("kind='count'")
st.pyplot(sns.catplot(data=tips, x="day", kind="count").figure)

# hue
st.subheader("hue='sex'")
st.pyplot(sns.catplot(data=tips, x="day", y="total_bill", hue="sex", kind="box").figure)

# col and row
st.subheader("Faceting with col and row")
st.pyplot(sns.catplot(data=tips, x="day", y="total_bill", col="sex", row="time", kind="violin").figure)

# height and aspect
st.subheader("height=4, aspect=1.5")
st.pyplot(sns.catplot(data=tips, x="day", y="total_bill", kind="bar", height=4, aspect=1.5).figure)

# order and hue_order
st.subheader("order and hue_order")
st.pyplot(sns.catplot(data=tips, x="day", y="total_bill", hue="sex", kind="point",
                      order=["Sun", "Sat", "Thur", "Fri"],
                      hue_order=["Female", "Male"]).figure)

# palette
st.subheader("palette='muted'")
st.pyplot(sns.catplot(data=tips, x="day", y="total_bill", hue="sex", kind="bar", palette="muted").figure)

# legend and margin_titles
st.subheader("legend='full', margin_titles=True")
st.pyplot(sns.catplot(data=tips, x="day", y="total_bill", col="sex", kind="box",
                      legend="full", margin_titles=True).figure)
```


### ✅ অন্তর্ভুক্ত উদাহরণসমূহ:

* `kind`: `'strip'`, `'swarm'`, `'box'`, `'violin'`, `'boxen'`, `'point'`, `'bar'`, `'count'`
* `hue`: গ্রুপভিত্তিক রঙ পরিবর্তন
* `col`, `row`: subplot তৈরি
* `height`, `aspect`: ফিগারের মাপ নিয়ন্ত্রণ
* `order`, `hue_order`: ক্যাটাগরির অর্ডার নির্ধারণ
* `palette`: কালার স্কিম
* `legend`, `margin_titles`: লেজেন্ড ও টাইটেল কনফিগার

