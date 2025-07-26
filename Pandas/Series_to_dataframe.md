
---

## ✅ ১. Series থেকে DataFrame তৈরি: মৌলিক ধারণা

`pandas.Series` হলো একটি এক-মাত্রিক ডেটা স্ট্রাকচার (index + values)। আর `pandas.DataFrame` হলো দুই-মাত্রিক টেবুলার ফরম্যাট (rows × columns)।

তুমি Series কে বিভিন্নভাবে DataFrame-এ রূপ দিতে পারো, নিচে আমি সব পদ্ধতি + প্রয়োজনীয় শর্ত ব্যাখ্যা করলাম।



---

## 🟦 `pd.Series()` – একমাত্রিক ডেটা স্ট্রাকচার

### ✅ প্রধান উদ্দেশ্য:

একটি 1-dimensional লেবেলড অ্যারে তৈরি করা। প্রতিটি ভ্যালুর সাথে ইনডেক্স যুক্ত থাকে।

### ✅ সিনট্যাক্স:

```python
pd.Series(data=None, index=None, dtype=None, name=None, copy=False, fastpath=False)
```

### ✅ প্যারামিটার ব্যাখ্যা:

| Parameter  | ধরন (Type)          | কাজ (Task)                                               | সাধারণ মান                      |
| ---------- | ------------------- | -------------------------------------------------------- | ------------------------------- |
| `data`     | array-like, scalar  | ভ্যালু যা Series-এ থাকবে                                 | `[1,2,3]`, `np.array()`, `dict` |
| `index`    | array-like          | ভ্যালুর ইনডেক্স নির্ধারণ করে (optional)                  | `['a','b','c']`                 |
| `dtype`    | data-type           | ডেটার টাইপ নির্দিষ্ট করে (optional)                      | `'int'`, `'float'`, `'object'`  |
| `name`     | str                 | Series-এর নাম (label) হিসেবে ব্যবহার হয়                  | `'marks'`, `'price'`            |
| `copy`     | bool                | True হলে data এর কপি নেওয়া হয়                            | `True` বা `False`               |
| `fastpath` | bool (internal use) | শুধুমাত্র internal optimization (you should avoid using) | —                               |

### ✅ উদাহরণ:

```python
import pandas as pd
s = pd.Series([10, 20, 30], index=['a', 'b', 'c'], dtype='int', name='scores')
print(s)
```

---

## 🟨 `pd.DataFrame()` – টেবুলার ডেটা স্ট্রাকচার (2D)

### ✅ প্রধান উদ্দেশ্য:

একটি 2-dimensional, লেবেলড ডেটা টেবিল তৈরি করা — যেখানে সারি ও কলাম থাকে।

### ✅ সিনট্যাক্স:

```python
pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=False)
```

### ✅ প্যারামিটার ব্যাখ্যা:

| Parameter | ধরন (Type)          | কাজ (Task)                                     | সাধারণ মান                 |
| --------- | ------------------- | ---------------------------------------------- | -------------------------- |
| `data`    | ndarray, list, dict | মূল ডেটা যা DataFrame তৈরি করে                 | list-of-list, dict-of-list |
| `index`   | array-like          | সারির label নির্ধারণ করে                       | `['row1','row2']`          |
| `columns` | array-like          | কলামের নাম নির্ধারণ করে                        | `['name','age']`           |
| `dtype`   | data-type           | সব কলামের জন্য এক টাইপ নির্ধারণ করে (optional) | `'float'`, `'object'`      |
| `copy`    | bool                | True হলে data এর কপি নেওয়া হয়                  | `True` বা `False`          |

### ✅ ডেটা হিসেবে কী কী দেওয়া যায়:

| Structure      | Result                    |
| -------------- | ------------------------- |
| list of lists  | সারি হিসেবে ইনপুট ধরে     |
| dict of lists  | কলাম অনুযায়ী ডেটা         |
| Series         | এক কলামের DataFrame       |
| 2D numpy array | rows x columns ডেটা       |
| list of dicts  | প্রতিটি dict → একেকটি row |

### ✅ উদাহরণ ১: List থেকে DataFrame

```python
pd.DataFrame([[1, 2], [3, 4]], columns=['A', 'B'])
```

### ✅ উদাহরণ ২: Dict থেকে DataFrame

```python
pd.DataFrame({'Name': ['Alice', 'Bob'], 'Age': [25, 30]})
```

---

## ✅ তুলনামূলক বিশ্লেষণ (Series vs DataFrame)

| দিক           | `Series`                         | `DataFrame`                         |
| ------------- | -------------------------------- | ----------------------------------- |
| Dimension     | 1D                               | 2D                                  |
| Use           | এক কলামের ডেটা                   | টেবিল/একাধিক কলাম ও সারি            |
| Index         | একমাত্র ইনডেক্স                  | সারি ও কলাম — দুই দিকেই label       |
| Common Params | `data`, `index`, `dtype`, `name` | `data`, `index`, `columns`, `dtype` |

---

## 🧠 শেষ কথা:

* তুমি যদি ছোট ডেটা রাখো (একটি কলাম) → `Series` ব্যবহার করো।
* যদি ডেটা টেবিল আকারে রাখো বা বিশ্লেষণ করো → `DataFrame` ব্যবহার করো।
* `Series` থেকে `DataFrame` বা `DataFrame` থেকে `Series` এ সহজেই যাওয়া যায়।



---

## 🧭 ২. পদ্ধতি ১: Series → ১ কলামের DataFrame

### ▶️ ব্যবহার করো: `pd.DataFrame(series)`

```python
import pandas as pd

s = pd.Series([10, 20, 30], name='marks')

# Series থেকে এক কলামের DataFrame তৈরি
df = pd.DataFrame(s)
print(df)
```

🔹 **শর্ত:** Series-এর `name` থাকলে তা কলামের নাম হয়, না থাকলে default `0` হয়।

---

## 🧭 ৩. পদ্ধতি ২: Series → DataFrame, কলামের নাম নির্ধারণ

### ▶️ ব্যবহার করো: `df = s.to_frame(name='column_name')`

```python
s = pd.Series([100, 200, 300])

df = s.to_frame(name='scores')
print(df)
```

🔹 **শর্ত:** `to_frame()` ফাংশনের `name` প্যারামিটার দিয়ে তুমি সরাসরি কলামের নাম দিতে পারো।

---

## 🧭 ৪. পদ্ধতি ৩: একাধিক Series → একসাথে DataFrame

### ▶️ ব্যবহার করো: `pd.concat([s1, s2], axis=1)`

```python
s1 = pd.Series([1, 2, 3], name='math')
s2 = pd.Series([4, 5, 6], name='english')

df = pd.concat([s1, s2], axis=1)
print(df)
```

🔹 **শর্ত:** সব Series-এর length সমান হতে হবে, না হলে missing value (`NaN`) আসবে।

---

## 📋 ৫. প্রয়োজনীয় প্যারামিটার ও ব্যাখ্যা

| প্যারামিটার     | কোথায় ব্যবহৃত হয়          | কাজ / ব্যাখ্যা               |
| --------------- | ------------------------- | ---------------------------- |
| `name`          | `Series`, `to_frame()`    | কলামের নাম হিসেবে ব্যবহৃত হয় |
| `columns=[...]` | `DataFrame()` constructor | একাধিক কলামের নাম নির্ধারণ   |
| `axis=1`        | `pd.concat()`             | কলাম ভিত্তিক merge করার জন্য |
| `index`         | `Series`, `DataFrame`     | কাস্টম ইনডেক্স সেট করার জন্য |
| `dtype`         | `Series`, `DataFrame`     | ডেটার টাইপ নির্ধারণ করতে     |

---

## ⚠️ সতর্কতা / কন্ডিশন

| শর্ত / কন্ডিশন                  | ব্যাখ্যা                                                |
| ------------------------------- | ------------------------------------------------------- |
| Series-এর ইনডেক্স unique না হলে | DataFrame-এর ইনডেক্সে সমস্যা হতে পারে                   |
| unnamed Series → default col    | যদি `name` না থাকে, তাহলে DataFrame-এর কলামের নাম হবে 0 |
| Series এর লেন্থ সমান না হলে     | `concat()` দিলে unmatched rows-এ `NaN` আসবে             |
| একাধিক Series → DataFrame       | সব Series এর ইনডেক্স না মিললে missing value আসবে        |

---

## 🎯 বাস্তব উদাহরণ: সব কৌশল একত্রে

```python
import pandas as pd

# একাধিক Series তৈরি
s1 = pd.Series([10, 20, 30], name='Math')
s2 = pd.Series([40, 50, 60], name='English')

# ১. to_frame ব্যবহার
df1 = s1.to_frame()
print("🔹 to_frame:\n", df1)

# ২. নাম দিয়ে DataFrame
df2 = pd.DataFrame(s2)
print("🔹 DataFrame with name:\n", df2)

# ৩. একাধিক Series একসাথে
df3 = pd.concat([s1, s2], axis=1)
print("🔹 Combined Series to DataFrame:\n", df3)
```

---

## ✅ উপসংহার

| কাজ                         | ফাংশন                         |
| --------------------------- | ----------------------------- |
| একক Series → DataFrame      | `pd.DataFrame(series)`        |
| Series → DataFrame (নাম সহ) | `series.to_frame(name='col')` |
| একাধিক Series → DataFrame   | `pd.concat([s1, s2], axis=1)` |




---

## ✅ 1. `pd.DataFrame(series)`

**যখন একটি Series কে এক কলামবিশিষ্ট DataFrame-এ রূপান্তর করতে চাও**

```python
import pandas as pd

s = pd.Series([10, 20, 30], name="marks")
df = pd.DataFrame(s)

print(df)
```

📌 **ব্যাখ্যা:**

* এখানে Series-টির index 그대로 DataFrame-এর index হবে
* Series-টির নাম হবে কলামের নাম

🧠 **কখন ব্যবহার করবো:**

* যখন তুমি Series-কে **1-column DataFrame** বানাতে চাও

---

## ✅ 2. `pd.DataFrame([series])`

**যখন Series-এর প্রতিটি element একটি column হবে (row-wise)**

```python
s = pd.Series([10, 20, 30], name="row1")
df = pd.DataFrame([s])

print(df)
```

📌 **ব্যাখ্যা:**

* এখানে Series-টি **একটি row হিসেবে** DataFrame-এ যাবে
* Series-এর index → DataFrame-এর columns হবে

🧠 **কখন ব্যবহার করবো:**

* যদি তুমি Series-কে **row হিসেবে** ব্যবহার করতে চাও

---

## ✅ 3. Multiple Series → Columns in DataFrame (with aligned index)

```python
s1 = pd.Series([1, 2, 3], index=['a', 'b', 'c'], name='math')
s2 = pd.Series([4, 5, 6], index=['a', 'b', 'c'], name='physics')

df = pd.DataFrame({'math': s1, 'physics': s2})
print(df)
```

📌 **ব্যাখ্যা:**

* দুইটা Series-এর index এক হওয়ায় **column-wise merge** হয়

🧠 **কখন ব্যবহার করবো:**

* যখন তুমি **একাধিক Series → ১টি DataFrame-এর বিভিন্ন কলাম** বানাতে চাও
* এবং তাদের index এক হলে automatic alignment হবে

---

## ✅ 4. Multiple Series → Columns in DataFrame (unaligned index)

```python
s1 = pd.Series([1, 2, 3], index=['a', 'b', 'c'], name='math')
s2 = pd.Series([4, 5, 6], index=['x', 'y', 'z'], name='physics')

df = pd.DataFrame({'math': s1, 'physics': s2})
print(df)
```

📌 **ব্যাখ্যা:**

* এখানে index mismatch → ফলে DataFrame-এ NaN আসবে
* full outer join-এর মতো কাজ করবে

🧠 **কখন ব্যবহার করবো:**

* যদি তুমি জানো Series-এর index ভিন্ন → তবে `.values` ব্যবহার করো

```python
df = pd.DataFrame({'math': s1, 'physics': s2.values})
```

---

## ✅ 5. `pd.concat([s1, s2], axis=1)`

**Series-গুলোকে column হিসেবে যুক্ত করতে**

```python
s1 = pd.Series([1, 2, 3], name='math')
s2 = pd.Series([4, 5, 6], name='physics')

df = pd.concat([s1, s2], axis=1)
print(df)
```

📌 **ব্যাখ্যা:**

* `axis=1` দিলে column-wise যোগ হয়
* index-align না হলে `NaN` আসবে

🧠 **কখন ব্যবহার করবো:**

* যখন তুমি চাইছো **একাধিক Series column হিসেবে বসাতে**

---

## 📌 Extra: Series এর `.to_frame()`

**Series কে সরাসরি DataFrame-এ রূপান্তর**

```python
s = pd.Series([7, 8, 9], name='score')
df = s.to_frame()
print(df)
```

🧠 ভালো যদি তুমি শুধুমাত্র **একটি Series → DataFrame** করতে চাও

---

## ✅ উপসংহার টেবিল

| পদ্ধতি                            | কাজ করে                             | কবে ব্যবহার করবো                  |
| --------------------------------- | ----------------------------------- | --------------------------------- |
| `pd.DataFrame(series)`            | Series → 1-column DataFrame         | Simple কলাম বানাতে                |
| `pd.DataFrame([series])`          | Series → Row                        | Series-এর প্রতিটি ভ্যালু কলাম হলে |
| `pd.DataFrame({s1, s2})`          | একাধিক Series → columns             | যদি index match করে               |
| `pd.DataFrame({'col': s.values})` | index mismatch হলে values দিয়ে বসাও | index mismatch করলে               |
| `pd.concat([...], axis=1)`        | Multiple Series → column bind       | Flexible/complex সিচুয়েশনে        |
| `s.to_frame()`                    | Series → DataFrame                  | সবচেয়ে সহজ এক-কলামের রূপান্তরে    |

---



## 📘 Documentation: Dictionary থেকে DataFrame এবং Series থেকে DataFrame এর Critical Handling

---

### ✅ Dictionary ➡ DataFrame

```python
import pandas as pd

data = {
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35]
}

df = pd.DataFrame(data)
print(df)
```

#### 🧠 ব্যাখ্যা:

* এখানে `dictionary`-র **key গুলো DataFrame এর column** হিসেবে পরিণত হয়।
* আর `value` গুলো হতে হবে **list-like**, যাদের length সমান হতে হবে।

📌 যদি `value` গুলোর length অসমান হয়, তাহলে **ValueError** দিবে।

---

### ✅ Series ➡ DataFrame: Critical Considerations

#### 🎯 লক্ষ্য:

Series গুলোকে DataFrame-এ রূপান্তর করতে হলে তাদের index গুলো **align** করতে হয়।

---

### ✅ সমস্যা: Index mismatch

```python
series1 = pd.Series([1, 2, 3], index=['a', 'b', 'c'], name='math')
series2 = pd.Series([4, 5, 6], index=['x', 'y', 'z'], name='physics')

df = pd.DataFrame({
    'math': series1,
    'physics': series2
})

print(df)
```

#### 🔍 Output:

```
   math  physics
a   1.0      NaN
b   2.0      NaN
c   3.0      NaN
x   NaN      4.0
y   NaN      5.0
z   NaN      6.0
```

**কারণ:** `Series` দুটির index আলাদা হওয়ায় Pandas তাদের **outer join** করে—তাই `NaN` আসে।

---

### ✅ সমাধান ১: `.values` দিয়ে alignment ঠিক রাখা

```python
series1.index = series2.index  # index match
df = pd.DataFrame({
    'math': series1.values,
    'physics': series2.values
}, index=series2.index)
print(df)
```

---

### ✅ সমাধান ২: `concat()` দিয়ে intelligent alignment

```python
df2 = pd.concat([series1.rename('math'), series2.rename('physics')], axis=1)
print(df2)
```

📌 এই পদ্ধতিতে Pandas index mismatch হ্যান্ডেল করে intelligently।

---

## 🔄 উপরের Example: Final Correct Version

```python
import pandas as pd

# Step 1: List
lst = [1,2,3,4,5,6,7,8,9,10]

# Step 2: Series 1 (index a-j)
series1 = pd.Series(lst, index=['a','b','c','d','e','f','g','h','i','j'], dtype='int', name='numbers1')

# Step 3: Series 2 (index A-J)
series2 = pd.Series(lst, index=['A','B','C','D','E','F','G','H','I','J'], dtype='int', name='numbers2')

# Step 4: Force index match
series1.index = series2.index  # both have index A-J

# Step 5: Create DataFrame
df = pd.DataFrame({
    'numbers1': series1.values,
    'numbers2': series2.values
}, index=series1.index)

print(df)
```

---

## 📋 Summary Table: Key Differences

| Structure     | Keys used as...     | When to align manually?     | Notes                             |
| ------------- | ------------------- | --------------------------- | --------------------------------- |
| `dict → df`   | Keys become columns | ❌ না (auto handled)         | Values must be equal-length lists |
| `Series → df` | Index becomes rows  | ✅ হ্যাঁ, if merging columns | Otherwise will show NaNs          |

---

## ✅ Conclusion

* `dictionary` থেকে DataFrame তৈরি করা সবচেয়ে সহজ — শুধু `key = column`, `value = list` হতে হয়।
* কিন্তু **multiple Series একসাথে যোগ করার সময় index mismatch সবচেয়ে বড় issue**।
* এই সমস্যা সমাধানে:

  * `.values` ব্যবহার করে alignment নিশ্চিত করা যায়
  * অথবা `pd.concat([...], axis=1)` intelligent alignment করতে পারে

---
# 📁 Full Documentation: Series to DataFrame with pd.concat(axis=1)

---

## 📄 Overview

In pandas, we often need to convert one or more Series into a DataFrame. One of the most common and powerful methods for this is using `pd.concat()` with the parameter `axis=1`. This method stacks Series horizontally as columns of a new DataFrame.

---

## 🔎 Why Use `pd.concat()`?

The function `pd.concat()` is used to concatenate pandas objects along a particular axis:

* **`axis=0`**: Vertical stacking (row-wise)
* **`axis=1`**: Horizontal stacking (column-wise)

### 💡 Purpose of `axis=1`

Using `axis=1` tells pandas to **align the Series side by side** using their index. This is critical when converting Series to a DataFrame where **each Series represents a column**.

---

## 📈 Syntax

```python
pd.concat(objs, axis=1, join='outer', ignore_index=False, keys=None)
```

### Key Parameters:

| Parameter      | Description                                       |
| -------------- | ------------------------------------------------- |
| `objs`         | List of Series or DataFrames to concatenate       |
| `axis=1`       | Stack objects column-wise (side by side)          |
| `join`         | Join method: `'outer'` (default), `'inner'`       |
| `ignore_index` | If True, do not use index labels                  |
| `keys`         | Create a hierarchical index using the passed keys |

---

## 🔢 Practical Example 1: Series to DataFrame (Correct Way)

```python
import pandas as pd

# Two Series with same index
s1 = pd.Series([10, 20, 30], index=['a', 'b', 'c'], name='math')
s2 = pd.Series([40, 50, 60], index=['a', 'b', 'c'], name='physics')

# Combine as DataFrame
df = pd.concat([s1, s2], axis=1)
print(df)
```

### Output:

```
   math  physics
a    10       40
b    20       50
c    30       60
```

---

## 🔢 Example 2: Mismatched Index with .values

```python
# Two Series with different indexes
s1 = pd.Series([1,2,3], index=['a','b','c'], name='A')
s2 = pd.Series([4,5,6], index=['x','y','z'], name='B')

# Option 1: Force alignment by resetting index
s2.index = s1.index

# Option 2: Use .values to bypass index mismatch
df = pd.DataFrame({
    'A': s1,
    'B': s2.values
})
print(df)
```

---

## 📖 Common Use Cases

| Scenario                            | Method                                 |
| ----------------------------------- | -------------------------------------- |
| Combine multiple Series as columns  | `pd.concat([...], axis=1)`             |
| Convert labeled Series to DataFrame | `pd.DataFrame({'col1': s1, ...})`      |
| Avoid misalignment                  | Use `.values` to ignore index mismatch |
| Use hierarchical columns            | Use `keys=['A', 'B']` in concat        |

---

## 📌 Bonus: Real-Life Code Example

```python
import pandas as pd

# Create list
lst = [1,2,3,4,5,6,7,8,9,10]

# Series with different indexes
series1 = pd.Series(lst, index=['a','b','c','d','e','f','g','h','i','j'], name='numbers1')
series2 = pd.Series(lst, index=['A','B','C','D','E','F','G','H','I','J'], name='numbers2')

# Reset index of series2 to match series1
series2.index = series1.index

# Merge into DataFrame
df = pd.concat([series1, series2], axis=1)

print(df)
```

---

## 📊 Conclusion

Using `pd.concat()` with `axis=1` is a reliable and flexible way to merge multiple Series as columns into a single DataFrame. It's especially useful in scenarios where the Series objects already have aligned indices or where you want to manually align them.



---

## ✅ 1. **Single Column → Series**

```python
import pandas as pd

df = pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie'],
                   'age': [25, 30, 35]})

# কেবল একটি column কে Series বানানো
s = df['age']
print(s)
```

🔹 Output:

```
0    25
1    30
2    35
Name: age, dtype: int64
```

---

## ✅ 2. **Single Row → Series (`iloc[]` বা `loc[]` ব্যবহার করে)**

```python
row_series = df.loc[1]  # index 1 এর row
print(row_series)
```

🔹 Output:

```
name    Bob
age       30
Name: 1, dtype: object
```

---

## ✅ 3. **Entire DataFrame → Series (flattened)**

এটা তখন করা হয় যখন পুরো DataFrame কে একটানা এক কলামে রূপান্তর করতে চাও:

```python
s = df.stack()  # Row-wise stack into a single column
print(s)
```

🔹 Output:

```
0    name      Alice
     age          25
1    name        Bob
     age          30
2    name    Charlie
     age          35
dtype: object
```

---

## ✅ 4. **DataFrame → Numpy → Series**

```python
s = pd.Series(df.values.flatten())
print(s)
```

🔹 Output:

```
0      Alice
1         25
2        Bob
3         30
4    Charlie
5         35
dtype: object
```

---

## ✅ 5. **Using `.squeeze()` for 1-column or 1-row DataFrame**

```python
# Single-column DataFrame
df_single = pd.DataFrame({'score': [90, 80, 70]})
s = df_single.squeeze()
print(s)
```

🔹 Output:

```
0    90
1    80
2    70
Name: score, dtype: int64
```

---

## 📋 Summary Table

| উদ্দেশ্য                        | পদ্ধতি                            |
| ------------------------------- | --------------------------------- |
| একটি column কে Series           | `df['col']`                       |
| একটি row কে Series              | `df.loc[index]`, `df.iloc[index]` |
| DataFrame কে ফ্ল্যাট করে Series | `df.stack()` or `flatten()`       |
| 1-column DataFrame → Series     | `df.squeeze()`                    |

---

---

## ✅ 1. **Dictionary → DataFrame**

### 📌 Case 1: Dictionary of lists/series (Column-wise)

```python
import pandas as pd

data = {
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35]
}

df = pd.DataFrame(data)
print(df)
```

📤 Output:

```
     name  age
0   Alice   25
1     Bob   30
2  Charlie  35
```

📍 **এখানে কী হচ্ছে:**

* dictionary-এর **keys** হচ্ছে column name
* **values (list)** হচ্ছে row-wise values

---

### 📌 Case 2: Dictionary of scalar values

```python
data = {'A': 10, 'B': 20}
df = pd.DataFrame([data])
print(df)
```

📤 Output:

```
    A   B
0  10  20
```

---

### 📌 Case 3: Dictionary of dictionaries (Nested dict)

```python
data = {
    'student1': {'name': 'Alice', 'age': 25},
    'student2': {'name': 'Bob', 'age': 30}
}

df = pd.DataFrame(data)
print(df.T)  # transpose if needed
```

📤 Output:

```
           name  age
student1  Alice   25
student2    Bob   30
```

---

## ✅ 2. **Set → DataFrame**

👉 Set unordered হয়, তাই tuple বা list বানিয়ে ব্যবহার করতে হয়।

### 📌 Case 1: Set of tuples → rows

```python
data = {(1, 'Alice'), (2, 'Bob'), (3, 'Charlie')}
df = pd.DataFrame(list(data), columns=['ID', 'Name'])
print(df)
```

📤 Output:

```
   ID     Name
0   1    Alice
1   2      Bob
2   3  Charlie
```

---

### 📌 Case 2: Set of scalars (1D)

```python
data = {10, 20, 30}
df = pd.DataFrame({'values': list(data)})
print(df)
```

📤 Output:

```
   values
0      10
1      20
2      30
```

---

## ✅ 3. **Tuple → DataFrame**

### 📌 Case 1: List of tuples → rows

```python
data = [(1, 'Math'), (2, 'Science'), (3, 'English')]
df = pd.DataFrame(data, columns=['ID', 'Subject'])
print(df)
```

📤 Output:

```
   ID  Subject
0   1     Math
1   2  Science
2   3  English
```

---

### 📌 Case 2: Tuple of tuples → rows

```python
data = ((1, 'A'), (2, 'B'))
df = pd.DataFrame(data, columns=['Roll', 'Grade'])
print(df)
```

📤 Output:

```
   Roll Grade
0     1     A
1     2     B
```

---

## 📋 Summary Table

| Structure            | Example                       | Method                                   |
| -------------------- | ----------------------------- | ---------------------------------------- |
| Dict of lists        | `{'a': [1,2], 'b':[3,4]}`     | `pd.DataFrame(dict)`                     |
| Dict of dicts        | `{'x': {'a':1}, 'y':{'a':2}}` | `pd.DataFrame(dict).T`                   |
| Set of tuples        | `{(1,'a'), (2,'b')}`          | `pd.DataFrame(list(set), columns=[...])` |
| Set of scalars       | `{1,2,3}`                     | `pd.DataFrame({'col': list(set)})`       |
| List/Tuple of tuples | `[(1,'x'), (2,'y')]`          | `pd.DataFrame(data, columns=[...])`      |

---




