## 📊 Pandas কী?

**Pandas** হলো Python-এর একটি শক্তিশালী Data Analysis ও Data Manipulation লাইব্রেরি।
এটি মূলত টেবিল আকারের ডেটা (যেমন CSV, Excel, SQL ডেটা) নিয়ে কাজ করার জন্য ব্যবহৃত হয়।

👉 Machine Learning, Data Science, AI, Financial Analysis — সব জায়গায় Pandas ব্যবহার হয়।

---

# 🔷 Pandas এর প্রধান ডেটা স্ট্রাকচার

| Structure     | কাজ                                |
| ------------- | ---------------------------------- |
| **Series**    | একমাত্রিক ডেটা (1D array)          |
| **DataFrame** | টেবিল আকারের ডেটা (rows + columns) |

---

# 🔥 Pandas Built-in Functions (Category-wise Full List)

আমি সবচেয়ে বেশি ব্যবহৃত গুরুত্বপূর্ণ built-in function গুলো সাজিয়ে দিচ্ছি।

---

# ✅ 1️⃣ Data Loading Functions

```python
pd.read_csv()
pd.read_excel()
pd.read_json()
pd.read_sql()
pd.read_html()
pd.read_parquet()
```

---

# ✅ 2️⃣ Data Saving Functions

```python
df.to_csv()
df.to_excel()
df.to_json()
df.to_sql()
df.to_parquet()
```

---

# ✅ 3️⃣ Basic Information Functions

```python
df.head()
df.tail()
df.info()
df.describe()
df.shape
df.columns
df.index
df.dtypes
df.value_counts()
df.nunique()
```

---

# ✅ 4️⃣ Data Selection & Indexing

```python
df.loc[]
df.iloc[]
df.at[]
df.iat[]
df.filter()
df.query()
```

---

# ✅ 5️⃣ Data Cleaning Functions

```python
df.isnull()
df.notnull()
df.dropna()
df.fillna()
df.replace()
df.drop()
df.rename()
df.astype()
df.duplicated()
df.drop_duplicates()
```

---

# ✅ 6️⃣ Sorting Functions

```python
df.sort_values()
df.sort_index()
```

---

# ✅ 7️⃣ Aggregation & Statistics

```python
df.mean()
df.median()
df.mode()
df.sum()
df.min()
df.max()
df.std()
df.var()
df.corr()
df.cov()
df.count()
```

---

# ✅ 8️⃣ Grouping Functions

```python
df.groupby()
df.agg()
df.transform()
df.pivot()
df.pivot_table()
df.crosstab()
```

---

# ✅ 9️⃣ Merge / Join Functions

```python
pd.concat()
pd.merge()
df.join()
```

---

# ✅ 🔟 Apply & Function Operations

```python
df.apply()
df.applymap()
df.map()
```

---

# ✅ 11️⃣ Time Series Functions

```python
pd.to_datetime()
df.resample()
df.shift()
df.rolling()
df.expanding()
```

---

# ✅ 12️⃣ String Operations

```python
df['col'].str.lower()
df['col'].str.upper()
df['col'].str.contains()
df['col'].str.replace()
df['col'].str.split()
```

---

# 🎯 Example

```python
import pandas as pd

df = pd.read_csv("data.csv")

print(df.head())
print(df.describe())

df = df.dropna()
df = df.sort_values("Age")

print(df.mean())
```

---

# 🔥 সংক্ষেপে

Pandas ব্যবহার হয়:

✔ ডেটা পড়া
✔ ডেটা পরিষ্কার করা
✔ ডেটা বিশ্লেষণ
✔ ডেটা রূপান্তর
✔ Machine Learning প্রস্তুত করা




---

# ✅ 🔷 GROUP 1: Data Loading Functions

---

# 1️⃣ `pd.read_csv()`

---

## 📌 কাজ কী?

CSV ফাইল থেকে DataFrame তৈরি করে।

## 🧾 Common Syntax

```python
pd.read_csv(filepath, sep=',', header=0, index_col=None)
```

## 🔍 গুরুত্বপূর্ণ Parameter

| Parameter | কাজ                 |
| --------- | ------------------- |
| filepath  | ফাইলের path বা নাম  |
| sep       | কলাম separator      |
| header    | কোন row header হবে  |
| index_col | কোন কলাম index হবে  |
| usecols   | নির্দিষ্ট কলাম পড়বে |
| nrows     | কত row পড়বে         |

## 💻 Example

```python
import pandas as pd

df = pd.read_csv("data.csv")
print(df.head())
```

---

# 2️⃣ `pd.read_excel()`

---

## 📌 কাজ কী?

Excel (.xlsx) ফাইল থেকে ডেটা পড়ে।

## 🧾 Common Syntax

```python
pd.read_excel(filepath, sheet_name=0)
```

## 🔍 Parameter

| Parameter  | কাজ              |
| ---------- | ---------------- |
| filepath   | Excel ফাইলের নাম |
| sheet_name | কোন sheet পড়বে   |
| usecols    | নির্দিষ্ট কলাম   |

## 💻 Example

```python
df = pd.read_excel("data.xlsx", sheet_name="Sheet1")
print(df.head())
```

---

# 3️⃣ `pd.read_json()`

---

## 📌 কাজ কী?

JSON ফাইল থেকে ডেটা পড়ে।

## 🧾 Syntax

```python
pd.read_json(filepath)
```

## 💻 Example

```python
df = pd.read_json("data.json")
print(df.head())
```

---

# 4️⃣ `pd.read_sql()`

---

## 📌 কাজ কী?

SQL database থেকে query result পড়ে DataFrame বানায়।

## 🧾 Syntax

```python
pd.read_sql(query, connection)
```

## 🔍 Parameter

| Parameter  | কাজ                        |
| ---------- | -------------------------- |
| query      | SQL query                  |
| connection | Database connection object |

## 💻 Example

```python
import sqlite3

conn = sqlite3.connect("database.db")

df = pd.read_sql("SELECT * FROM users", conn)

print(df.head())
```

---

# 5️⃣ `pd.read_html()`

---

## 📌 কাজ কী?

ওয়েব পেজের HTML table থেকে ডেটা পড়ে।

## 🧾 Syntax

```python
pd.read_html(url)
```

## 💻 Example

```python
url = "https://example.com/table"
tables = pd.read_html(url)

df = tables[0]
print(df.head())
```

---

# 6️⃣ `pd.read_parquet()`

---

## 📌 কাজ কী?

Parquet format ফাইল থেকে ডেটা পড়ে।

(Parquet বড় dataset এর জন্য optimized)

## 🧾 Syntax

```python
pd.read_parquet(filepath)
```

## 💻 Example

```python
df = pd.read_parquet("data.parquet")
print(df.head())
```

---

# 🎯 Group 1 Summary

| Function     | ব্যবহার         |
| ------------ | --------------- |
| read_csv     | CSV ফাইল        |
| read_excel   | Excel ফাইল      |
| read_json    | JSON            |
| read_sql     | Database        |
| read_html    | Web table       |
| read_parquet | Big data format |

---



# ✅ 🔷 GROUP 2: Data Saving Functions

এই গ্রুপের সব function দিয়ে DataFrame থেকে ডেটা বিভিন্ন ফরম্যাটে save করা যায়।

ফরম্যাট আগের মতোই থাকবে 👇

---

# 1️⃣ `df.to_csv()`

---

## 📌 কাজ কী?

DataFrame কে CSV ফাইলে সংরক্ষণ করে।

## 🧾 Common Syntax

```python
df.to_csv(filepath, index=True)
```

## 🔍 গুরুত্বপূর্ণ Parameter

| Parameter | কাজ                                  |
| --------- | ------------------------------------ |
| filepath  | ফাইলের নাম                           |
| index     | index save করবে কি না (default=True) |
| sep       | separator                            |
| header    | column name save করবে কি না          |

## 💻 Example

```python
import pandas as pd

data = {"Name": ["Rahim", "Karim"], "Age": [22, 25]}
df = pd.DataFrame(data)

df.to_csv("output.csv", index=False)
```

---

# 2️⃣ `df.to_excel()`

---

## 📌 কাজ কী?

DataFrame কে Excel ফাইলে save করে।

## 🧾 Syntax

```python
df.to_excel(filepath, sheet_name="Sheet1", index=True)
```

## 🔍 Parameter

| Parameter  | কাজ                  |
| ---------- | -------------------- |
| filepath   | Excel ফাইলের নাম     |
| sheet_name | কোন sheet এ save হবে |
| index      | index save হবে কি না |

## 💻 Example

```python
df.to_excel("output.xlsx", sheet_name="Data", index=False)
```

---

# 3️⃣ `df.to_json()`

---

## 📌 কাজ কী?

DataFrame কে JSON format এ save করে।

## 🧾 Syntax

```python
df.to_json(filepath)
```

## 💻 Example

```python
df.to_json("output.json")
```

---

# 4️⃣ `df.to_sql()`

---

## 📌 কাজ কী?

DataFrame কে database table এ save করে।

## 🧾 Syntax

```python
df.to_sql(table_name, connection, if_exists='fail')
```

## 🔍 Parameter

| Parameter  | কাজ                     |
| ---------- | ----------------------- |
| table_name | টেবিলের নাম             |
| connection | database connection     |
| if_exists  | fail / replace / append |

## 💻 Example

```python
import sqlite3

conn = sqlite3.connect("database.db")

df.to_sql("users", conn, if_exists="replace", index=False)
```

---

# 5️⃣ `df.to_parquet()`

---

## 📌 কাজ কী?

DataFrame কে Parquet format এ save করে (Big data এর জন্য efficient)।

## 🧾 Syntax

```python
df.to_parquet(filepath)
```

## 💻 Example

```python
df.to_parquet("output.parquet")
```

---

# 🎯 Group 2 Summary

| Function   | ব্যবহার         |
| ---------- | --------------- |
| to_csv     | CSV file        |
| to_excel   | Excel file      |
| to_json    | JSON file       |
| to_sql     | Database table  |
| to_parquet | Big data format |

---



# ✅ 🔷 GROUP 3: Basic Information Functions

এই function গুলো দিয়ে আমরা dataset সম্পর্কে প্রাথমিক তথ্য পাই।

ফরম্যাট একই থাকবে 👇

---

# 1️⃣ `df.head()`

---

## 📌 কাজ কী?

DataFrame এর প্রথম কয়েকটি row দেখায়।

## 🧾 Syntax

```python
df.head(n)
```

## 🔍 Parameter

| Parameter | কাজ                         |
| --------- | --------------------------- |
| n         | কত row দেখাবে (default = 5) |

## 💻 Example

```python
import pandas as pd

df = pd.read_csv("data.csv")

print(df.head())
print(df.head(3))
```

---

# 2️⃣ `df.tail()`

---

## 📌 কাজ কী?

DataFrame এর শেষ কয়েকটি row দেখায়।

## 🧾 Syntax

```python
df.tail(n)
```

## 🔍 Parameter

| Parameter | কাজ                         |
| --------- | --------------------------- |
| n         | কত row দেখাবে (default = 5) |

## 💻 Example

```python
print(df.tail())
print(df.tail(2))
```

---

# 3️⃣ `df.info()`

---

## 📌 কাজ কী?

Dataset এর summary দেখায়:

* কত row
* কত column
* Data type
* Null value

## 🧾 Syntax

```python
df.info()
```

## 💻 Example

```python
df.info()
```

---

# 4️⃣ `df.describe()`

---

## 📌 কাজ কী?

Numeric column এর statistical summary দেয়।

* count
* mean
* std
* min
* 25%
* 50%
* 75%
* max

## 🧾 Syntax

```python
df.describe()
```

## 💻 Example

```python
print(df.describe())
```

---

# 5️⃣ `df.shape`

---

## 📌 কাজ কী?

Dataset এ মোট কত row ও column আছে তা দেয়।

## 🧾 Syntax

```python
df.shape
```

## 💻 Example

```python
print(df.shape)
```

Output:

```
(100, 5)
```

---

# 6️⃣ `df.columns`

---

## 📌 কাজ কী?

সব column এর নাম দেখায়।

## 🧾 Syntax

```python
df.columns
```

## 💻 Example

```python
print(df.columns)
```

---

# 7️⃣ `df.index`

---

## 📌 কাজ কী?

Row index দেখায়।

## 🧾 Syntax

```python
df.index
```

## 💻 Example

```python
print(df.index)
```

---

# 8️⃣ `df.dtypes`

---

## 📌 কাজ কী?

প্রতিটি column এর data type দেখায়।

## 🧾 Syntax

```python
df.dtypes
```

## 💻 Example

```python
print(df.dtypes)
```

---

# 9️⃣ `df.value_counts()`

---

## 📌 কাজ কী?

কোনো column এর unique value কয়বার আছে তা গণনা করে।

## 🧾 Syntax

```python
df["column_name"].value_counts()
```

## 💻 Example

```python
print(df["Gender"].value_counts())
```

---

# 🔟 `df.nunique()`

---

## 📌 কাজ কী?

প্রতিটি column এ কয়টি unique value আছে তা দেখায়।

## 🧾 Syntax

```python
df.nunique()
```

## 💻 Example

```python
print(df.nunique())
```

---

# 🎯 Group 3 Summary

| Function     | কাজ                 |
| ------------ | ------------------- |
| head         | প্রথম row           |
| tail         | শেষ row             |
| info         | ডেটা summary        |
| describe     | Statistical summary |
| shape        | Row & Column সংখ্যা |
| columns      | Column list         |
| index        | Row index           |
| dtypes       | Data type           |
| value_counts | Frequency           |
| nunique      | Unique count        |

---



# ✅ 🔷 GROUP 4: Data Selection & Indexing

এই group এর function গুলো দিয়ে আমরা নির্দিষ্ট row/column নির্বাচন করতে পারি।

ফরম্যাট আগের মতোই 👇

---

# 1️⃣ `df.loc[]`

---

## 📌 কাজ কী?

Label ভিত্তিক row ও column নির্বাচন করে।

## 🧾 Syntax

```python
df.loc[row_label, column_label]
```

## 🔍 Parameter

| Parameter    | কাজ                 |
| ------------ | ------------------- |
| row_label    | row এর নাম বা index |
| column_label | column এর নাম       |

## 💻 Example

```python
import pandas as pd

data = {"Name": ["Rahim", "Karim", "Hasan"],
        "Age": [22, 25, 23]}

df = pd.DataFrame(data)

# Row index 1 দেখানো
print(df.loc[1])

# নির্দিষ্ট কলাম
print(df.loc[1, "Age"])
```

---

# 2️⃣ `df.iloc[]`

---

## 📌 কাজ কী?

Position ভিত্তিক (সংখ্যা দিয়ে) row/column নির্বাচন করে।

## 🧾 Syntax

```python
df.iloc[row_position, column_position]
```

## 🔍 Parameter

| Parameter       | কাজ               |
| --------------- | ----------------- |
| row_position    | row এর অবস্থান    |
| column_position | column এর অবস্থান |

## 💻 Example

```python
# প্রথম row
print(df.iloc[0])

# প্রথম row এর দ্বিতীয় কলাম
print(df.iloc[0, 1])
```

---

# 3️⃣ `df.at[]`

---

## 📌 কাজ কী?

একটি নির্দিষ্ট cell দ্রুত access করে (label ভিত্তিক)।

## 🧾 Syntax

```python
df.at[row_label, column_label]
```

## 💻 Example

```python
print(df.at[0, "Name"])
```

---

# 4️⃣ `df.iat[]`

---

## 📌 কাজ কী?

একটি নির্দিষ্ট cell দ্রুত access করে (position ভিত্তিক)।

## 🧾 Syntax

```python
df.iat[row_position, column_position]
```

## 💻 Example

```python
print(df.iat[0, 1])
```

---

# 5️⃣ `df.filter()`

---

## 📌 কাজ কী?

নির্দিষ্ট column বা label filter করে।

## 🧾 Syntax

```python
df.filter(items=["col1", "col2"])
```

## 💻 Example

```python
print(df.filter(items=["Name"]))
```

---

# 6️⃣ `df.query()`

---

## 📌 কাজ কী?

Condition ব্যবহার করে row filter করে।

## 🧾 Syntax

```python
df.query("condition")
```

## 💻 Example

```python
print(df.query("Age > 22"))
```

---

# 🎯 Group 4 Summary

| Function | কাজ                        |
| -------- | -------------------------- |
| loc      | Label ভিত্তিক selection    |
| iloc     | Position ভিত্তিক selection |
| at       | Single cell (label)        |
| iat      | Single cell (position)     |
| filter   | Column filter              |
| query    | Condition দিয়ে filter      |

---


# ✅ 🔷 GROUP 5: Data Cleaning Functions

এই function গুলো দিয়ে ডেটা পরিষ্কার (clean), পরিবর্তন (modify), এবং ঠিক (fix) করা হয়।

ফরম্যাট আগের মতোই থাকবে 👇

---

# 1️⃣ `df.isnull()`

---

## 📌 কাজ কী?

DataFrame এ কোথায় null (NaN) আছে তা দেখায়।

## 🧾 Syntax

```python
df.isnull()
```

## 💻 Example

```python
import pandas as pd
import numpy as np

data = {"Name": ["Rahim", "Karim", None],
        "Age": [22, np.nan, 23]}

df = pd.DataFrame(data)

print(df.isnull())
```

---

# 2️⃣ `df.notnull()`

---

## 📌 কাজ কী?

কোথায় null নেই তা দেখায়।

## 🧾 Syntax

```python
df.notnull()
```

## 💻 Example

```python
print(df.notnull())
```

---

# 3️⃣ `df.dropna()`

---

## 📌 কাজ কী?

যে row-তে null আছে তা মুছে দেয়।

## 🧾 Syntax

```python
df.dropna(axis=0, inplace=False)
```

## 🔍 Parameter

| Parameter | কাজ                           |
| --------- | ----------------------------- |
| axis      | 0=row, 1=column               |
| inplace   | True দিলে মূল df পরিবর্তন হবে |

## 💻 Example

```python
print(df.dropna())
```

---

# 4️⃣ `df.fillna()`

---

## 📌 কাজ কী?

Null value পূরণ করে।

## 🧾 Syntax

```python
df.fillna(value)
```

## 💻 Example

```python
df["Age"] = df["Age"].fillna(0)
print(df)
```

---

# 5️⃣ `df.replace()`

---

## 📌 কাজ কী?

নির্দিষ্ট value পরিবর্তন করে।

## 🧾 Syntax

```python
df.replace(old_value, new_value)
```

## 💻 Example

```python
df.replace("Rahim", "Rafi")
```

---

# 6️⃣ `df.drop()`

---

## 📌 কাজ কী?

Row বা column মুছে দেয়।

## 🧾 Syntax

```python
df.drop(labels, axis=0)
```

## 🔍 Parameter

| Parameter | কাজ                  |
| --------- | -------------------- |
| labels    | কোন row/column মুছবে |
| axis      | 0=row, 1=column      |

## 💻 Example

```python
df.drop("Age", axis=1)
```

---

# 7️⃣ `df.rename()`

---

## 📌 কাজ কী?

Column বা index এর নাম পরিবর্তন করে।

## 🧾 Syntax

```python
df.rename(columns={"old": "new"})
```

## 💻 Example

```python
df.rename(columns={"Name": "Full_Name"})
```

---

# 8️⃣ `df.astype()`

---

## 📌 কাজ কী?

Data type পরিবর্তন করে।

## 🧾 Syntax

```python
df.astype({"column": "datatype"})
```

## 💻 Example

```python
df["Age"] = df["Age"].astype("int")
```

---

# 9️⃣ `df.duplicated()`

---

## 📌 কাজ কী?

Duplicate row চিহ্নিত করে।

## 🧾 Syntax

```python
df.duplicated()
```

## 💻 Example

```python
print(df.duplicated())
```

---

# 🔟 `df.drop_duplicates()`

---

## 📌 কাজ কী?

Duplicate row মুছে দেয়।

## 🧾 Syntax

```python
df.drop_duplicates()
```

## 💻 Example

```python
df = df.drop_duplicates()
```

---

# 🎯 Group 5 Summary

| Function        | কাজ                |
| --------------- | ------------------ |
| isnull          | Null চেক           |
| notnull         | Null নয়            |
| dropna          | Null row remove    |
| fillna          | Null পূরণ          |
| replace         | Value পরিবর্তন     |
| drop            | Row/Column remove  |
| rename          | নাম পরিবর্তন       |
| astype          | Data type পরিবর্তন |
| duplicated      | Duplicate চেক      |
| drop_duplicates | Duplicate remove   |

---


# ✅ 🔷 GROUP 6: Sorting Functions

এই group এর function দিয়ে ডেটা সাজানো (sort) করা হয়।

ফরম্যাট আগের মতোই 👇

---

# 1️⃣ `df.sort_values()`

---

## 📌 কাজ কী?

নির্দিষ্ট column অনুযায়ী ডেটা ascending বা descending order এ সাজায়।

---

## 🧾 Common Syntax

```python
df.sort_values(by="column_name", ascending=True)
```

---

## 🔍 গুরুত্বপূর্ণ Parameter

| Parameter   | কাজ                                     |
| ----------- | --------------------------------------- |
| by          | কোন column অনুযায়ী sort করবে            |
| ascending   | True = ছোট থেকে বড়, False = বড় থেকে ছোট |
| inplace     | True দিলে মূল df পরিবর্তন হবে           |
| na_position | null উপরে বা নিচে থাকবে                 |

---

## 💻 Code Example

```python
import pandas as pd

data = {
    "Name": ["Rahim", "Karim", "Hasan"],
    "Age": [22, 25, 23]
}

df = pd.DataFrame(data)

# Age অনুযায়ী ascending sort
print(df.sort_values(by="Age"))

# Descending sort
print(df.sort_values(by="Age", ascending=False))
```

---

## 🔎 Multiple Column দিয়ে Sort

```python
df.sort_values(by=["Age", "Name"])
```

---

# 2️⃣ `df.sort_index()`

---

## 📌 কাজ কী?

Index অনুযায়ী ডেটা সাজায়।

---

## 🧾 Syntax

```python
df.sort_index(ascending=True)
```

---

## 🔍 Parameter

| Parameter | কাজ                           |
| --------- | ----------------------------- |
| ascending | True = ছোট থেকে বড়            |
| inplace   | True দিলে মূল df পরিবর্তন হবে |

---

## 💻 Example

```python
df = df.sort_index(ascending=False)
print(df)
```

---

# 🎯 Group 6 Summary

| Function    | কাজ                 |
| ----------- | ------------------- |
| sort_values | Column অনুযায়ী sort |
| sort_index  | Index অনুযায়ী sort  |

---



# ✅ 🔷 GROUP 7: Aggregation & Statistics Functions

এই function গুলো দিয়ে আমরা ডেটার গাণিতিক বিশ্লেষণ (mean, sum, correlation ইত্যাদি) করতে পারি।

ফরম্যাট আগের মতোই 👇

---

# 1️⃣ `df.mean()`

---

## 📌 কাজ কী?

Numeric column গুলোর গড় (mean) বের করে।

---

## 🧾 Common Syntax

```python
df.mean()
```

---

## 🔍 গুরুত্বপূর্ণ Parameter

| Parameter    | কাজ                           |
| ------------ | ----------------------------- |
| axis         | 0 = column wise, 1 = row wise |
| numeric_only | শুধু numeric column নেবে      |

---

## 💻 Code Example

```python
import pandas as pd

data = {"Age": [20, 25, 30],
        "Salary": [20000, 30000, 40000]}

df = pd.DataFrame(data)

print(df.mean())
```

---

# 2️⃣ `df.median()`

---

## 📌 কাজ কী?

মধ্যম মান (median) বের করে।

---

## 🧾 Syntax

```python
df.median()
```

---

## 💻 Example

```python
print(df.median())
```

---

# 3️⃣ `df.mode()`

---

## 📌 কাজ কী?

সবচেয়ে বেশি বার আসা মান বের করে।

---

## 🧾 Syntax

```python
df.mode()
```

---

## 💻 Example

```python
data = {"A": [1,2,2,3]}
df = pd.DataFrame(data)

print(df.mode())
```

---

# 4️⃣ `df.sum()`

---

## 📌 কাজ কী?

সব মান যোগ করে।

---

## 🧾 Syntax

```python
df.sum()
```

---

## 💻 Example

```python
print(df.sum())
```

---

# 5️⃣ `df.min()`

---

## 📌 কাজ কী?

সর্বনিম্ন মান বের করে।

---

## 🧾 Syntax

```python
df.min()
```

---

## 💻 Example

```python
print(df.min())
```

---

# 6️⃣ `df.max()`

---

## 📌 কাজ কী?

সর্বোচ্চ মান বের করে।

---

## 🧾 Syntax

```python
df.max()
```

---

## 💻 Example

```python
print(df.max())
```

---

# 7️⃣ `df.std()`

---

## 📌 কাজ কী?

Standard Deviation বের করে।

---

## 🧾 Syntax

```python
df.std()
```

---

## 💻 Example

```python
print(df.std())
```

---

# 8️⃣ `df.var()`

---

## 📌 কাজ কী?

Variance বের করে।

---

## 🧾 Syntax

```python
df.var()
```

---

## 💻 Example

```python
print(df.var())
```

---

# 9️⃣ `df.corr()`

---

## 📌 কাজ কী?

Numeric column গুলোর মধ্যে correlation বের করে।

---

## 🧾 Syntax

```python
df.corr()
```

---

## 💻 Example

```python
print(df.corr())
```

---

# 🔟 `df.cov()`

---

## 📌 কাজ কী?

Covariance matrix বের করে।

---

## 🧾 Syntax

```python
df.cov()
```

---

## 💻 Example

```python
print(df.cov())
```

---

# 1️⃣1️⃣ `df.count()`

---

## 📌 কাজ কী?

Null ছাড়া কয়টি মান আছে তা গুনে।

---

## 🧾 Syntax

```python
df.count()
```

---

## 💻 Example

```python
print(df.count())
```

---

# 🎯 Group 7 Summary

| Function | কাজ                |
| -------- | ------------------ |
| mean     | গড়                 |
| median   | মধ্যম মান          |
| mode     | সবচেয়ে বেশি মান    |
| sum      | যোগফল              |
| min      | সর্বনিম্ন          |
| max      | সর্বোচ্চ           |
| std      | Standard deviation |
| var      | Variance           |
| corr     | Correlation        |
| cov      | Covariance         |
| count    | Non-null count     |

---


# ✅ 🔷 GROUP 8: Grouping Functions

এই group এর function দিয়ে আমরা data group করে analysis করতে পারি।
এটা Data Analysis এ সবচেয়ে গুরুত্বপূর্ণ অংশগুলোর একটি।

ফরম্যাট আগের মতোই 👇

---

# 1️⃣ `df.groupby()`

---

## 📌 কাজ কী?

নির্দিষ্ট column অনুযায়ী data group করে।

---

## 🧾 Common Syntax

```python
df.groupby("column_name")
```

---

## 🔍 গুরুত্বপূর্ণ Parameter

| Parameter | কাজ                           |
| --------- | ----------------------------- |
| by        | কোন column অনুযায়ী group করবে |
| as_index  | Group key index হবে কি না     |
| sort      | Group key sort করবে কি না     |

---

## 💻 Code Example

```python
import pandas as pd

data = {
    "Department": ["IT", "HR", "IT", "HR"],
    "Salary": [50000, 40000, 60000, 45000]
}

df = pd.DataFrame(data)

grouped = df.groupby("Department")
print(grouped.mean())
```

---

# 2️⃣ `df.agg()`

---

## 📌 কাজ কী?

একাধিক aggregation function একসাথে প্রয়োগ করতে দেয়।

---

## 🧾 Syntax

```python
df.groupby("column").agg(["mean", "sum"])
```

---

## 🔍 Parameter

| Parameter     | কাজ                       |
| ------------- | ------------------------- |
| function list | কোন কোন aggregation লাগবে |

---

## 💻 Example

```python
result = df.groupby("Department").agg(["mean", "sum"])
print(result)
```

---

# 3️⃣ `df.transform()`

---

## 📌 কাজ কী?

Group অনুযায়ী calculation করে কিন্তু original shape বজায় রাখে।

---

## 🧾 Syntax

```python
df.groupby("column")["target"].transform("mean")
```

---

## 💻 Example

```python
df["Dept_Avg"] = df.groupby("Department")["Salary"].transform("mean")
print(df)
```

---

# 4️⃣ `df.pivot()`

---

## 📌 কাজ কী?

Data reshape করে pivot table তৈরি করে।

---

## 🧾 Syntax

```python
df.pivot(index="col1", columns="col2", values="col3")
```

---

## 💻 Example

```python
data = {
    "Name": ["A", "A", "B", "B"],
    "Year": [2020, 2021, 2020, 2021],
    "Sales": [100, 150, 200, 250]
}

df = pd.DataFrame(data)

pivot_table = df.pivot(index="Name", columns="Year", values="Sales")
print(pivot_table)
```

---

# 5️⃣ `df.pivot_table()`

---

## 📌 কাজ কী?

Pivot এর মতো, কিন্তু aggregation support করে।

---

## 🧾 Syntax

```python
df.pivot_table(index="col1", values="col2", aggfunc="mean")
```

---

## 💻 Example

```python
pivot_table = df.pivot_table(index="Name", values="Sales", aggfunc="mean")
print(pivot_table)
```

---

# 6️⃣ `pd.crosstab()`

---

## 📌 কাজ কী?

দুইটি categorical column এর frequency table তৈরি করে।

---

## 🧾 Syntax

```python
pd.crosstab(df["col1"], df["col2"])
```

---

## 💻 Example

```python
data = {
    "Gender": ["M", "F", "M", "F", "M"],
    "Result": ["Pass", "Pass", "Fail", "Pass", "Fail"]
}

df = pd.DataFrame(data)

table = pd.crosstab(df["Gender"], df["Result"])
print(table)
```

---

# 🎯 Group 8 Summary

| Function    | কাজ                    |
| ----------- | ---------------------- |
| groupby     | Data group             |
| agg         | Multiple aggregation   |
| transform   | Same shape calculation |
| pivot       | Reshape table          |
| pivot_table | Aggregated pivot       |
| crosstab    | Frequency table        |

---


# ✅ 🔷 GROUP 9: Merge / Join Functions

এই group এর function দিয়ে আমরা একাধিক DataFrame একসাথে যুক্ত (combine) করতে পারি।

ফরম্যাট আগের মতোই 👇

---

# 1️⃣ `pd.concat()`

---

## 📌 কাজ কী?

একাধিক DataFrame row-wise বা column-wise যুক্ত করে।

---

## 🧾 Common Syntax

```python
pd.concat([df1, df2], axis=0)
```

---

## 🔍 গুরুত্বপূর্ণ Parameter

| Parameter    | কাজ                           |
| ------------ | ----------------------------- |
| [df1, df2]   | কোন DataFrame গুলো যুক্ত হবে  |
| axis         | 0 = row-wise, 1 = column-wise |
| ignore_index | নতুন index তৈরি করবে          |

---

## 💻 Code Example

```python
import pandas as pd

df1 = pd.DataFrame({"Name": ["Rahim", "Karim"], "Age": [22, 25]})
df2 = pd.DataFrame({"Name": ["Hasan"], "Age": [23]})

result = pd.concat([df1, df2], ignore_index=True)

print(result)
```

---

# 2️⃣ `pd.merge()`

---

## 📌 কাজ কী?

SQL JOIN এর মতো দুইটি DataFrame key অনুযায়ী যুক্ত করে।

---

## 🧾 Common Syntax

```python
pd.merge(df1, df2, on="column_name", how="inner")
```

---

## 🔍 গুরুত্বপূর্ণ Parameter

| Parameter | কাজ                          |
| --------- | ---------------------------- |
| on        | কোন column অনুযায়ী join হবে  |
| how       | inner / left / right / outer |
| left_on   | left df এর key               |
| right_on  | right df এর key              |

---

## 💻 Code Example

```python
df1 = pd.DataFrame({
    "ID": [1, 2, 3],
    "Name": ["Rahim", "Karim", "Hasan"]
})

df2 = pd.DataFrame({
    "ID": [1, 2, 4],
    "Salary": [50000, 60000, 70000]
})

merged = pd.merge(df1, df2, on="ID", how="inner")

print(merged)
```

---

## 🔎 Different JOIN Types

| how   | কাজ                 |
| ----- | ------------------- |
| inner | মিল থাকা row        |
| left  | left সব + matching  |
| right | right সব + matching |
| outer | সব row              |

---

# 3️⃣ `df.join()`

---

## 📌 কাজ কী?

Index ভিত্তিক join করে।

---

## 🧾 Common Syntax

```python
df1.join(df2)
```

---

## 🔍 গুরুত্বপূর্ণ Parameter

| Parameter | কাজ                          |
| --------- | ---------------------------- |
| on        | কোন column দিয়ে join করবে    |
| how       | left / right / inner / outer |

---

## 💻 Code Example

```python
df1 = pd.DataFrame({
    "Name": ["Rahim", "Karim"],
    "Age": [22, 25]
})

df2 = pd.DataFrame({
    "Salary": [50000, 60000]
})

result = df1.join(df2)

print(result)
```

---

# 🎯 Group 9 Summary

| Function | কাজ                |
| -------- | ------------------ |
| concat   | Row/Column যুক্ত   |
| merge    | SQL join           |
| join     | Index ভিত্তিক join |

---



# ✅ 🔷 GROUP 10: Apply & Function Operations

এই group এর function দিয়ে আমরা custom function প্রয়োগ করতে পারি DataFrame বা Series-এর উপর।

ফরম্যাট আগের মতোই 👇

---

# 1️⃣ `df.apply()`

---

## 📌 কাজ কী?

Row বা Column এর উপর custom function প্রয়োগ করে।

---

## 🧾 Common Syntax

```python id="t6p2lw"
df.apply(function, axis=0)
```

---

## 🔍 গুরুত্বপূর্ণ Parameter

| Parameter   | কাজ                           |
| ----------- | ----------------------------- |
| function    | যে function প্রয়োগ করবে       |
| axis        | 0 = column wise, 1 = row wise |
| result_type | expand / reduce               |

---

## 💻 Code Example

```python id="d8t4yz"
import pandas as pd

data = {
    "A": [1, 2, 3],
    "B": [4, 5, 6]
}

df = pd.DataFrame(data)

# Column wise sum
print(df.apply(sum))

# Row wise sum
print(df.apply(sum, axis=1))
```

---

## 🔎 Custom Function Example

```python id="c1h8rk"
def square(x):
    return x * x

print(df["A"].apply(square))
```

---

# 2️⃣ `df.applymap()`

---

## 📌 কাজ কী?

পুরো DataFrame এর প্রতিটি element-এ function প্রয়োগ করে।

---

## 🧾 Common Syntax

```python id="uyj8i3"
df.applymap(function)
```

---

## 💻 Code Example

```python id="h3u1zw"
def double(x):
    return x * 2

print(df.applymap(double))
```

---

# 3️⃣ `df.map()`

---

## 📌 কাজ কী?

Series (একটি column) এর প্রতিটি element-এ function প্রয়োগ করে।

---

## 🧾 Common Syntax

```python id="n5r0ak"
df["column"].map(function_or_dict)
```

---

## 🔍 গুরুত্বপূর্ণ Parameter

| Parameter | কাজ             |
| --------- | --------------- |
| function  | custom function |
| dict      | value mapping   |

---

## 💻 Code Example (Function)

```python id="c1r5qe"
print(df["A"].map(lambda x: x * 10))
```

---

## 💻 Code Example (Dictionary Mapping)

```python id="9yx4n0"
data = {"Grade": ["A", "B", "A", "C"]}
df = pd.DataFrame(data)

grade_map = {"A": "Excellent", "B": "Good", "C": "Average"}

print(df["Grade"].map(grade_map))
```

---

# 🎯 Group 10 Summary

| Function | কাজ                         |
| -------- | --------------------------- |
| apply    | Row/Column-wise function    |
| applymap | পুরো DataFrame element-wise |
| map      | Series element-wise         |

---



# ✅ 🔷 GROUP 10: Apply & Function Operations

এই group এর function দিয়ে আমরা custom function প্রয়োগ করতে পারি DataFrame বা Series-এর উপর।

ফরম্যাট আগের মতোই 👇

---

# 1️⃣ `df.apply()`

---

## 📌 কাজ কী?

Row বা Column এর উপর custom function প্রয়োগ করে।

---

## 🧾 Common Syntax

```python id="t6p2lw"
df.apply(function, axis=0)
```

---

## 🔍 গুরুত্বপূর্ণ Parameter

| Parameter   | কাজ                           |
| ----------- | ----------------------------- |
| function    | যে function প্রয়োগ করবে       |
| axis        | 0 = column wise, 1 = row wise |
| result_type | expand / reduce               |

---

## 💻 Code Example

```python id="d8t4yz"
import pandas as pd

data = {
    "A": [1, 2, 3],
    "B": [4, 5, 6]
}

df = pd.DataFrame(data)

# Column wise sum
print(df.apply(sum))

# Row wise sum
print(df.apply(sum, axis=1))
```

---

## 🔎 Custom Function Example

```python id="c1h8rk"
def square(x):
    return x * x

print(df["A"].apply(square))
```

---

# 2️⃣ `df.applymap()`

---

## 📌 কাজ কী?

পুরো DataFrame এর প্রতিটি element-এ function প্রয়োগ করে।

---

## 🧾 Common Syntax

```python id="uyj8i3"
df.applymap(function)
```

---

## 💻 Code Example

```python id="h3u1zw"
def double(x):
    return x * 2

print(df.applymap(double))
```

---

# 3️⃣ `df.map()`

---

## 📌 কাজ কী?

Series (একটি column) এর প্রতিটি element-এ function প্রয়োগ করে।

---

## 🧾 Common Syntax

```python id="n5r0ak"
df["column"].map(function_or_dict)
```

---

## 🔍 গুরুত্বপূর্ণ Parameter

| Parameter | কাজ             |
| --------- | --------------- |
| function  | custom function |
| dict      | value mapping   |

---

## 💻 Code Example (Function)

```python id="c1r5qe"
print(df["A"].map(lambda x: x * 10))
```

---

## 💻 Code Example (Dictionary Mapping)

```python id="9yx4n0"
data = {"Grade": ["A", "B", "A", "C"]}
df = pd.DataFrame(data)

grade_map = {"A": "Excellent", "B": "Good", "C": "Average"}

print(df["Grade"].map(grade_map))
```

---

# 🎯 Group 10 Summary

| Function | কাজ                         |
| -------- | --------------------------- |
| apply    | Row/Column-wise function    |
| applymap | পুরো DataFrame element-wise |
| map      | Series element-wise         |

---



# ✅ 🔷 GROUP 11: Time Series Functions

এই group এর function গুলো সময়ভিত্তিক ডেটা (date, time, timeline data) বিশ্লেষণে ব্যবহার হয়।

ফরম্যাট আগের মতোই 👇

---

# 1️⃣ `pd.to_datetime()`

---

## 📌 কাজ কী?

String বা অন্যান্য ফরম্যাটের তারিখকে datetime ফরম্যাটে রূপান্তর করে।

---

## 🧾 Common Syntax

```python
pd.to_datetime(arg, format=None)
```

---

## 🔍 গুরুত্বপূর্ণ Parameter

| Parameter | কাজ                           |
| --------- | ----------------------------- |
| arg       | যে কলাম বা value convert করবে |
| format    | তারিখের format                |
| errors    | 'raise', 'coerce', 'ignore'   |

---

## 💻 Code Example

```python
import pandas as pd

data = {"Date": ["2024-01-01", "2024-02-15", "2024-03-10"]}
df = pd.DataFrame(data)

df["Date"] = pd.to_datetime(df["Date"])

print(df.dtypes)
```

---

# 2️⃣ `df.resample()`

---

## 📌 কাজ কী?

Time index অনুযায়ী ডেটা পুনরায় sample করে (daily → monthly, yearly ইত্যাদি)।

⚠ ব্যবহার করতে হলে index অবশ্যই datetime হতে হবে।

---

## 🧾 Syntax

```python
df.resample("M").mean()
```

---

## 🔍 Parameter

| Code | মানে    |
| ---- | ------- |
| 'D'  | Daily   |
| 'M'  | Monthly |
| 'Y'  | Yearly  |

---

## 💻 Example

```python
data = {
    "Date": pd.date_range("2024-01-01", periods=6, freq="D"),
    "Sales": [100, 120, 130, 90, 150, 110]
}

df = pd.DataFrame(data)
df.set_index("Date", inplace=True)

monthly_avg = df.resample("M").mean()

print(monthly_avg)
```

---

# 3️⃣ `df.shift()`

---

## 📌 কাজ কী?

Data সামনে বা পিছনে সরায় (Lag / Lead তৈরি করতে)।

---

## 🧾 Syntax

```python
df.shift(periods=1)
```

---

## 🔍 Parameter

| Parameter | কাজ           |
| --------- | ------------- |
| periods   | কয় step সরাবে |

---

## 💻 Example

```python
data = {"Sales": [100, 120, 130, 90]}
df = pd.DataFrame(data)

df["Previous_Sales"] = df["Sales"].shift(1)

print(df)
```

---

# 4️⃣ `df.rolling()`

---

## 📌 কাজ কী?

Moving window calculation করে (moving average ইত্যাদি)।

---

## 🧾 Syntax

```python
df.rolling(window=3).mean()
```

---

## 🔍 Parameter

| Parameter | কাজ                      |
| --------- | ------------------------ |
| window    | কয়টি row নিয়ে হিসাব করবে |

---

## 💻 Example

```python
df["Moving_Avg"] = df["Sales"].rolling(window=2).mean()

print(df)
```

---

# 5️⃣ `df.expanding()`

---

## 📌 কাজ কী?

Cumulative calculation করে (start থেকে বর্তমান পর্যন্ত)।

---

## 🧾 Syntax

```python
df.expanding().mean()
```

---

## 💻 Example

```python
df["Cumulative_Avg"] = df["Sales"].expanding().mean()

print(df)
```

---

# 🎯 Group 11 Summary

| Function    | কাজ                    |
| ----------- | ---------------------- |
| to_datetime | Date convert           |
| resample    | Time aggregation       |
| shift       | Lag/Lead               |
| rolling     | Moving calculation     |
| expanding   | Cumulative calculation |

---



# ✅ 🔷 GROUP 12: String Operations

এই group এর function গুলো String (text) data নিয়ে কাজ করার জন্য ব্যবহার হয়।
এগুলো সাধারণত ব্যবহার করা হয়:

✔ Text cleaning
✔ NLP preprocessing
✔ Feature engineering

⚠ মনে রাখবে:
String function ব্যবহার করতে হলে:

```python
df["column"].str
```

---

# 1️⃣ `df["col"].str.lower()`

---

## 📌 কাজ কী?

সব text ছোট হাতের অক্ষরে (lowercase) রূপান্তর করে।

---

## 🧾 Common Syntax

```python
df["column"].str.lower()
```

---

## 💻 Code Example

```python
import pandas as pd

data = {"Name": ["RAHIM", "Karim", "Hasan"]}
df = pd.DataFrame(data)

df["Name"] = df["Name"].str.lower()

print(df)
```

---

# 2️⃣ `df["col"].str.upper()`

---

## 📌 কাজ কী?

সব text বড় হাতের অক্ষরে (uppercase) রূপান্তর করে।

---

## 🧾 Syntax

```python
df["column"].str.upper()
```

---

## 💻 Example

```python
df["Name"] = df["Name"].str.upper()
print(df)
```

---

# 3️⃣ `df["col"].str.contains()`

---

## 📌 কাজ কী?

নির্দিষ্ট শব্দ আছে কি না তা চেক করে (True/False দেয়)।

---

## 🧾 Syntax

```python
df["column"].str.contains("word")
```

---

## 💻 Example

```python
data = {"Text": ["I love Python", "Java is good", "Python is easy"]}
df = pd.DataFrame(data)

print(df["Text"].str.contains("Python"))
```

---

# 4️⃣ `df["col"].str.replace()`

---

## 📌 কাজ কী?

String এর ভেতরের অংশ পরিবর্তন করে।

---

## 🧾 Syntax

```python
df["column"].str.replace("old", "new")
```

---

## 💻 Example

```python
df["Text"] = df["Text"].str.replace("Python", "AI")
print(df)
```

---

# 5️⃣ `df["col"].str.split()`

---

## 📌 কাজ কী?

String কে নির্দিষ্ট separator দিয়ে ভাগ করে।

---

## 🧾 Syntax

```python
df["column"].str.split(" ")
```

---

## 💻 Example

```python
df["Words"] = df["Text"].str.split(" ")
print(df)
```

---

# 🔎 Extra Useful String Functions

| Function         | কাজ                       |
| ---------------- | ------------------------- |
| str.strip()      | সামনে/পিছনের space remove |
| str.len()        | String length             |
| str.startswith() | শুরুতে আছে কি না          |
| str.endswith()   | শেষে আছে কি না            |

---

# 🎯 Group 12 Summary

| Function | কাজ            |
| -------- | -------------- |
| lower    | ছোট হাত        |
| upper    | বড় হাত         |
| contains | শব্দ আছে কি না |
| replace  | শব্দ বদল       |
| split    | ভাগ করা        |
| strip    | space remove   |
| len      | দৈর্ঘ্য        |

---

# 
