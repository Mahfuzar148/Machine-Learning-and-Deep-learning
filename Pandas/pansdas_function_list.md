
### 📂 I/O & Data Creation

* `read_csv()` – CSV থেকে ডেটা লোড
* `read_excel()` – Excel ফাইল ইম্পোর্ট
* `read_json()` – JSON ফাইল লোড
* `to_csv()` – DataFrame → CSV সেভ
* `to_excel()` – Excel ফরম্যাটে সেভ
* `to_numpy()` – DataFrame → NumPy array
* `DataFrame()` – নতুন ডেটাফ্রেম তৈরি

---

### 🔍 Data Inspection

* `head()`, `tail()` – প্রথম/শেষ কয়েকটি রেকর্ড
* `info()` – সারাংশ তথ্য
* `describe()` – গড়, std, min, max
* `shape`, `columns`, `dtypes` – গঠন ও টাইপ
* `memory_usage()` – মেমোরি ব্যবহার
* `sample()` – র‍্যান্ডম স্যাম্পল

---

### 🧼 Missing / Cleaning

* `isnull()`, `notnull()` – null detection
* `dropna()` – null বাদ
* `fillna()` – missing পূরণ
* `replace()` – ভ্যালু পরিবর্তন
* `astype()` – টাইপ কনভার্সন
* `drop_duplicates()` – ডুপ্লিকেট বাদ
* `duplicated()` – ডুপ্লিকেট ডিটেক্ট
* `clip()` – outlier trim

---

### 🔎 Filtering & Selection

* `loc[]`, `iloc[]` – label vs position select
* `query()` – SQL style ফিল্টার
* `isin()` – membership চেক
* `between()` – range ফিল্টার
* `where()` – condition apply
* `mask()` – where-এর বিপরীত
* `at[]`, `iat[]` – single cell access

---

### 🔁 Transformations

* `rename()` – column/row নাম বদল
* `drop()` – column বাদ
* `map()` – Series value mapping
* `apply()` – row/column wise ফাংশন
* `applymap()` – cell-wise ফাংশন
* `pipe()` – chaining function

---

### 📊 Aggregation / Summary

* `groupby()` – group-wise stats
* `agg()` – একাধিক ফাংশন একসাথে
* `mean()`, `sum()`, `median()` – basic stats
* `std()`, `var()`, `count()` – dispersion stats
* `value_counts()` – frequency
* `nunique()` – unique count

---

### 🧮 Cumulative & Rolling Stats

* `cumsum()`, `cumprod()`, `cummax()` – ধাপে ধাপে যোগ/গুণ
* `diff()` – পার্থক্য
* `rolling()` – moving window stats
* `expanding()` – cumulative expanding stats

---

### 🔄 Reshaping / Merge

* `melt()` – wide → long
* `pivot()`, `pivot_table()` – long → wide
* `stack()`, `unstack()` – reshaping
* `concat()` – join multiple dfs
* `merge()` – SQL-style join
* `join()` – index-based join
* `append()` – এক্সটেন্ড

---

### ⏳ Date & Time

* `to_datetime()` – string → datetime
* `date_range()` – time index তৈরি
* `resample()` – frequency change
* `dt` accessor – `.dt.year`, `.dt.month`
* `shift()`, `tshift()` – lag/lead

---

### 🧠 Index Handling

* `set_index()`, `reset_index()` – index হ্যান্ডলিং
* `reindex()` – custom index অ্যাপ্লাই
* `sort_index()` – index অনুযায়ী sort
* `sort_values()` – value অনুযায়ী sort

---

### 📈 Plotting & Visual

* `plot()` – matplotlib backend
* `hist()`, `box()`, `line()` – ডিরেক্ট প্লট
* `plot.bar()`, `plot.pie()` – chart-specific plot

---

### 🧪 Misc / Advance

* `eval()` – efficient expression eval
* `insert()` – column insert
* `pop()` – column remove + return
* `memory_usage()` – মেমোরি এনালাইসিস
* `equals()` – df equality check
* `from_dict()`, `from_records()` – dict থেকে df

---


