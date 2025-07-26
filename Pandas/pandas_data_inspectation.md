
---

## 📊 **Data Inspection Functions in Pandas**

Let’s assume we are working with a dataset named `sample missing data.csv`.

### ✅ 1. **Load the Dataset**

```python
import pandas as pd

# Load data
df = pd.read_csv("sample missing data.csv")
```

---

## 1️⃣ `head()` – প্রথম কয়েকটি রেকর্ড

```python
print(df.head())        # Default: first 5 rows
print(df.head(3))       # First 3 rows
```

🧠 **ব্যাখ্যা**: ডেটাসেটের শুরু থেকে কিছু রেকর্ড দেখতে ব্যবহার করা হয়। এটা খুবই কাজের যখন ডেটা স্ট্রাকচার বুঝতে চাই।

---

## 2️⃣ `tail()` – শেষ কয়েকটি রেকর্ড

```python
print(df.tail())        # Default: last 5 rows
print(df.tail(2))       # Last 2 rows
```

🧠 **ব্যাখ্যা**: শেষের দিকের রেকর্ডগুলো দেখতে চাইলে এই ফাংশন ব্যবহৃত হয়।

---

## 3️⃣ `info()` – সারাংশ তথ্য

```python
print(df.info())
```

🧠 **ব্যাখ্যা**:

* কোন কোন কলামে কতগুলো non-null ভ্যালু আছে
* ডেটাটাইপ কী (int, float, object)
* মোট কতগুলো রেকর্ড ও কলাম আছে
  ➡️ ডেটাসেটের **health checkup** বলা যেতে পারে।

---

## 4️⃣ `describe()` – গড়, std, min, max ইত্যাদি

```python
print(df.describe())
```

🧠 **ব্যাখ্যা**:

* শুধু সংখ্যাসূচক কলামের জন্য summary stats দেয়
* যেমন: mean (গড়), std (স্ট্যান্ডার্ড ডিভিয়েশন), min, max, 25%, 50%, 75%

---

## 5️⃣ `shape` – রো ও কলামের সংখ্যা

```python
print(df.shape)
```

🧠 **ব্যাখ্যা**: ডেটাফ্রেমে **(row, column)** কতগুলো আছে — সেটা tuple আকারে জানায়।

---

## 6️⃣ `columns` – কলামের নাম

```python
print(df.columns)
```

🧠 **ব্যাখ্যা**: ডেটাসেটের সব কলামের নাম দেখায়।

---

## 7️⃣ `dtypes` – প্রতিটি কলামের ডেটা টাইপ

```python
print(df.dtypes)
```

🧠 **ব্যাখ্যা**: কোন কলামটা `int`, কোনটা `float`, আর কোনটা `object` টাইপ — সেটা জানায়।

---

## 8️⃣ `memory_usage()` – মেমোরি ব্যবহার

```python
print(df.memory_usage())
```

🧠 **ব্যাখ্যা**: প্রতিটি কলাম কত KB বা Bytes মেমোরি নিচ্ছে সেটা জানায়। এটা **পারফরম্যান্স অপটিমাইজেশনে** কাজে লাগে।

---

## 9️⃣ `sample()` – র‍্যান্ডম স্যাম্পল

```python
print(df.sample(3))      # Random 3 rows
print(df.sample(frac=0.2))  # 20% sample
```

🧠 **ব্যাখ্যা**: ডেটাসেট থেকে **র‍্যান্ডমভাবে রেকর্ড** নেওয়ার জন্য ব্যবহৃত হয়।
➡️ টেস্টিং, যাচাই বা ভিজ্যুয়ালাইজেশনের জন্য বেশ উপকারী।

---

## ✅ Summary Table

| Function         | কাজ                               |
| ---------------- | --------------------------------- |
| `head()`         | প্রথম কয়েকটি রেকর্ড দেখায়         |
| `tail()`         | শেষ দিকের রেকর্ড দেখায়            |
| `info()`         | সারসংক্ষেপ, null, টাইপ, রো-কাউন্ট |
| `describe()`     | গড়, std, min/max – পরিসংখ্যান     |
| `shape`          | মোট রো ও কলাম সংখ্যা              |
| `columns`        | কলামের নাম                        |
| `dtypes`         | কোন কলামের টাইপ কী                |
| `memory_usage()` | কোন কলাম কত মেমোরি নিচ্ছে         |
| `sample()`       | র‍্যান্ডম রো/স্যাম্পল             |

---

---

## ✅ Full Code for Data Inspection in Pandas

```python
import pandas as pd

# ✅ Load dataset (make sure the CSV file is in the same directory)
df = pd.read_csv("sample missing data.csv")

# 1️⃣ Show the first few records
print("📌 First 5 Rows:")
print(df.head())  # By default shows first 5 rows

print("\n📌 First 3 Rows:")
print(df.head(3))  # First 3 rows

# 2️⃣ Show the last few records
print("\n📌 Last 5 Rows:")
print(df.tail())  # Last 5 rows

print("\n📌 Last 2 Rows:")
print(df.tail(2))  # Last 2 rows

# 3️⃣ General info about the dataset
print("\n📌 Dataset Info:")
print(df.info())

# 4️⃣ Summary statistics for numeric columns
print("\n📌 Descriptive Statistics:")
print(df.describe())

# 5️⃣ Shape of the DataFrame (rows, columns)
print("\n📌 Shape of DataFrame:")
print(df.shape)

# 6️⃣ List all column names
print("\n📌 Column Names:")
print(df.columns)

# 7️⃣ Data types of each column
print("\n📌 Data Types:")
print(df.dtypes)

# 8️⃣ Memory usage of each column
print("\n📌 Memory Usage:")
print(df.memory_usage())

# 9️⃣ Random samples from the data
print("\n📌 Random 3 Sample Rows:")
print(df.sample(3))  # Random 3 rows

print("\n📌 Random 20% of the data:")
print(df.sample(frac=0.2))  # 20% of data sampled randomly
```

---

## 📂 How to Use This

1. Save this code to a `.py` file (e.g., `inspect_data.py`)
2. Make sure `sample missing data.csv` is in the same folder.
3. Run the script in terminal/command prompt or any IDE.

---

## ✅ Output Preview (Partial)

Depending on your file, this will print:

* First/last few rows
* Data shape (e.g., `(28, 12)`)
* Columns like `Name`, `Age`, `Department`, etc.
* Data types (e.g., `int64`, `float64`, `object`)
* Summary stats like `mean`, `std`, `min`, `max`
* Memory footprint in bytes
* Random records

---


