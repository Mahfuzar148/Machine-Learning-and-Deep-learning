
---

## 🧼 Missing / Cleaning Data in Pandas

We'll use a sample dataset to demonstrate:

```python
import pandas as pd

# Sample data with missing and duplicate values
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Bob', 'Eve', None],
    'Age': [25, 30, None, 22, 30, None, 29],
    'Salary': [50000, 54000, 61000, None, 54000, 62000, None]
}

df = pd.DataFrame(data)
print("🔹 Original DataFrame:")
print(df)
```

---

### 1️⃣ `isnull()` – Null values চিহ্নিত করে

```python
print(df.isnull())         # Null থাকলে True
print(df.isnull().sum())   # প্রতিটি কলামে কতগুলো null আছে
```

📌 **Syntax**:

```python
df.isnull()
df['column'].isnull()
```

🧠 **ব্যাখ্যা**: কোন সেল null (NaN) সেটা True/False আকারে দেখায়।

---

### 2️⃣ `notnull()` – Non-null values চিহ্নিত করে

```python
print(df[df['Name'].notnull()])  # Name null না হলে সেসব রো দেখাও
```

📌 **Syntax**:

```python
df.notnull()
df['column'].notnull()
```

---

### 3️⃣ `dropna()` – Null বাদ দেওয়ার জন্য

```python
df_drop_rows = df.dropna()               # যেসব রোতে null আছে সেগুলো বাদ
df_drop_cols = df.dropna(axis=1)         # যেসব কলামে null আছে সেগুলো বাদ
df_thresh = df.dropna(thresh=2)          # যেসব রোতে কমপক্ষে 2টি non-null, শুধুমাত্র সেগুলো রাখো
```

📌 **Common Parameters**:

* `axis=0` (default): row বাদ
* `axis=1`: column বাদ
* `thresh=n`: কমপক্ষে nটি non-null থাকলে রাখবে

---

### 4️⃣ `fillna()` – Null পূরণ করতে

```python
df_fill_zero = df.fillna(0)                        # সব null 0 দিয়ে পূরণ
df_fill_method = df.fillna(method='ffill')         # আগের মান দিয়ে পূরণ (forward fill)
df_fill_custom = df.fillna({'Age': 28, 'Salary': df['Salary'].mean()})
```

📌 **Common Parameters**:

* `value`: একক মান বা dict
* `method`: `'ffill'` (forward), `'bfill'` (backward)
* `inplace=True`: স্থায়ীভাবে পরিবর্তন

---

### 5️⃣ `replace()` – নির্দিষ্ট ভ্যালু পরিবর্তন

```python
df_replace = df.replace(54000, 55000)                    # সব 54000 -> 55000
df_replace_multi = df.replace({None: 'Unknown'})         # None -> 'Unknown'
```

📌 **Syntax**:

```python
df.replace(old_value, new_value)
df.replace({column: {old: new}})
```

---

### 6️⃣ `astype()` – টাইপ পরিবর্তন

```python
df['Age'] = df['Age'].fillna(0)     # NaN থাকলে int-এ রূপান্তর সম্ভব নয়
df['Age'] = df['Age'].astype(int)   # float -> int
```

📌 **Syntax**:

```python
df['col'] = df['col'].astype(new_type)
```

---

### 7️⃣ `drop_duplicates()` – ডুপ্লিকেট রো বাদ

```python
df_no_dupes = df.drop_duplicates()
df_no_dupes_name = df.drop_duplicates(subset='Name')
```

📌 **Common Parameters**:

* `subset`: কোন কলাম ধরে ডুপ্লিকেট খোঁজা হবে
* `keep='first'` (default) or `'last'` or `False`

---

### 8️⃣ `duplicated()` – ডুপ্লিকেট ডিটেক্ট

```python
print(df.duplicated())                  # পুরো রো মিললে True
print(df.duplicated(subset='Name'))     # শুধু Name কলাম মিললে
```

📌 **Returns**: Boolean Series (True মানে duplicate)

---

### 9️⃣ `clip()` – Value boundary fix (Outlier trim)

```python
print(df['Age'].clip(lower=20, upper=30))  # Age 20-30 এর বাইরে হলে সেটিকে সীমায় আটকে দাও
```

📌 **Syntax**:

```python
df['col'].clip(lower=min_val, upper=max_val)
```

🧠 **ব্যাখ্যা**: outlier ভ্যালু কাটছাঁট করার জন্য।

---

## ✅ Full Summary in Table

| Function            | কাজ                              | Example                   |
| ------------------- | -------------------------------- | ------------------------- |
| `isnull()`          | Null খুঁজে বের করা               | `df.isnull()`             |
| `notnull()`         | Null না এমন খুঁজে বের করা        | `df.notnull()`            |
| `dropna()`          | Null থাকা রো/কলাম বাদ দেওয়া      | `df.dropna()`             |
| `fillna()`          | Null পূরণ করা                    | `df.fillna(0)`            |
| `replace()`         | নির্দিষ্ট ভ্যালু বদলানো          | `df.replace(54000, 0)`    |
| `astype()`          | ডেটা টাইপ পরিবর্তন               | `df['col'].astype(int)`   |
| `drop_duplicates()` | ডুপ্লিকেট বাদ                    | `df.drop_duplicates()`    |
| `duplicated()`      | ডুপ্লিকেট চিহ্নিত করা            | `df.duplicated()`         |
| `clip()`            | Outlier ভ্যালু সীমায় বেঁধে দেওয়া | `df['col'].clip(10, 100)` |

---

---

### ✅ Full Code: Missing / Cleaning in Pandas

```python
import pandas as pd

# 🔹 Sample dataset with missing and duplicate values
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Bob', 'Eve', None],
    'Age': [25, 30, None, 22, 30, None, 29],
    'Salary': [50000, 54000, 61000, None, 54000, 62000, None]
}

df = pd.DataFrame(data)
print("🔹 Original DataFrame:\n", df)

# 1️⃣ isnull(): Null detection
print("\n🔍 Null Detection (isnull):\n", df.isnull())
print("\n🔢 Null Count Per Column:\n", df.isnull().sum())

# 2️⃣ notnull(): Non-null detection
print("\n✅ Rows where Name is NOT null:\n", df[df['Name'].notnull()])

# 3️⃣ dropna(): Remove missing values
print("\n🚫 Drop rows with any nulls:\n", df.dropna())
print("\n🚫 Drop columns with any nulls:\n", df.dropna(axis=1))
print("\n🚫 Drop rows with less than 2 non-null values:\n", df.dropna(thresh=2))

# 4️⃣ fillna(): Fill missing values
print("\n🧼 Fill all nulls with 0:\n", df.fillna(0))

fill_custom = df.fillna({'Age': df['Age'].mean(), 'Salary': df['Salary'].median(), 'Name': 'Unknown'})
print("\n🧼 Fill with custom values:\n", fill_custom)

print("\n🔁 Forward fill method:\n", df.fillna(method='ffill'))

# 5️⃣ replace(): Replace specific values
print("\n🔁 Replace 54000 with 55000:\n", df.replace(54000, 55000))
print("\n🔁 Replace None with 'Unknown':\n", df.replace({None: 'Unknown'}))

# 6️⃣ astype(): Type conversion (only after filling nulls)
df['Age'] = df['Age'].fillna(0)
df['Age'] = df['Age'].astype(int)
print("\n🔣 Age after type conversion to int:\n", df['Age'])

# 7️⃣ drop_duplicates(): Drop duplicate rows
print("\n🗑️ Drop duplicate rows:\n", df.drop_duplicates())
print("\n🗑️ Drop duplicates based on 'Name':\n", df.drop_duplicates(subset='Name'))

# 8️⃣ duplicated(): Detect duplicates
print("\n🔍 Detect duplicate rows:\n", df.duplicated())
print("\n🔍 Detect duplicates in 'Name' column:\n", df.duplicated(subset='Name'))

# 9️⃣ clip(): Trim outliers
print("\n✂️ Clip Age between 20 and 30:\n", df['Age'].clip(lower=20, upper=30))
```

---

### 🧪 Requirements

Make sure you have Pandas installed:

```bash
pip install pandas
```

---

### 📌 Output Examples (Partial)

```text
🔢 Null Count Per Column:
Name      1
Age       2
Salary    2
dtype: int64

🧼 Fill all nulls with 0:
     Name   Age   Salary
0  Alice  25.0  50000.0
...

🔣 Age after type conversion to int:
0    25
1    30
2     0
...
```

---

