
---

## 🎯 **Dataset Setup**

```python
import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [25, 32, 28, 45, 22],
    'Salary': [50000, 60000, 55000, 80000, 48000],
    'Department': ['HR', 'IT', 'IT', 'Finance', 'HR']
}

df = pd.DataFrame(data)
```

This creates the following DataFrame:

|    |    Name | Age | Salary | Department |
| -: | ------: | --: | -----: | ---------: |
|  0 |   Alice |  25 |  50000 |         HR |
|  1 |     Bob |  32 |  60000 |         IT |
|  2 | Charlie |  28 |  55000 |         IT |
|  3 |   David |  45 |  80000 |    Finance |
|  4 |     Eve |  22 |  48000 |         HR |

---

## 🔍 Step-by-Step Filtering & Selection Techniques

---

### 1️⃣ `loc[]` – **Label-based selection**

```python
df.loc[1:3, ['Name', 'Salary']]
```

📌 **Explanation**:

* `1:3` = select rows with index 1 to 3
* `['Name', 'Salary']` = only show these two columns
* Useful when you know **column names** and **row labels**

---

### 2️⃣ `iloc[]` – **Position-based selection**

```python
df.iloc[0:3, 0:2]
```

📌 **Explanation**:

* Select rows at position 0, 1, 2
* Select columns at position 0 and 1 (Name and Age)
* Used when you refer by **integer index**, not name

---

### 3️⃣ `query()` – **SQL-style filter**

```python
df.query("Age > 30")
```

📌 **Explanation**:

* Select rows where Age is greater than 30
* You can use logical expressions just like SQL

Another example:

```python
df.query("Department == 'HR' and Salary > 49000")
```

---

### 4️⃣ `isin()` – **Membership check**

```python
df[df['Department'].isin(['HR', 'IT'])]
```

📌 **Explanation**:

* Filters rows where `'Department'` is either `"HR"` or `"IT"`

---

### 5️⃣ `between()` – **Range filtering**

```python
df[df['Salary'].between(50000, 60000)]
```

📌 **Explanation**:

* Keeps rows where Salary is between 50000 and 60000 (inclusive)

---

### 6️⃣ `where()` – **Keep only condition-matching rows**

```python
df.where(df['Age'] > 30)
```

📌 **Explanation**:

* Keeps rows where Age > 30
* Replaces **other rows with NaN**

---

### 7️⃣ `mask()` – **Opposite of `where()`**

```python
df.mask(df['Age'] > 30)
```

📌 **Explanation**:

* Replaces rows **where Age > 30** with NaN
* Keeps others intact

---

### 8️⃣ `at[]` – **Access a single value (label-based)**

```python
df.at[2, 'Name']
```

📌 **Explanation**:

* Gets the value from row `2`, column `'Name'` → `'Charlie'`
* Very fast for single cell access

---

### 9️⃣ `iat[]` – **Access a single value (position-based)**

```python
df.iat[2, 1]
```

📌 **Explanation**:

* Gets value at row 2, column 1 → `28` (Age of Charlie)

---

## ✅ Summary Table

| Method      | Use Case                            | Based on      |
| ----------- | ----------------------------------- | ------------- |
| `loc[]`     | Select rows/columns by **name**     | Label         |
| `iloc[]`    | Select rows/columns by **position** | Integer Index |
| `query()`   | SQL-like filtering                  | Column names  |
| `isin()`    | Filter by values in a list          | Membership    |
| `between()` | Filter numeric range                | Range         |
| `where()`   | Keep condition-matching rows        | Conditional   |
| `mask()`    | Hide condition-matching rows        | Inverse logic |
| `at[]`      | Fast access by **label**            | Label         |
| `iat[]`     | Fast access by **position**         | Integer Index |

---


```python
import pandas as pd

# Create sample DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [25, 32, 28, 45, 22],
    'Salary': [50000, 60000, 55000, 80000, 48000],
    'Department': ['HR', 'IT', 'IT', 'Finance', 'HR']
}
df = pd.DataFrame(data)

# loc[]: Label-based selection
loc_selection = df.loc[1:3, ['Name', 'Salary']]

# iloc[]: Position-based selection
iloc_selection = df.iloc[0:3, 0:2]

# query(): SQL-style filtering
query_result_1 = df.query("Age > 30")
query_result_2 = df.query("Department == 'HR' and Salary > 49000")

# isin(): Check membership
isin_result = df[df['Department'].isin(['HR', 'IT'])]

# between(): Range filter
between_result = df[df['Salary'].between(50000, 60000)]

# where(): Keep only values that match condition
where_result = df.where(df['Age'] > 30)

# mask(): Replace values that match condition
mask_result = df.mask(df['Age'] > 30)

# at[]: Single cell access (label-based)
at_value = df.at[2, 'Name']

# iat[]: Single cell access (position-based)
iat_value = df.iat[2, 1]

# Output to verify (you may use print or display in notebook)
# Example: print(loc_selection), print(at_value), etc.
```

### Notes:

* This code can run in any Python environment (e.g., VS Code, Jupyter, Google Colab).
* You can optionally `print()` the variables like `loc_selection`, `query_result_1`, etc., to view outputs.
* The last two (`at_value` and `iat_value`) return single scalar values like `'Charlie'` or `28`.


