
---

## ✅ Setup: Sample Dataset

```python
import pandas as pd

data = {
    'Department': ['Sales', 'Sales', 'HR', 'HR', 'IT', 'IT'],
    'Employee': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank'],
    'Salary': [50000, 60000, 45000, 47000, 70000, 72000],
    'Experience': [2, 3, 1, 2, 5, 6]
}

df = pd.DataFrame(data)
```

---

## 📊 Aggregation & Summary

### 1️⃣ `groupby()` – Group-wise statistics

```python
grouped = df.groupby('Department')['Salary'].mean()
```

📌 **Explanation**: Calculates average salary for each department.

---

### 2️⃣ `agg()` – Multiple aggregations at once

```python
agg_result = df.groupby('Department').agg({
    'Salary': ['mean', 'sum', 'count'],
    'Experience': ['min', 'max']
})
```

📌 **Explanation**: Runs multiple functions (`mean`, `sum`, `count`) on `Salary`, and `min`, `max` on `Experience`.

---

### 3️⃣ Basic Stats: `mean()`, `sum()`, `median()`

```python
avg_salary = df['Salary'].mean()
total_salary = df['Salary'].sum()
median_exp = df['Experience'].median()
```

📌 **Explanation**: Basic overall summaries.

---

### 4️⃣ Dispersion: `std()`, `var()`, `count()`

```python
salary_std = df['Salary'].std()
salary_var = df['Salary'].var()
row_count = df['Salary'].count()
```

📌 **Explanation**: Standard deviation, variance, and total count.

---

### 5️⃣ `value_counts()` – Frequency count

```python
department_counts = df['Department'].value_counts()
```

📌 **Explanation**: How many times each department appears.

---

### 6️⃣ `nunique()` – Number of unique values

```python
unique_departments = df['Department'].nunique()
```

📌 **Explanation**: Number of unique departments.

---

## 🧮 Cumulative & Rolling Statistics

### 1️⃣ `cumsum()`, `cumprod()`, `cummax()`

```python
df['Cumulative_Salary'] = df['Salary'].cumsum()
df['Cumulative_Product'] = df['Experience'].cumprod()
df['Cumulative_Max'] = df['Salary'].cummax()
```

📌 **Explanation**:

* `cumsum()`: Running total
* `cumprod()`: Running multiplication
* `cummax()`: Running maximum

---

### 2️⃣ `diff()` – Row-wise difference

```python
df['Salary_Change'] = df['Salary'].diff()
```

📌 **Explanation**: Difference in salary from previous row.

---

### 3️⃣ `rolling()` – Moving window statistics

```python
df['Rolling_Mean'] = df['Salary'].rolling(window=2).mean()
```

📌 **Explanation**: Moving average of salary over a window of 2 rows.

---

### 4️⃣ `expanding()` – Cumulative stats over all previous rows

```python
df['Expanding_Mean'] = df['Salary'].expanding().mean()
```

📌 **Explanation**: Expanding mean (includes all rows up to that point).

---

## ✅ Summary Table

| Method           | Description                    |
| ---------------- | ------------------------------ |
| `groupby()`      | Group-based calculations       |
| `agg()`          | Multiple stats at once         |
| `mean()`, etc.   | Standard statistics            |
| `value_counts()` | Frequency of each value        |
| `nunique()`      | Unique value count             |
| `cumsum()`       | Running sum                    |
| `cumprod()`      | Running product                |
| `cummax()`       | Running max                    |
| `diff()`         | Row difference                 |
| `rolling()`      | Moving window stats            |
| `expanding()`    | Cumulative stats over all rows |

---

✅ Aggregation & summary operations
✅ Cumulative & rolling statistics
✅ With a **sample dataset**

---

### ✅ Full Python Code

```python
import pandas as pd

# Sample dataset
data = {
    'Department': ['Sales', 'Sales', 'HR', 'HR', 'IT', 'IT'],
    'Employee': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank'],
    'Salary': [50000, 60000, 45000, 47000, 70000, 72000],
    'Experience': [2, 3, 1, 2, 5, 6]
}

df = pd.DataFrame(data)

# 1. Group-wise mean salary
grouped = df.groupby('Department')['Salary'].mean()

# 2. Multiple aggregations with agg()
agg_result = df.groupby('Department').agg({
    'Salary': ['mean', 'sum', 'count'],
    'Experience': ['min', 'max']
})

# 3. Basic statistics
avg_salary = df['Salary'].mean()
total_salary = df['Salary'].sum()
median_exp = df['Experience'].median()

# 4. Dispersion stats
salary_std = df['Salary'].std()
salary_var = df['Salary'].var()
row_count = df['Salary'].count()

# 5. Value counts (frequency)
department_counts = df['Department'].value_counts()

# 6. Unique count
unique_departments = df['Department'].nunique()

# 7. Cumulative stats
df['Cumulative_Salary'] = df['Salary'].cumsum()
df['Cumulative_Product'] = df['Experience'].cumprod()
df['Cumulative_Max'] = df['Salary'].cummax()

# 8. Difference between rows
df['Salary_Change'] = df['Salary'].diff()

# 9. Rolling mean (window = 2)
df['Rolling_Mean'] = df['Salary'].rolling(window=2).mean()

# 10. Expanding mean
df['Expanding_Mean'] = df['Salary'].expanding().mean()

# Display final DataFrame
print(df)

# Optionally export to CSV
# df.to_csv("aggregated_stats_output.csv", index=False)
```

---

