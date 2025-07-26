
---

## 📘 1. **Shape of the DataFrame**

```python
df.shape
```

🔸 **Task**: মোট কতটি row এবং column আছে, তা জানায়
🔸 **Return**: একটি tuple (rows, columns)

---

## 📘 2. **Top N Rows**

```python
df.head(n)
```

🔸 **Parameter**:

* `n`: কতটি row দেখতে চাও (default = 5)

🔸 **Example**:

```python
df.head(10)
```

---

## 📘 3. **Last N Rows**

```python
df.tail(n)
```

🔸 **Parameter**:

* `n`: শেষ দিক থেকে কতটি row

🔸 **Example**:

```python
df.tail(5)
```

---

## 📘 4. **Range of Entries**

```python
df[start:end]
```

🔸 **Explanation**: row slicing এর মাধ্যমে নির্দিষ্ট range এর row দেখা

🔸 **Example**:

```python
df[5:10]
```

---

## 📘 5. **Accessing Columns**

```python
df['column_name']
# অথবা
df.column_name
```

🔸 **Return**: Series type output

🔸 **Example**:

```python
df['Week Day']
```

---

## 📘 6. **Type of Column**

```python
df['column_name'].dtype
```

🔸 **Example**:

```python
df['productivity'].dtype
```

---

## 📊 **Basic Statistical Functions**

| Function           | Syntax                     | Description           |
| ------------------ | -------------------------- | --------------------- |
| Max value          | `df['col'].max()`          | সর্বোচ্চ মান          |
| Min value          | `df['col'].min()`          | সর্বনিম্ন মান         |
| Mean (average)     | `df['col'].mean()`         | গড় মান                |
| Standard Deviation | `df['col'].std()`          | মান বিচ্যুতি          |
| Mode               | `df['col'].mode()`         | সবচেয়ে বেশি পাওয়া মান |
| Count Frequency    | `df['col'].value_counts()` | মানগুলোর গণনা         |
| 25th Percentile    | `df['col'].quantile(0.25)` | প্রথম Quartile (Q1)   |

---

## 🔃 **Value Sorting**

```python
df.sort_values(by='column', ascending=True/False)
```

🔸 **Parameter**:

* `by`: কোন column অনুসারে sort করবে
* `ascending`: True (ascending), False (descending)

🔸 **Example**:

```python
df.sort_values(by='productivity', ascending=False)
```

---

## ⚙️ **Conditional Statements**

```python
df[df['column'] condition]
```

🔸 **Example**:

```python
df[df['productivity'] > 15]
df[df['Week Day'] == 'Friday']
```

---

## 📌 **LOC vs ILOC**

| Function  | Syntax                  | Use Case                         |
| --------- | ----------------------- | -------------------------------- |
| `.loc[]`  | `df.loc[row_index]`     | লেবেল (label) অনুযায়ী row/column |
| `.iloc[]` | `df.iloc[row_position]` | পজিশন অনুযায়ী row/column         |

🔸 **Example**:

```python
df.loc[2]
df.iloc[5]
```

---

## ❓ **Test Questions Answer Implementation**

### Question 1:

```python
df[(df['Week Day'] == 'Friday') & (df['productivity'] > 15)]
```

### Question 2:

```python
df[df['productivity'] > 15]
```

---

## ✅ উপসংহার: Summary Table

| 🔢 Topic        | 📘 Function Used                        |
| --------------- | --------------------------------------- |
| Shape           | `df.shape`                              |
| Top/Bottom Rows | `df.head()`, `df.tail()`                |
| Range Rows      | `df[start:end]`                         |
| Access Column   | `df['col']`, `df.col`                   |
| Type of Column  | `df['col'].dtype`                       |
| Statistics      | `mean()`, `max()`, `min()`, `std()` etc |
| Sort            | `df.sort_values()`                      |
| Filter          | `df[df['col'] > val]`                   |
| Location        | `df.loc[]`, `df.iloc[]`                 |

---

---

## ✅ Full Code Example with All Topics Covered

```python
import pandas as pd

# Sample data for testing
data = {
    "Week Day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    "productivity": [14, 17, 15, 19, 16, 20, 13]
}

# Create DataFrame
df = pd.DataFrame(data)

# 1. Shape of DataFrame
print("Shape of DataFrame:", df.shape)  # (rows, columns)

# 2. First N rows
print("\nTop 3 rows using head:")
print(df.head(3))  # Default n=5 if not provided

# 3. Last N rows
print("\nLast 2 rows using tail:")
print(df.tail(2))

# 4. Range/Slice of rows
print("\nRows from index 2 to 5:")
print(df[2:6])  # Start inclusive, end exclusive

# 5. Accessing a column
print("\nAccessing 'Week Day' column:")
print(df["Week Day"])

# 6. Column Data Type
print("\nData type of 'productivity' column:")
print(df["productivity"].dtype)

# 7. Max value
print("\nMaximum productivity value:")
print(df["productivity"].max())

# 8. Min value
print("\nMinimum productivity value:")
print(df["productivity"].min())

# 9. Mean
print("\nMean of productivity:")
print(df["productivity"].mean())

# 10. Standard Deviation
print("\nStandard deviation of productivity:")
print(df["productivity"].std())

# 11. Mode
print("\nMode of productivity:")
print(df["productivity"].mode())

# 12. Value Counts (Frequency)
print("\nFrequency of productivity values:")
print(df["productivity"].value_counts())

# 13. Quantile (25%)
print("\n25th percentile (Q1) of productivity:")
print(df["productivity"].quantile(0.25))

# 14. Sorting by column
print("\nSorted by productivity descending:")
print(df.sort_values(by="productivity", ascending=False))

# 15. Conditional Selection – Example 1
print("\nAll rows where productivity > 15:")
print(df[df["productivity"] > 15])

# 16. Conditional Selection – Example 2
print("\nRows where Week Day is Friday and productivity > 15:")
print(df[(df["Week Day"] == "Friday") & (df["productivity"] > 15)])

# 17. Using loc[] by index label
print("\nUsing loc to access index 2:")
print(df.loc[2])

# 18. Using iloc[] by position
print("\nUsing iloc to access 5th row:")
print(df.iloc[4])
```

---

## 📌 Quick Recap of Core Concepts:

| 📘 Feature            | 🧠 Function / Syntax                 |
| --------------------- | ------------------------------------ |
| Head/Tail/Shape       | `df.head()`, `df.tail()`, `df.shape` |
| Access Columns        | `df['col']`, `df.col`                |
| Type of Columns       | `df['col'].dtype`                    |
| Basic Statistics      | `max()`, `min()`, `mean()`, etc.     |
| Frequency Counts      | `df['col'].value_counts()`           |
| Conditional Selection | `df[df['col'] > 15]`                 |
| Location Access       | `.loc[]`, `.iloc[]`                  |

---



