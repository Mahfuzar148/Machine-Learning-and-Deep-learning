
---

# 🧠 `.loc` vs `.iloc` in `pandas`

| Feature          | `.loc[]`                             | `.iloc[]`                                     |
| ---------------- | ------------------------------------ | --------------------------------------------- |
| **Access by**    | **Labels** (row/column names)        | **Integer position** (like list indices)      |
| **Includes end** | Yes – `df.loc[0:2]` includes row `2` | No – `df.iloc[0:2]` includes only `0` and `1` |
| **Use Case**     | When you know row/column names       | When you know row/column numbers (like index) |

---

## ✅ Example Dataset

```python
import pandas as pd

# Sample DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'Age': [25, 30, 35, 40, 45],
    'Department': ['HR', 'Finance', 'IT', 'Marketing', 'Admin']
}
df = pd.DataFrame(data)

print("🔹 Full DataFrame:")
print(df)
```

---

## 🟢 Using `.loc[]` (Label-based access)

```python
print("\n🔹 Rows 1 to 3 (inclusive) using loc:")
print(df.loc[1:3])

print("\n🔹 Specific rows and columns using loc:")
print(df.loc[1:3, ['Name', 'Department']])
```

* Accesses rows with labels 1 to 3 (inclusive).
* Column names are used directly (`'Name'`, `'Department'`).

---

## 🔵 Using `.iloc[]` (Index-based access)

```python
print("\n🔹 Rows 1 to 3 (exclusive) using iloc:")
print(df.iloc[1:3])  # Only row 1 and 2

print("\n🔹 Specific rows and columns using iloc:")
print(df.iloc[1:4, [0, 2]])  # Rows 1 to 3, columns 0 and 2
```

* Accesses rows by **index position** (start inclusive, end exclusive).
* Columns accessed by integer index.

---

## 📌 Full Working Code (Copy & Run)

```python
import pandas as pd

# Step 1: Create DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'Age': [25, 30, 35, 40, 45],
    'Department': ['HR', 'Finance', 'IT', 'Marketing', 'Admin']
}
df = pd.DataFrame(data)

print("🔹 Full DataFrame:")
print(df)

# Step 2: Using loc
print("\n🔹 Rows 1 to 3 using loc (label-based):")
print(df.loc[1:3])

print("\n🔹 Columns Name and Department from rows 1 to 3 using loc:")
print(df.loc[1:3, ['Name', 'Department']])

# Step 3: Using iloc
print("\n🔹 Rows 1 to 3 using iloc (position-based, excludes end):")
print(df.iloc[1:3])  # Row 1 and 2 only

print("\n🔹 Rows 1 to 3 and columns 0 and 2 using iloc:")
print(df.iloc[1:4, [0, 2]])  # Rows 1,2,3 and columns Name, Department
```

---

## 🧠 Summary

| Method  | Used For         | Syntax Example                 |
| ------- | ---------------- | ------------------------------ |
| `.loc`  | Label-based      | `df.loc[1:3, ['Name', 'Age']]` |
| `.iloc` | Integer-position | `df.iloc[1:4, [0, 1]]`         |

---


