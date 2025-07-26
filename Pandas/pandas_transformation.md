
---

### ✅ Dataset Setup

```python
import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'Salary': [50000, 60000, 70000]
}
df = pd.DataFrame(data)
```

---

### 1️⃣ `rename()` – Change column or row labels

```python
# Rename columns
df_renamed = df.rename(columns={'Salary': 'Income'})

# Rename index
df_renamed_index = df.rename(index={0: 'A', 1: 'B'})
```

📘 **Use case**: To make column names more meaningful or standardized.

---

### 2️⃣ `drop()` – Remove column(s) or row(s)

```python
# Drop a column
df_dropped_col = df.drop(columns='Salary')

# Drop a row
df_dropped_row = df.drop(index=1)
```

📘 **Use case**: Clean up unnecessary data.

---

### 3️⃣ `map()` – Element-wise transformation on a Series

```python
# Create a new column using map on Age
df['Age_Group'] = df['Age'].map(lambda x: 'Young' if x < 30 else 'Adult')
```

📘 **Use case**: Apply simple functions or mappings to a single column.

---

### 4️⃣ `apply()` – Row/column-wise function application

```python
# Apply function to column
df['Double Salary'] = df['Salary'].apply(lambda x: x * 2)

# Apply row-wise (axis=1)
df['Name_Age'] = df.apply(lambda row: f"{row['Name']}-{row['Age']}", axis=1)
```

📘 **Use case**: Apply complex logic row-by-row or column-by-column.

---

### 5️⃣ `applymap()` – Cell-wise function on entire DataFrame

```python
# Only works on DataFrames (not Series)
df_numeric = df[['Age', 'Salary']]
df_upper = df_numeric.applymap(lambda x: x + 10)
```

📘 **Use case**: Apply cell-level transformations on numeric/textual DataFrames.

---

### 6️⃣ `pipe()` – Chain custom functions

```python
# Define a custom function
def add_bonus(df, bonus):
    df['Bonus'] = df['Salary'] * bonus
    return df

# Use pipe to apply it
df_transformed = df.pipe(add_bonus, bonus=0.10)
```

📘 **Use case**: Clean and readable function chaining in pipelines.

---

### ✅ Summary Table

| Method     | Scope        | Typical Use                          |
| ---------- | ------------ | ------------------------------------ |
| rename()   | column/index | Rename labels                        |
| drop()     | column/row   | Remove unnecessary parts             |
| map()      | Series       | Element-wise operation               |
| apply()    | Series/DF    | Column/row-wise transformation       |
| applymap() | DataFrame    | Cell-wise operation                  |
| pipe()     | DataFrame    | Functional pipelines (modular logic) |

---
```python
import pandas as pd

# Sample dataset
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'Salary': [50000, 60000, 70000]
}
df = pd.DataFrame(data)

# 1. rename(): Rename column and index
df_renamed = df.rename(columns={'Salary': 'Income'})
df_renamed_index = df.rename(index={0: 'A', 1: 'B'})

# 2. drop(): Drop a column and a row
df_dropped_col = df.drop(columns='Salary')
df_dropped_row = df.drop(index=1)

# 3. map(): Apply logic to one column
df['Age_Group'] = df['Age'].map(lambda x: 'Young' if x < 30 else 'Adult')

# 4. apply(): Column-wise and row-wise function
df['Double_Salary'] = df['Salary'].apply(lambda x: x * 2)
df['Name_Age'] = df.apply(lambda row: f"{row['Name']}-{row['Age']}", axis=1)

# 5. applymap(): Cell-wise transformation (only on DataFrame, not Series)
df_numeric = df[['Age', 'Salary']]
df_modified = df_numeric.applymap(lambda x: x + 10)

# 6. pipe(): Functional chaining
def add_bonus(dataframe, bonus):
    dataframe['Bonus'] = dataframe['Salary'] * bonus
    return dataframe

df_piped = df.pipe(add_bonus, bonus=0.1)
```

### Notes:

* You can insert `print()` statements or view results in Jupyter/Colab using `display(df_renamed)` etc.
* All examples are standalone and safe to run one after another.
* `applymap()` works **only** on entire DataFrames (not individual Series).
* `pipe()` helps modularize transformation logic.



