
---

## 🧱 1. **Importing and Installing Required Libraries**

```python
!pip install pandas
import pandas as pd
```

* You installed and imported **Pandas**, a powerful library for data analysis and manipulation.
* Pandas provides two key structures:

  * **Series** – 1D labeled array.
  * **DataFrame** – 2D table-like structure.

---

## 🔢 2. **Creating a Pandas Series with Custom Index**

```python
data = [1,2,3,4,5,6,7,8,9,10]
s = pd.Series(data, name='Num', index=['a','b','c','d','e','f','g','h','i','j'])
```

* You created a `Series` named `"Num"` from a list of 10 integers and assigned custom alphabetical indices.

---

## 🔁 3. **Converting Series to DataFrame**

```python
df = pd.DataFrame(s, columns=['NUM'])
```

* Converts the Series `s` to a DataFrame `df` with a column named `'NUM'`.


---

## 🧪 4. **DataFrame Construction with Lists and Tuples (Expanded)**

You explored various ways to construct `Series` and `DataFrame` objects in pandas using different data structures like lists, dictionaries, sets, and tuples.

---

### ✅ 4.1. **From a Dictionary of Lists**

```python
l2 = [1, 2, '', 4]
l3 = [5, 6, 7, 8]

dic = {
    'num1': l2,
    'num2': l3
}

print(pd.Series(dic))
print(pd.DataFrame(dic))
```

**Explanation**:

* When passed to `pd.Series`, the dictionary is treated as a mapping from keys to full lists, producing:

  ```text
  num1    [1, 2, '', 4]
  num2    [5, 6, 7, 8]
  dtype: object
  ```
* When passed to `pd.DataFrame`, each key becomes a column, and pandas aligns rows by index:

  ```text
     num1  num2
  0     1     5
  1     2     6
  2          7
  3     4     8
  ```

---

### ✅ 4.2. **From a Set (unordered and unique)**

```python
data1 = {1, 2, 3, 5, 6}
print(pd.Series(tuple(data1)))
print(pd.DataFrame(data1))
```

**Explanation**:

* Sets are unordered, so results may vary by run. You converted the set to a `tuple` before using it in `pd.Series`, and passed the set directly to `pd.DataFrame`:

  ```text
  0    1
  1    2
  2    3
  3    5
  4    6
  dtype: int64
  ```

  ```text
     0
  0  1
  1  2
  2  3
  3  5
  4  6
  ```

---

### ✅ 4.3. **From a Tuple**

```python
t = (1, 2, 3, 4, 5)
print(type(t))  # Output: <class 'tuple'>
print(pd.Series(t))
print(pd.DataFrame(t))
```

**Explanation**:

* Like a list, a tuple is a one-dimensional array-like structure. It’s easily converted into both Series and DataFrame:

  ```text
  0    1
  1    2
  2    3
  3    4
  4    5
  dtype: int64
  ```

  ```text
     0
  0  1
  1  2
  2  3
  3  4
  4  5
  ```

---

### 🔍 Summary of Data Type Behaviors

| Data Type         | Converted To Series                   | Converted To DataFrame               |
| ----------------- | ------------------------------------- | ------------------------------------ |
| **List**          | Works cleanly                         | Works cleanly                        |
| **Tuple**         | Works cleanly                         | Works cleanly                        |
| **Set**           | Must convert to list/tuple for Series | Works, but order is not preserved    |
| **Dict of lists** | Creates Series of lists               | Columns from keys, values as columns |

---



## 📂 5. **Uploading and Reading a CSV File**

```python
df4 = pd.read_csv('Screen Time Data.csv')
```

* This loads screen time data into a DataFrame `df4`.

---

## 🔍 6. **Exploring DataFrame Properties**

```python
df4.shape          # (28, 12)
df4.info()         # Data types and non-null counts
df4.columns        # List of column names
df4.describe()     # Summary statistics
```

These give a comprehensive view of structure, size, and contents.

---

## 📊 7. **Summary Statistics on 'Total Screen Time '** (Note the trailing space!)

```python
df4['Total Screen Time '].min()      # 52
df4['Total Screen Time '].max()      # 198
df4['Total Screen Time '].mean()     # 113.25
df4['Total Screen Time '].std()      # 43.56
df4['Total Screen Time '].median()   # 111.0
df4['Total Screen Time '].mode()     # [52, 58]
```

* You calculated key descriptive stats: min, max, mean, std, median, and mode.
* Notably, `mode()` returned **two values**: both 52 and 58 occurred most frequently.

---

## 📅 8. **Week Day Mode (Most Frequent Day)**

```python
df4['Week Day'].mode()
```

Returned all weekdays — meaning each appeared **equally** (4 times), so Pandas returns **all**.

---

## 🔝 9. **Sorting the DataFrame by Screen Time**

```python
df4.sort_values('Total Screen Time ', ascending=False, inplace=True)
```

* This sorts the DataFrame by `'Total Screen Time '` in descending order (highest to lowest).
* `inplace=True` modifies `df4` directly (no need to assign).

✅ **Note** (as you mentioned): when `inplace=True`, the function does **not return** anything — using `df4 = df4.sort_values(...)` would make `df4` become `None`.

---

## ✅ Summary

You:

* Practiced Series and DataFrame creation from various data types.
* Explored key Pandas methods (`shape`, `info`, `describe`, `mean`, `mode`, etc.).
* Loaded and analyzed a real dataset.
* Cleaned and sorted data properly.

---


---

## 📄 **🧠 Full Python Code for Screen Time Data Analysis**

```python
# -------------------------------------------
# Step 1: Install and import required libraries
# -------------------------------------------
!pip install pandas
import pandas as pd

# -------------------------------------------
# Step 2: Create Series and DataFrame examples
# -------------------------------------------
# Creating a list and converting it to Series
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
s = pd.Series(data, name='Num', index=['a','b','c','d','e','f','g','h','i','j'])
print("Series s:\n", s)

# Convert Series to DataFrame
df = pd.DataFrame(s, columns=['NUM'])
print("\nDataFrame from Series:\n", df)

# Create DataFrame from list directly
df1 = pd.DataFrame(data, columns=['NUM'])
print("\nDataFrame df1 from list:\n", df1)

# Slicing and checking column selection
print("\nSelecting column NUM as DataFrame:\n", df1[['NUM']])
print("\nType of df1[['NUM']]:", type(df1[['NUM']]))

# -------------------------------------------
# Step 3: Create and test Series/DataFrames from various data types
# -------------------------------------------
l2 = [1, 2, '', 4]
l3 = [5, 6, 7, 8]
dic = {'num1': l2, 'num2': l3}

print("\nSeries from dictionary:\n", pd.Series(dic))
print("\nDataFrame from dictionary:\n", pd.DataFrame(dic))

# From set and tuple
data1 = {1, 2, 3, 5, 6}
print("\nSeries from tuple:\n", pd.Series(tuple(data1)))
print("\nDataFrame from set:\n", pd.DataFrame(data1))

t = (1, 2, 3, 4, 5)
print("\nSeries from tuple:\n", pd.Series(t))
print("\nDataFrame from tuple:\n", pd.DataFrame(t))

# -------------------------------------------
# Step 4: Upload and read CSV file
# -------------------------------------------
# For Google Colab users:
# from google.colab import files
# uploaded = files.upload()  # Use to upload Screen Time Data.csv

# Read the uploaded file
df4 = pd.read_csv('Screen Time Data.csv')
print("\nLoaded DataFrame df4:\n", df4.head())

# -------------------------------------------
# Step 5: Explore data
# -------------------------------------------
print("\nShape of df4:", df4.shape)
row, col = df4.shape
print("Rows:", row)
print("Columns:", col)

print("\nInfo:\n")
df4.info()

print("\nColumns:\n", df4.columns)

# -------------------------------------------
# Step 6: Basic statistics on Total Screen Time
# -------------------------------------------
col = 'Total Screen Time '  # Notice the trailing space
print("\nMin:", df4[col].min())
print("Max:", df4[col].max())
print("Mean:", df4[col].mean())
print("Median:", df4[col].median())
print("Standard Deviation:", df4[col].std())
print("Mode:\n", df4[col].mode())

# -------------------------------------------
# Step 7: Mode of Week Days
# -------------------------------------------
print("\nMode of 'Week Day':\n", df4['Week Day'].mode())

# -------------------------------------------
# Step 8: Describe the dataset
# -------------------------------------------
print("\nDescriptive statistics:\n", df4.describe())

# -------------------------------------------
# Step 9: Viewing portions of data
# -------------------------------------------
print("\nFirst 6 rows:\n", df4.head(6))
print("\nLast 5 rows:\n", df4.tail())
print("\nRows from index 2 to 4:\n", df4[2:5])

# -------------------------------------------
# Step 10: Sorting the data by Total Screen Time
# -------------------------------------------
df4.sort_values('Total Screen Time ', ascending=False, inplace=True)
print("\nSorted by Total Screen Time (descending):\n", df4[['Date', 'Week Day', 'Total Screen Time ']])

# Optional: Save sorted data
# df4.to_csv("Sorted_Screen_Time.csv", index=False)
```

---

## 🧠 What This Code Does

* Builds both `Series` and `DataFrame` from lists, tuples, sets, and dictionaries.
* Reads and analyzes real screen time data.
* Performs statistical analysis like `mean`, `median`, `std`, `mode`.
* Uses sorting and slicing techniques.
* Fully compatible with **Google Colab** and **Jupyter Notebook**.

---



