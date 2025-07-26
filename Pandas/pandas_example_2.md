
---

# 📊 Full Documentation: Data Analysis + Profiling with `pandas` & `ydata_profiling`

---

## ✅ **Step 1: Import Required Libraries**

```python
import pandas as pd
from ydata_profiling import ProfileReport
```

* `pandas`: Used for data manipulation and analysis.
* `ydata_profiling`: Generates an automated EDA (exploratory data analysis) report.

---

## ✅ **Step 2: Load the Dataset**

```python
df = pd.read_csv('sample missing data.csv')
```

* Loads the CSV into a DataFrame `df`.
* Make sure the file is in the **same directory** as your script.

---

## 🧾 **Step 3: View & Slice the Data**

```python
print(df.head())                         # First 5 rows (default)
print(df.loc[0:5])                       # Rows with labels 0 to 5 (inclusive)
print(df.loc[0:5, ['Name', 'Age']])     # Rows 0–5 and columns 'Name' & 'Age'
print(df.loc[0:5, ['Name', 'Age']].head())      # Same, then first 5
print(df.loc[0:5, ['Name', 'Age']].head(3))     # First 3 of selected data
```

* `.loc[]`: Label-based access to rows and columns.
* `head(n)`: Returns the first `n` rows.

```python
print(df.iloc[0:5])
```

* `.iloc[]`: Integer-position based row selection.
* Here, selects rows at index 0 to 4.

---

## 🛠️ **Step 4: Create & Modify a New DataFrame**

```python
dic = {
    'num1': [1, 2, 3, 4],
    'num2': [5, 6, 7, 8]
}
df1 = pd.DataFrame(dic)
print(df1)
```

* Creates a new DataFrame `df1` using a dictionary.

```python
df1['num3'] = [9, 10, 11, 12]
print(df1)
```

* Adds a new column `'num3'`.

```python
df1.loc[len(df1)] = [1, 1, 1]
print(df1)
```

* Adds a new row at the end using `.loc`.

```python
df1.drop(['num1', 'num3'], axis=1, inplace=True)
```

* Drops columns `'num1'` and `'num3'`.

---

## 🔎 **Step 5: Data Inspection on Main Dataset**

```python
df.columns         # Shows all column names
df.head()          # Displays first 5 rows again
```

---

## 📌 **Step 6: Conditional Filtering**

```python
df[(df['Week Day'] == 'Friday') | (df['Productivity'] >= 15)]
```

* Returns rows where:

  * `'Week Day'` is `'Friday'`
  * OR `'Productivity'` is greater than or equal to 15

---

## 🧼 **Step 7: Null Value Handling**

```python
df3 = pd.read_csv('sample missing data.csv')
df3.isnull().sum()
```

* Reloads data to a new DataFrame `df3`.
* `.isnull().sum()` shows how many `NaN` values per column.

```python
df3.info()
df3.fillna(0, inplace=True)
df3.dropna(inplace=True)
```

* `.info()` displays data types and non-null counts.
* `fillna(0)` fills all `NaN` with 0.
* `dropna()` drops all rows with any `NaN`.

---

## 🧠 **Step 8: Fill Missing Data with Specific Statistics**

```python
dic3 = {
    'Age': df3['Age'].mean(),
    'Salary': df3['Salary'].median(),
    'Department': df3['Department'].mode().iloc[0]  # Most frequent value
}
df3.fillna(dic3, inplace=True)
```

* Creates a dictionary with:

  * Mean of `'Age'`
  * Median of `'Salary'`
  * Mode of `'Department'`
* Fills null values with those stats using `fillna`.

---

## 📊 **Step 9: Generate an Automated Profiling Report**

```python
profile = ProfileReport(df, title="Sample Data Profiling Report", explorative=True)
profile.to_file("profiling_report.html")
```

* `ProfileReport(...)` builds a full EDA report.
* `to_file(...)` saves the report as an HTML file.
* Output: `profiling_report.html` opens in your browser with charts, summaries, correlations, missing values, etc.

---

## ✅ Final Output

```python
print("✅ Report generated: profiling_report.html")
```

Confirms successful report generation.

---

## 📁 How to View the HTML Report

* Go to the folder where your script is.
* Find `profiling_report.html`.
* Double-click to open it in your default browser.

---

## ✅ Summary of Functions Used

| Code                        | Purpose                           |
| --------------------------- | --------------------------------- |
| `read_csv()`                | Load CSV file into DataFrame      |
| `head()`, `loc[]`, `iloc[]` | View and slice data               |
| `drop()`, `fillna()`        | Handle columns and missing values |
| `isnull().sum()`            | Count missing values              |
| `ProfileReport()`           | Generate EDA report               |
| `to_file()`                 | Export profiling report as HTML   |


---

## ✅ Full Python Code for Data Analysis and Profiling

```python
# 📦 Import libraries
import pandas as pd
from ydata_profiling import ProfileReport

# 🔹 Load dataset
df = pd.read_csv("sample missing data.csv")

# 🔍 Basic inspection
print("🔸 First few rows of dataset:")
print(df.head())

print("🔸 Rows 0 to 5:")
print(df.loc[0:5])

print("🔸 'Name' and 'Age' columns from rows 0 to 5:")
print(df.loc[0:5, ['Name', 'Age']])

print("🔸 First few rows of selected columns:")
print(df.loc[0:5, ['Name', 'Age']].head())

print("🔸 First 3 rows of selected columns:")
print(df.loc[0:5, ['Name', 'Age']].head(3))

# 🔸 Integer-location based slicing
print("🔸 Rows 0 to 4 using iloc:")
print(df.iloc[0:5])

# 🔸 Create a new DataFrame from dictionary
dic = {
    'num1': [1, 2, 3, 4],
    'num2': [5, 6, 7, 8]
}
df1 = pd.DataFrame(dic)
print("🔸 New DataFrame:")
print(df1)

# 🔸 Add new column
df1['num3'] = [9, 10, 11, 12]
print("🔸 After adding 'num3' column:")
print(df1)

# 🔸 Add a new row to the end
df1.loc[len(df1)] = [1, 1, 1]
print("🔸 After adding a new row at the end:")
print(df1)

# 🔸 Drop selected columns
df1.drop(['num1', 'num3'], axis=1, inplace=True)
print("🔸 After dropping 'num1' and 'num3':")
print(df1)

# 🔸 Show column names
print("🔸 Columns in df:")
print(df.columns)

# 🔸 Filter rows by condition (Week Day is 'Friday' or Productivity >= 15)
filtered_df = df[(df['Week Day'] == 'Friday') | (df['Productivity'] >= 15)]
print("🔸 Filtered DataFrame (Friday or Productivity >= 15):")
print(filtered_df)

# 🔸 Reload dataset to new DataFrame for missing value processing
df3 = pd.read_csv("sample missing data.csv")

# 🔸 Count null values
print("🔸 Null values before cleaning:")
print(df3.isnull().sum())

# 🔸 Info before cleaning
print("🔸 Data info before fill/drop:")
df3.info()

# 🔸 Fill missing values with 0
df3.fillna(0, inplace=True)

# 🔸 Drop rows with any remaining NaN (if any)
df3.dropna(inplace=True)

# 🔸 Null values after cleaning
print("🔸 Null values after fillna and dropna:")
print(df3.isnull().sum())

# 🔸 Fill missing values with specific values (mean, median, mode)
# Reload again to keep original data
df3 = pd.read_csv("sample missing data.csv")

# 🔸 Define imputation rules
dic3 = {
    'Age': df3['Age'].mean(),
    'Salary': df3['Salary'].median(),
    'Department': df3['Department'].mode().iloc[0]
}

# 🔸 Fill missing values based on calculated stats
df3.fillna(dic3, inplace=True)

# 📈 Generate Profiling Report
profile = ProfileReport(df3, title="Sample Data Profiling Report", explorative=True)

# 💾 Save the profiling report to an HTML file
profile.to_file("profiling_report.html")

print("✅ Report generated: profiling_report.html")
```

---

## 📝 Notes:

* Ensure that `"sample missing data.csv"` exists in the same folder.
* `ydata_profiling` must be installed:

  ```bash
  pip install ydata_profiling
  ```
* The profiling report will be saved as `profiling_report.html` — double-click to open it in a browser.

---



