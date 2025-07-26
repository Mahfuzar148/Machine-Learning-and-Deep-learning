
---

## 📘 Class Topics + Matching Pandas Functions

| SL | 📌 Task Description                                  | 🛠️ Required Pandas Functions                 |
| -- | ---------------------------------------------------- | --------------------------------------------- |
| 1  | Create a list of data                                | `list()` (Python built-in)                    |
| 2  | Create a series from the list                        | `pd.Series()`                                 |
| 3  | Create a dataframe from the series                   | `pd.DataFrame()`                              |
| 4  | DataFrame convert to series                          | `df['col']` or `df.squeeze()`                 |
| 5  | Create a series for 2 integer columns                | `pd.Series(zip(col1, col2))`                  |
| 6  | Create dataframe from list and rename columns        | `pd.DataFrame(data, columns=[...])`           |
| 7  | Create a series from the dataframe                   | `df['col']` or `df.iloc[:, 0]`                |
| 8  | Create a dictionary                                  | `dict()` (Python built-in)                    |
| 9  | Create a series from the dictionary                  | `pd.Series(dict)`                             |
| 10 | Set to series and dataframe                          | `pd.Series(set)`, `pd.DataFrame(set)`         |
| 11 | Tuple to series and dataframe                        | `pd.Series(tuple)`, `pd.DataFrame(tuple)`     |
| 12 | Data Types                                           | `.dtypes`                                     |
| 13 | Check dataframe and its datatypes                    | `df.info()`, `df.dtypes`                      |
| 14 | Numeric Data Types                                   | `int`, `float`, `.select_dtypes()`            |
| 15 | Object                                               | dtype `'object'`, text/strings                |
| 16 | Datetime                                             | `pd.to_datetime()`                            |
| 17 | Create list of dates                                 | `dates = ['2023-01-01', ...]`                 |
| 18 | Create a dataframe with a datetime column            | `pd.DataFrame({'date': pd.to_datetime(...)})` |
| 19 | Check data type of date column                       | `df['col'].dtype`, `df.info()`                |
| 20 | `pd.to_datetime(dates)`                              | `pd.to_datetime()`                            |
| 21 | Series convert (unspecified; likely list <-> series) | `pd.Series()`, `.tolist()`                    |
| 22 | Read Excel `.xlsx`                                   | `pd.read_excel()`                             |
| 23 | Read CSV                                             | `pd.read_csv()`                               |

---

## ✅ পূর্ণ কোড উদাহরণ (সংক্ষেপে)

```python
import pandas as pd

# 1. List of data
data = [10, 20, 30, 40]

# 2. Series from list
s = pd.Series(data)

# 3. DataFrame from series
df = pd.DataFrame(s, columns=['numbers'])

# 4. DataFrame to series
s2 = df['numbers']

# 5. Series from 2 integer columns
col1 = [1, 2, 3]
col2 = [4, 5, 6]
s3 = pd.Series(list(zip(col1, col2)))

# 6. DataFrame from list + rename columns
df2 = pd.DataFrame([[1, 2], [3, 4]], columns=['A', 'B'])

# 7. Series from DataFrame
s4 = df2['A']

# 8. Dictionary
d = {'a': 100, 'b': 200}

# 9. Series from dict
s5 = pd.Series(d)

# 10. Set to series & df
s6 = pd.Series({1, 2, 3})
df3 = pd.DataFrame({1, 2, 3})

# 11. Tuple to series & df
s7 = pd.Series((10, 20, 30))
df4 = pd.DataFrame([(1, 'a'), (2, 'b')])

# 12 & 13. Data types
print(df4.dtypes)
print(df4.info())

# 14-16. Numeric, Object, Datetime
df5 = pd.DataFrame({
    'num': [1.5, 2.6],
    'text': ['hello', 'world'],
    'date': pd.to_datetime(['2023-01-01', '2023-02-01'])
})
print(df5.dtypes)

# 17. List of dates
dates = ['2023-03-01', '2023-03-02']

# 18. DataFrame with datetime column
df6 = pd.DataFrame({'my_date': pd.to_datetime(dates)})

# 19. Check datetime dtype
print(df6.dtypes)

# 20. to_datetime
converted = pd.to_datetime(dates)

# 21. Series conversion
lst = s.tolist()
new_s = pd.Series(lst)

# 22. Read Excel
# df_excel = pd.read_excel('file.xlsx')

# 23. Read CSV
# df_csv = pd.read_csv('file.csv')
```

---


এই পাঁচটি ক্লাস টাস্কের জন্য নিচে প্রতিটি ধাপের **Python/Pandas কোড উদাহরণ সহ ব্যাখ্যা** দেওয়া হলো:

---

### ✅ 1. **List তৈরি করা**

```python
# Python built-in list
my_list = [10, 20, 30, 40]
```

📌 সাধারণ Python লিস্ট — এটা হলো ডেটা স্টোর করার প্রাথমিক কন্টেইনার।

---

### ✅ 2. **List থেকে Series তৈরি করা**

```python
import pandas as pd

my_series = pd.Series(my_list)
print(my_series)
```

📌 `pd.Series()` দিয়ে যেকোনো list কে Series-এ রূপান্তর করা যায় — যেটা single-dimensional labeled array।

---

### ✅ 3. **Series থেকে DataFrame তৈরি করা**

```python
df_from_series = pd.DataFrame(my_series, columns=["Numbers"])
print(df_from_series)
```

📌 এক বা একাধিক Series থেকে DataFrame বানাতে পারো — column নামে `columns=["..."]` দিতে হয়।

---

### ✅ 4. **DataFrame থেকে Series এ রূপান্তর করা**

```python
series_from_df = df_from_series["Numbers"]
print(series_from_df)
```

📌 কোনো একটি কলামকে Series হিসেবে আনতে চাইলে `df["col_name"]` বা `df.col_name` ব্যবহার করা হয়।

---

### ✅ 5. **দুইটি integer কলাম থেকে Series বানানো**

```python
col1 = [1, 2, 3]
col2 = [4, 5, 6]

paired_series = pd.Series(list(zip(col1, col2)))
print(paired_series)
```

📌 `zip()` দিয়ে দুইটি লিস্টকে টিউপল হিসেবে পেয়ার করে Series বানানো হয়।

---

---

### ✅ **Task 6: List থেকে DataFrame তৈরি ও কলাম রিনেম করা**

```python
import pandas as pd

# একটি 2D লিস্ট
data = [[1, 'Alice'], [2, 'Bob'], [3, 'Charlie']]

# DataFrame তৈরি এবং কলাম নাম সেট করা
df = pd.DataFrame(data, columns=['ID', 'Name'])
print(df)
```

📌 এখানে আমরা `columns` প্যারামিটার ব্যবহার করে কলাম নাম 'ID' এবং 'Name' দিয়েছি।

---

### ✅ **Task 7: DataFrame থেকে একটি Series তৈরি করা**

```python
# 'Name' কলামটি Series হিসেবে আলাদা করছি
name_series = df['Name']
print(name_series)
```

📌 DataFrame থেকে যেকোনো একটি কলাম নির্বাচন করলে তা Series আকারে রিটার্ন হয়।

---

### ✅ **Task 8: একটি Dictionary তৈরি করা**

```python
# Python built-in dictionary
student_info = {'name': 'David', 'age': 21, 'dept': 'CSE'}
print(student_info)
```

📌 সাধারণ একটি `dict` object যা Pandas Series বা DataFrame-এ কনভার্ট করা যাবে।

---

### ✅ **Task 9: Dictionary থেকে Series তৈরি করা**

```python
# Dictionary থেকে Series
student_series = pd.Series(student_info)
print(student_series)
```

📌 Dictionary-এর key গুলো Series-এর index হয়, আর value হয় Series-এর values।

---

### ✅ **Task 10: Set থেকে Series ও DataFrame তৈরি করা**

```python
# একটি সেট
num_set = {10, 20, 30}

# Set → list → Series
set_series = pd.Series(list(num_set))
print("Set to Series:\n", set_series)

# Set → list → DataFrame
set_df = pd.DataFrame(list(num_set), columns=['Numbers'])
print("Set to DataFrame:\n", set_df)
```

📌 যেহেতু set এর মধ্যে order থাকে না, তাই প্রথমে list-এ রূপান্তর করে Series বা DataFrame বানাতে হয়।

---

### ✅ **Task 11: Tuple থেকে Series ও DataFrame তৈরি করা**

```python
# Tuple
my_tuple = (100, 200, 300)

# Tuple → Series
tuple_series = pd.Series(my_tuple)
print("Tuple to Series:\n", tuple_series)

# Tuple of tuples → DataFrame
tuple_data = [(1, 'X'), (2, 'Y')]
tuple_df = pd.DataFrame(tuple_data, columns=['ID', 'Label'])
print("Tuple to DataFrame:\n", tuple_df)
```

📌 একক tuple হলে তা Series হয়; একাধিক tuple হলে তা DataFrame হিসাবে ব্যাখ্যা করা হয়।

---

### ✅ **Task 12: DataFrame এর Data Types দেখা**

```python
# DataFrame তৈরি
df12 = pd.DataFrame({
    'int_col': [1, 2],
    'float_col': [1.5, 2.5],
    'text_col': ['a', 'b']
})

# ডেটা টাইপ দেখা
print(df12.dtypes)
```

📌 `.dtypes` দিয়ে প্রতিটি কলামের ডেটা টাইপ দেখা যায়।

---

### ✅ **Task 13: DataFrame এবং তার ডেটাটাইপ বিশ্লেষণ করা**

```python
# DataFrame এর সারাংশ (non-null, dtype, memory info)
df12.info()
```

📌 `.info()` DataFrame-এর structure, dtype ও null/non-null value দেখায়।

---

### ✅ **Task 14: Numeric Data Types বুঝা ও আলাদা করা**

```python
# Numeric column সিলেক্ট করা
numeric_df = df12.select_dtypes(include=['number'])
print(numeric_df)
```

📌 শুধুমাত্র সংখ্যা টাইপ কলামগুলো বের করতে `select_dtypes(include=['number'])` ব্যবহার করা হয়।

---

### ✅ **Task 15: Object টাইপ (Text/String) কলাম দেখা**

```python
# Text বা object টাইপ ফিল্টার করা
text_df = df12.select_dtypes(include=['object'])
print(text_df)
```


---

### ✅ **Task 16: Datetime টাইপ ব্যবহার করা**

```python
import pandas as pd

# Date string → datetime
date_str = ['2023-01-01', '2023-02-01']
date_series = pd.to_datetime(date_str)
print(date_series)
```

📌 `pd.to_datetime()` ব্যবহার করলে string টাইপের তারিখগুলোকে `datetime64` টাইপে রূপান্তর করা যায়।

---

### ✅ **Task 17: List of Dates তৈরি করা**

```python
# একটি তারিখের লিস্ট তৈরি করা
date_list = ['2023-05-01', '2023-05-02', '2023-05-03']
print(date_list)
```

📌 সাধারণভাবে Python এর list ব্যবহার করে date string রাখা হয়।

---

### ✅ **Task 18: Datetime column সহ DataFrame তৈরি করা**

```python
# List থেকে datetime রূপান্তর করে DataFrame
df18 = pd.DataFrame({'Dates': pd.to_datetime(date_list)})
print(df18)
```

📌 এখানে আমরা সরাসরি DataFrame বানানোর সময় `pd.to_datetime()` ব্যবহার করে কলাম তৈরি করেছি।

---

### ✅ **Task 19: একটি column এর ডেটা টাইপ চেক করা**

```python
# dtype দেখা
print("Column dtype:", df18['Dates'].dtype)
```

📌 `dtype` দেখালে বোঝা যায় কোনো কলাম `datetime64[ns]` কিনা।

---

### ✅ **Task 20: `pd.to_datetime(dates)` ব্যবহার করা**

```python
# আবারও তারিখ কনভার্ট করছি datetime-এ
converted_dates = pd.to_datetime(date_list)
print(converted_dates)
```

📌 এটি `Task 16` এর মতই, আলাদা করে অন্য সময় বা অন্য list কে datetime রূপে নেওয়ার জন্য।

---

### ✅ **Task 21: Series কনভার্সন (list ↔ series)**

```python
# Series → list
s21 = pd.Series([10, 20, 30])
list21 = s21.tolist()
print("Series to List:", list21)

# List → Series
s21_new = pd.Series(list21)
print("List to Series:\n", s21_new)
```

📌 `.tolist()` দিয়ে Series → List, আবার `pd.Series()` দিয়ে List → Series করা যায়।

---

### ✅ **Task 22: Excel ফাইল (.xlsx) পড়া**

```python
# Excel ফাইল থেকে পড়া (file.xlsx আগে থাকতে হবে)
# df_excel = pd.read_excel("file.xlsx")
# print(df_excel)
```

📌 এটি কাজ করতে হলে `openpyxl` বা `xlrd` ইনস্টল থাকতে হবে এবং ফাইলটি একই ডিরেক্টরিতে থাকতে হবে।

---

### ✅ **Task 23: CSV ফাইল পড়া**

```python
# CSV ফাইল থেকে DataFrame তৈরি
# df_csv = pd.read_csv("file.csv")
# print(df_csv)
```

📌 `.csv` ফাইল সাধারণত data science-এ সবচেয়ে বেশি ব্যবহৃত হয়।

---





