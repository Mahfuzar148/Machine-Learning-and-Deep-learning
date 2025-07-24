
---

# 🧾 Python `collections.Counter` Full Documentation (with examples)

`Counter` হলো Python-এর built-in module `collections` এর একটি class, যা **hashable items-এর frequency (গণনা)** রাখতে সাহায্য করে। এটি মূলত dictionary-এর মতই কাজ করে — কিন্তু element গুলোর **count** রাখে।

---

## 🔹 Basic Syntax

```python
from collections import Counter

counter = Counter(iterable)
```

* `iterable`: list, tuple, string বা অন্য কোনো iterable
* Return করে: `Counter` object যা dict-এর মতো কাজ করে

---

## 🔹 Example 1: Count items in a list

```python
from collections import Counter

nums = [1, 2, 2, 3, 3, 3, 4]
count = Counter(nums)
print(count)
```

**Output:**

```
Counter({3: 3, 2: 2, 1: 1, 4: 1})
```

👉 অর্থাৎ `3` এসেছে ৩ বার, `2` এসেছে ২ বার, ইত্যাদি।

---

## 🔹 Example 2: Count characters in a string

```python
s = "banana"
c = Counter(s)
print(c)
```

**Output:**

```
Counter({'a': 3, 'n': 2, 'b': 1})
```

---

## 🔹 Example 3: Access individual counts

```python
print(c['a'])  # 3
print(c['z'])  # 0 (missing items return 0, not error)
```

---

## 🔹 Example 4: Most Common Elements

```python
words = ['apple', 'banana', 'apple', 'orange', 'banana', 'apple']
c = Counter(words)

print(c.most_common(2))
```

**Output:**

```
[('apple', 3), ('banana', 2)]
```

👉 শীর্ষ ২টি সবচেয়ে বেশি আসা element ও তাদের সংখ্যা

---

## 🔹 Example 5: Add / Subtract Counters

```python
A = Counter('apple')
B = Counter('plea')

# Add
print(A + B)  # Combined counts

# Subtract
print(A - B)  # Only positive results
```

---

## 🔹 Example 6: Convert to dictionary / list

```python
d = dict(Counter("aabc"))
print(d)  # {'a': 2, 'b': 1, 'c': 1}

items = list(Counter("aabc").elements())
print(items)  # ['a', 'a', 'b', 'c']
```

---

## 🔹 Example 7: Update with new items

```python
c = Counter("hello")
c.update("heaven")
print(c)
```

**Output:**

```
Counter({'e': 2, 'h': 2, 'l': 2, 'o': 1, 'a': 1, 'v': 1, 'n': 1})
```

---

## 🔸 Methods Summary

| Method                | Description                                |
| --------------------- | ------------------------------------------ |
| `Counter(iterable)`   | Create counter from iterable               |
| `.most_common([n])`   | Return n most common elements              |
| `.elements()`         | Return iterator of elements (with repeats) |
| `.update(iterable)`   | Add more elements                          |
| `.subtract(iterable)` | Subtract counts                            |
| `.clear()`            | Remove all counts                          |

---

## ✅ Practical Use-Cases

| Use-case                     | Example                                |
| ---------------------------- | -------------------------------------- |
| Word frequency count         | `Counter(text.split())`                |
| Counting characters          | `Counter(string)`                      |
| Frequency of numbers in list | `Counter(list)`                        |
| Inventory tracking           | Adding/removing items by `.update()`   |
| Comparing two data sources   | Using subtraction between two Counters |

---

## 🧠 Bonus: Counter supports set operations

```python
a = Counter('abbbc')
b = Counter('bccd')

print(a & b)  # Intersection: min counts → Counter({'b': 1, 'c': 1})
print(a | b)  # Union: max counts → Counter({'b': 3, 'c': 2, 'a': 1, 'd': 1})
```

---

## 📌 Summary

| Feature             | Description                                     |
| ------------------- | ----------------------------------------------- |
| Import from         | `from collections import Counter`               |
| Works on            | list, string, tuple, dict keys                  |
| Output type         | `Counter`, subclass of `dict`                   |
| Duplicates handled? | ✅ Yes (counts how many times)                   |
| Missing key lookup  | Returns 0 (not error)                           |
| Useful for          | Frequency analysis, word count, inventory, etc. |

---


