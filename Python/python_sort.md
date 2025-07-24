

# 🧾 Python Sorting Documentation with Examples

---

## 🔹 1. `sorted()` — Universal Sorting Function ✅

**`sorted()`** Python-এর একটি built-in function যা সব iterable-এর ওপর কাজ করে এবং একটি **new sorted list** রিটার্ন করে।

### 📌 Syntax:

```python
sorted(iterable, key=None, reverse=False)
```

* **iterable**: যেকোনো iterable (list, tuple, set, string, dictionary)
* **key** *(optional)*: একটি function যেটি sorting-এর জন্য value নির্ধারণ করে
* **reverse** *(optional)*: `True` দিলে descending order-এ sort হয়

---

## 🔸 A. List

```python
arr = [4, 1, 3, 2]
sorted_list = sorted(arr)
print(sorted_list)  # [1, 2, 3, 4]
```

### Descending:

```python
print(sorted(arr, reverse=True))  # [4, 3, 2, 1]
```

---

## 🔸 B. Tuple

```python
t = (9, 3, 6, 1)
print(sorted(t))  # [1, 3, 6, 9]
```

➡️ Output always becomes a **list**, even if input is tuple.

---

## 🔸 C. Set

```python
s = {7, 2, 9, 1}
print(sorted(s))  # [1, 2, 7, 9]
```

➡️ Set unordered, তাই sorted করলে order নির্ধারিত হয়।

---

## 🔸 D. String

```python
name = "python"
print(sorted(name))  # ['h', 'n', 'o', 'p', 't', 'y']
```

➡️ String কে character-wise list বানিয়ে sort করে।

---

## 🔸 E. Dictionary

By default, `sorted()` dictionary-এর **keys** sort করে।

```python
d = {'b': 2, 'a': 1, 'c': 3}
print(sorted(d))  # ['a', 'b', 'c']
```

### Keys with values:

```python
sorted_items = sorted(d.items())
print(sorted_items)  # [('a', 1), ('b', 2), ('c', 3)]
```

### Values অনুযায়ী sort:

```python
sorted_by_values = sorted(d.items(), key=lambda item: item[1])
print(sorted_by_values)  # [('a', 1), ('b', 2), ('c', 3)]
```

---

## 🔹 2. `list.sort()` — In-place Sort ✅

**শুধু list-এর জন্য প্রযোজ্য**। এটি মূল list কেই পরিবর্তন করে, কিছু return করে না।

```python
arr = [3, 1, 2]
arr.sort()
print(arr)  # [1, 2, 3]
```

### Descending:

```python
arr.sort(reverse=True)
```

---

## 🔸 Comparing `sorted()` vs `.sort()`

| Feature           | `sorted()`                      | `.sort()`        |
| ----------------- | ------------------------------- | ---------------- |
| Applies to        | All iterables (list, set, etc.) | Only lists       |
| Returns           | New sorted list                 | Nothing (`None`) |
| Original changed? | ❌ না                            | ✅ হ্যাঁ          |

---

## 🔹 3. Custom Sorting using `key` Parameter

### Example: Sort by length of string

```python
words = ['banana', 'fig', 'apple']
print(sorted(words, key=len))  # ['fig', 'apple', 'banana']
```

---

### Example: Sort dictionary by value (descending)

```python
d = {'a': 5, 'b': 2, 'c': 9}
sorted_items = sorted(d.items(), key=lambda item: item[1], reverse=True)
print(sorted_items)  # [('c', 9), ('a', 5), ('b', 2)]
```

---

## 🔹 Summary Table

| Data Type  | Use `sorted()` | Use `.sort()` | Notes                          |
| ---------- | -------------- | ------------- | ------------------------------ |
| List       | ✅              | ✅             | `.sort()` modifies list        |
| Tuple      | ✅              | ❌             | Returns list                   |
| Set        | ✅              | ❌             | Returns list                   |
| String     | ✅              | ❌             | Returns list of characters     |
| Dictionary | ✅              | ❌             | Sorts keys/items, returns list |

---

## 🧠 Bonus Tip: Convert Sorted Output Back

```python
# Tuple sorted back to tuple
t = (3, 1, 2)
sorted_tuple = tuple(sorted(t))
print(sorted_tuple)  # (1, 2, 3)

# String sorted back
s = "data"
sorted_s = ''.join(sorted(s))
print(sorted_s)  # 'aadt'
```

---

## ✅ Conclusion:

🔹 Use `sorted()` when you want a sorted copy of **any iterable**.
🔹 Use `.sort()` only for **in-place sorting of lists**.
🔹 Use `key=` for **custom logic-based sorting**.
🔹 Use `reverse=True` for **descending order**.


