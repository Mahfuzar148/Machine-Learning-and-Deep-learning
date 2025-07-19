Here's a comprehensive documentation of Python's **`list`** data structure, focusing first on listing all built-in methods (functions), and then explaining each in detail with syntax, usage, and examples.

---

## 🔹 Python `list`: Overview

A `list` in Python is an **ordered**, **mutable**, and **heterogeneous** collection of items. It allows **duplicates** and supports a wide range of **built-in methods**.

### Syntax

```python
my_list = [1, 2, 3, 'a', 'b']
```

---

## 🔹 Built-in List Methods

Here’s a complete list of built-in methods for Python lists:

| Method      | Description                             |
| ----------- | --------------------------------------- |
| `append()`  | Adds a single item to the end           |
| `extend()`  | Adds all elements from another iterable |
| `insert()`  | Inserts an item at a given index        |
| `remove()`  | Removes the first occurrence of a value |
| `pop()`     | Removes and returns an item by index    |
| `clear()`   | Removes all elements                    |
| `index()`   | Returns index of first occurrence       |
| `count()`   | Returns the number of occurrences       |
| `sort()`    | Sorts the list in place                 |
| `reverse()` | Reverses the list in place              |
| `copy()`    | Returns a shallow copy                  |

---

## 🔹 Method-by-Method Explanation

### 1. `append(x)`

**Adds** a single element `x` to the **end** of the list.

```python
lst = [1, 2]
lst.append(3)
print(lst)  # [1, 2, 3]
```

---

### 2. `extend(iterable)`

Appends **each element** from `iterable` (like list, tuple) to the list.

```python
lst = [1, 2]
lst.extend([3, 4])
print(lst)  # [1, 2, 3, 4]
```

---

### 3. `insert(i, x)`

Inserts `x` at position `i`. Elements after `i` are shifted.

```python
lst = [1, 3]
lst.insert(1, 2)
print(lst)  # [1, 2, 3]
```

---

### 4. `remove(x)`

Removes the **first occurrence** of `x`.

```python
lst = [1, 2, 3, 2]
lst.remove(2)
print(lst)  # [1, 3, 2]
```

---

### 5. `pop([i])`

Removes and returns the element at index `i`. Default is last element.

```python
lst = [1, 2, 3]
lst.pop()     # 3
lst.pop(0)    # 1
```

---

### 6. `clear()`

Removes **all items** from the list.

```python
lst = [1, 2, 3]
lst.clear()
print(lst)  # []
```

---

### 7. `index(x, [start], [end])`

Returns **index** of first occurrence of `x`. Optional start/end for search range.

```python
lst = ['a', 'b', 'c', 'a']
lst.index('a')        # 0
lst.index('a', 1)     # 3
```

---

### 8. `count(x)`

Returns the **number of times** `x` appears.

```python
lst = [1, 2, 2, 3]
lst.count(2)  # 2
```

---

### 9. `sort(key=None, reverse=False)`

Sorts list in-place. Can take a key function and reverse flag.

```python
lst = [3, 1, 2]
lst.sort()
print(lst)  # [1, 2, 3]

lst.sort(reverse=True)  # [3, 2, 1]
```

Custom sort:

```python
lst = ['ccc', 'bb', 'a']
lst.sort(key=len)  # ['a', 'bb', 'ccc']
```

---

### 10. `reverse()`

Reverses the list **in place**.

```python
lst = [1, 2, 3]
lst.reverse()
print(lst)  # [3, 2, 1]
```

---

### 11. `copy()`

Returns a **shallow copy** of the list.

```python
lst = [1, 2, 3]
new_lst = lst.copy()
print(new_lst)  # [1, 2, 3]
```

> For deep copy (when list has nested structures), use `copy` module:

```python
import copy
deep_lst = copy.deepcopy(lst)
```

---

## 🔹 Bonus: Other Useful Concepts

### List Comprehension

A concise way to create lists:

```python
squares = [x**2 for x in range(5)]
```

### Membership Test

```python
3 in [1, 2, 3]       # True
4 not in [1, 2, 3]   # True
```

### Iteration

```python
for item in ['a', 'b', 'c']:
    print(item)
```

---

## ✅ Summary

Python lists are incredibly versatile and equipped with many built-in methods for efficient data manipulation. Mastering these will enhance your programming fluency, especially when dealing with collections of items.


---

## 🔹 `append()` – Add an item to the end

```python
my_list = [1, 2, 3]
print("Before append:", my_list)

my_list.append(4)
print("After append:", my_list)
```

---

## 🔹 `extend()` – Add multiple items

```python
my_list = [1, 2]
print("Before extend:", my_list)

my_list.extend([3, 4])
print("After extend:", my_list)
```

---

## 🔹 `insert()` – Insert at a specific index

```python
my_list = ['a', 'b', 'd']
print("Before insert:", my_list)

my_list.insert(2, 'c')
print("After insert:", my_list)
```

---

## 🔹 `remove()` – Remove the first matching item

```python
my_list = [10, 20, 30, 20]
print("Before remove:", my_list)

my_list.remove(20)
print("After remove (removes first 20):", my_list)
```

---

## 🔹 `pop()` – Remove and return an item

```python
my_list = ['x', 'y', 'z']
print("Before pop:", my_list)

last_item = my_list.pop()
print("Popped item:", last_item)
print("After pop:", my_list)

first_item = my_list.pop(0)
print("Popped first item:", first_item)
print("Final list:", my_list)
```

---

## 🔹 `clear()` – Remove all items

```python
my_list = [1, 2, 3]
print("Before clear:", my_list)

my_list.clear()
print("After clear:", my_list)
```

---

## 🔹 `index()` – Get index of an item

```python
my_list = ['apple', 'banana', 'cherry', 'apple']
print("Index of 'apple':", my_list.index('apple'))
print("Index of 'apple' after index 1:", my_list.index('apple', 1))
```

---

## 🔹 `count()` – Count occurrences

```python
my_list = [1, 2, 2, 3, 2, 4]
print("Count of 2:", my_list.count(2))
```

---

## 🔹 `sort()` – Sort the list

```python
nums = [3, 1, 4, 2]
print("Before sort:", nums)

nums.sort()
print("After sort (ascending):", nums)

nums.sort(reverse=True)
print("After sort (descending):", nums)
```

Custom key:

```python
words = ['aaa', 'bb', 'c']
words.sort(key=len)
print("Sorted by length:", words)
```

---

## 🔹 `reverse()` – Reverse the list

```python
letters = ['a', 'b', 'c']
print("Before reverse:", letters)

letters.reverse()
print("After reverse:", letters)
```

---

## 🔹 `copy()` – Make a shallow copy

```python
original = [1, 2, 3]
copy_list = original.copy()

print("Original:", original)
print("Copy:", copy_list)

copy_list.append(4)
print("Modified Copy:", copy_list)
print("Original remains unchanged:", original)
```

---

## 🔹 Extra: List Comprehension Example

```python
squares = [x**2 for x in range(6)]
print("Squares:", squares)
```



---

## ✅ 25 Python List Practice Problems – With Explanations & Solutions

---

### 1. **Sum all elements in a list**

**Explanation**: Use the built-in `sum()` function.

```python
my_list = [1, 2, 3, 4]
total = sum(my_list)  # Output: 10
```

---

### 2. **Find the maximum element in a list**

**Explanation**: Use the built-in `max()` function.

```python
my_list = [5, 8, 1]
maximum = max(my_list)  # Output: 8
```

---

### 3. **Remove all duplicates from a list**

**Explanation**: Convert the list to a set and back to list.

```python
my_list = [1, 2, 2, 3]
unique = list(set(my_list))  # Output: [1, 2, 3] (order not preserved)
```

---

### 4. **Count the frequency of a specific element**

**Explanation**: Use the `count()` method.

```python
my_list = [1, 2, 2, 3]
count_2 = my_list.count(2)  # Output: 2
```

---

### 5. **Reverse a list**

**Explanation**: Use slicing with a negative step.

```python
my_list = [1, 2, 3]
reversed_list = my_list[::-1]  # Output: [3, 2, 1]
```

---

### 6. **Check if a list is empty**

**Explanation**: Use `not` to check if list is empty.

```python
my_list = []
is_empty = not my_list  # Output: True
```

---

### 7. **Find the index of the first occurrence**

**Explanation**: Use `index()` method.

```python
my_list = ['a', 'b', 'a']
index = my_list.index('a')  # Output: 0
```

---

### 8. **Extract even numbers from a list**

**Explanation**: Use list comprehension.

```python
my_list = [1, 2, 3, 4]
evens = [x for x in my_list if x % 2 == 0]  # Output: [2, 4]
```

---

### 9. **Square each number in a list**

**Explanation**: Use list comprehension.

```python
my_list = [1, 2, 3]
squares = [x**2 for x in my_list]  # Output: [1, 4, 9]
```

---

### 10. **Sort a list in ascending order**

**Explanation**: Use `sorted()` or `sort()`.

```python
my_list = [3, 1, 2]
sorted_list = sorted(my_list)  # Output: [1, 2, 3]
```

---

### 11. **Check if all elements are positive**

**Explanation**: Use `all()` with a condition.

```python
my_list = [1, 2, 3]
is_all_positive = all(x > 0 for x in my_list)  # Output: True
```

---

### 12. **Find the second largest number**

**Explanation**: Convert to a set and sort.

```python
my_list = [3, 1, 4, 4, 2]
second_largest = sorted(set(my_list))[-2]  # Output: 3
```

---

### 13. **Remove all instances of an item**

**Explanation**: Use list comprehension.

```python
my_list = [1, 2, 3, 2]
filtered = [x for x in my_list if x != 2]  # Output: [1, 3]
```

---

### 14. **Find common elements between two lists**

**Explanation**: Use set intersection.

```python
a = [1, 2, 3]
b = [2, 3, 4]
common = list(set(a) & set(b))  # Output: [2, 3]
```

---

### 15. **Check if a list is a palindrome**

**Explanation**: Compare list with its reverse.

```python
my_list = [1, 2, 1]
is_palindrome = my_list == my_list[::-1]  # Output: True
```

---

### 16. **Flatten a list of lists**

**Explanation**: Use nested list comprehension.

```python
nested = [[1, 2], [3, 4]]
flat = [item for sublist in nested for item in sublist]  # Output: [1, 2, 3, 4]
```

---

### 17. **Calculate the product of list elements**

**Explanation**: Use `reduce()` and `operator.mul`.

```python
from functools import reduce
from operator import mul

my_list = [2, 3, 4]
product = reduce(mul, my_list)  # Output: 24
```

---

### 18. **Remove first and last elements**

**Explanation**: Use slicing.

```python
my_list = [1, 2, 3, 4]
trimmed = my_list[1:-1]  # Output: [2, 3]
```

---

### 19. **Get the difference between two lists**

**Explanation**: Use list comprehension.

```python
a = [1, 2, 3]
b = [2, 3]
diff = [x for x in a if x not in b]  # Output: [1]
```

---

### 20. **Merge and sort two lists**

**Explanation**: Concatenate and sort.

```python
a = [3, 1]
b = [4, 2]
merged = sorted(a + b)  # Output: [1, 2, 3, 4]
```

---

### 21. **Find the longest sublist**

**Explanation**: Use `max()` with `key=len`.

```python
lists = [[1], [1, 2], [1, 2, 3]]
longest = max(lists, key=len)  # Output: [1, 2, 3]
```

---

### 22. **Split a list into chunks**

**Explanation**: Use slicing in a loop.

```python
my_list = [1, 2, 3, 4, 5, 6]
n = 2
chunks = [my_list[i:i+n] for i in range(0, len(my_list), n)]
# Output: [[1, 2], [3, 4], [5, 6]]
```

---

### 23. **Count even and odd numbers**

**Explanation**: Use generator expressions.

```python
my_list = [1, 2, 3, 4]
evens = sum(x % 2 == 0 for x in my_list)  # Output: 2
odds = sum(x % 2 != 0 for x in my_list)   # Output: 2
```

---

### 24. **Find last occurrence of an element**

**Explanation**: Reverse the list and use `index()`.

```python
my_list = [1, 2, 3, 2, 4]
target = 2
last_index = len(my_list) - 1 - my_list[::-1].index(target)  # Output: 3
```

---

### 25. **Rotate list to the left by `n` positions**

**Explanation**: Use slicing.

```python
my_list = [1, 2, 3, 4, 5]
n = 2
rotated = my_list[n:] + my_list[:n]  # Output: [3, 4, 5, 1, 2]
```

---

## 🔷 LeetCode – Python Array/List Practice

Known problem lists from LeetCode:

* **LeetCode: Array Problem List** 🔗 \[Problem List – Arrays] ([LeetCode][1], [LeetCode][2])
* **LeetCode: Fun with Arrays Tutorial + Problems** 🔗 \[Explore – Fun with Arrays] ([LeetCode][1])

From these, here are 20 specific problems with direct links:

1. **Two Sum** – [https://leetcode.com/problems/two-sum/](https://leetcode.com/problems/two-sum/)
2. **Remove Element** – [https://leetcode.com/problems/remove-element/](https://leetcode.com/problems/remove-element/) ([LeetCode][3], [LeetCode][4])
3. **Merge Sorted Array** – [https://leetcode.com/problems/merge-sorted-array/](https://leetcode.com/problems/merge-sorted-array/) ([LeetCode][5])
4. **Best Time to Buy and Sell Stock** – [https://leetcode.com/problems/best-time-to-buy-and-sell-stock/](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/)
5. **Contains Duplicate** – [https://leetcode.com/problems/contains-duplicate/](https://leetcode.com/problems/contains-duplicate/)
6. **Product of Array Except Self** – [https://leetcode.com/problems/product-of-array-except-self/](https://leetcode.com/problems/product-of-array-except-self/)
7. **Maximum Subarray** – [https://leetcode.com/problems/maximum-subarray/](https://leetcode.com/problems/maximum-subarray/)
8. **Rotate Array** – [https://leetcode.com/problems/rotate-array/](https://leetcode.com/problems/rotate-array/)
9. **Find Pivot Index** – [https://leetcode.com/problems/find-pivot-index/](https://leetcode.com/problems/find-pivot-index/)
10. **Valid Mountain Array** – [https://leetcode.com/problems/valid-mountain-array/](https://leetcode.com/problems/valid-mountain-array/)
11. **Monotonic Array** – [https://leetcode.com/problems/monotonic-array/](https://leetcode.com/problems/monotonic-array/) ([LeetCode][4], [LeetCode][6])
12. **Move Zeroes** – [https://leetcode.com/problems/move-zeroes/](https://leetcode.com/problems/move-zeroes/)
13. **Intersection of Two Arrays II** – [https://leetcode.com/problems/intersection-of-two-arrays-ii/](https://leetcode.com/problems/intersection-of-two-arrays-ii/)
14. **Plus One** – [https://leetcode.com/problems/plus-one/](https://leetcode.com/problems/plus-one/)
15. **Plus One Linked List** (array variation) – [https://leetcode.com/problems/plus-one/](https://leetcode.com/problems/plus-one/)
16. **Find All Numbers Disappeared in an Array** – [https://leetcode.com/problems/find-all-numbers-disappeared-in-an-array/](https://leetcode.com/problems/find-all-numbers-disappeared-in-an-array/)
17. **Set Matrix Zeroes** – [https://leetcode.com/problems/set-matrix-zeroes/](https://leetcode.com/problems/set-matrix-zeroes/)
18. **Sort Colors** – [https://leetcode.com/problems/sort-colors/](https://leetcode.com/problems/sort-colors/)
19. **Missing Number** – [https://leetcode.com/problems/missing-number/](https://leetcode.com/problems/missing-number/)
20. **Beautiful Array** – [https://leetcode.com/problems/beautiful-array/](https://leetcode.com/problems/beautiful-array/) ([LeetCode][7])

*(These are commonly included in the Array problem list)*

---

## 🔶 Codeforces – Array-Centric Problems

Here are **5 classic Codeforces problems** centered around array/list operations:

21. **300A – Array**: Divide an array into three sets (negative, positive, zero)
    🔗 [https://codeforces.com/problemset/problem/300/A](https://codeforces.com/problemset/problem/300/A) ([Codeforces][8])

22. **Divisible Subarrays**: CF Round – count subarrays with sum divisible by *m*
    🔗 (Example) [https://codeforces.com/problemset/problem/2111/G](https://codeforces.com/problemset/problem/2111/G) ([Codeforces][9])

23. **Shrinking Array** – CF 2112B: perform greedy removals from array ends
    🔗 [https://codeforces.com/problemset/problem/2112/B](https://codeforces.com/problemset/problem/2112/B) ([Codeforces][9])

24. **Watermelon** – simplest parity-based split problem
    🔗 [https://codeforces.com/problemset/problem/4/A](https://codeforces.com/problemset/problem/4/A)

25. **Beautiful Matrix** – CF 263A: transform a matrix to make the center element 1
    🔗 [https://codeforces.com/problemset/problem/263/A](https://codeforces.com/problemset/problem/263/A)

---

### ✅ Next Steps

* On **LeetCode**, start by exploring the **Array** list and solving the 20 above.
* On **Codeforces**, check out the 5 listed problems and search for tags like `array` or `implementation` to find more.

Would you like me to generate:

* A **Jupyter notebook** with starter code for all 25?
* A more detailed **difficulty + approach summary** for each?
* Additional **Codeforces array problems** with filtering by Python-specific difficulty?

Let me know!

[1]: https://leetcode.com/explore/learn/card/fun-with-arrays/?utm_source=chatgpt.com "Explore - LeetCode"
[2]: https://leetcode.com/problem-list/arrays/?utm_source=chatgpt.com "Problem List - LeetCode"
[3]: https://leetcode.com/problems/remove-element/solutions/12695/accepted-java-solution-using-arraylist/?utm_source=chatgpt.com "Remove Element - LeetCode"
[4]: https://leetcode.com/problems/monotonic-array/?utm_source=chatgpt.com "Monotonic Array - LeetCode"
[5]: https://leetcode.com/problems/merge-sorted-array/?utm_source=chatgpt.com "Merge Sorted Array - LeetCode"
[6]: https://leetcode.com/problems/find-in-mountain-array/?utm_source=chatgpt.com "Find in Mountain Array - LeetCode"
[7]: https://leetcode.com/problems/beautiful-array/?utm_source=chatgpt.com "Beautiful Array - LeetCode"
[8]: https://codeforces.com/problemset/problem/300/a?utm_source=chatgpt.com "Problem - 300a - Codeforces"
[9]: https://codeforces.com/problemset?utm_source=chatgpt.com "Problemset - Codeforces"


