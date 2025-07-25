
---

## 📘 **Space Complexity in Python (and CS generally)**

### 🔹 সংজ্ঞা (Definition):

> **Space Complexity** হল একটি অ্যালগোরিদম **চালাতে কতটুকু অতিরিক্ত মেমোরি বা RAM লাগে**, সেটা পরিমাপ করার একটি পদ্ধতি।

🧠 বুঝে রাখো:

* এটি শুধু input data না — **অতিরিক্ত memory** (extra variables, data structures, call stack ইত্যাদি) কেমন ব্যবহার হচ্ছে সেটাই মূল।
* Time Complexity → সময় নেয় কত
* Space Complexity → মেমোরি খায় কত

---

## 📊 Space Complexity এর Units

যেমন:

* **O(1)** = Constant space → fixed amount of memory লাগে
* **O(n)** = Linear space → input size যত, মেমোরি তত বাড়ে
* **O(n²)** = Memory grows as square of input

---

## ✅ ধরো একটি উদাহরণ:

### 🔸Example 1: Constant Space – O(1)

```python
def sum_array(nums):
    total = 0
    for num in nums:
        total += num
    return total
```

**Explanation**:

* আমরা শুধুমাত্র `total` নামের একটা ভ্যারিয়েবল নিচ্ছি।
* Input array `nums` আগে থেকেই আছে।
* আমরা **extra কিছু তৈরি করছি না**।

✅ **Space Complexity = O(1)**

---

### 🔸Example 2: Linear Space – O(n)

```python
def double_values(nums):
    result = []
    for num in nums:
        result.append(num * 2)
    return result
```

**Explanation**:

* আমরা input array `nums` এর প্রতিটি উপাদানের জন্য নতুন list তৈরি করছি।
* Input size n হলে result list-এর size হবে n।

✅ **Space Complexity = O(n)**

---

### 🔸Example 3: Quadratic Space – O(n²)

```python
def create_matrix(n):
    matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(0)
        matrix.append(row)
    return matrix
```

**Explanation**:

* আমরা `n × n` সাইজের matrix তৈরি করছি।
* তাই মেমোরি usage হবে **n²**।

✅ **Space Complexity = O(n²)**

---

## 🧠 কি কি জিনিস গুনে ধরা হয় Space Complexity-তে?

| ধরা হয়? | জিনিস           | উদাহরণ                              |
| ------- | --------------- | ----------------------------------- |
| ✅       | Extra variables | `total`, `result`, `matrix` ইত্যাদি |
| ✅       | Recursion stack | ফাংশন নিজেকে যতবার কল করে           |
| ❌       | Input size      | ধরেই নেওয়া হয়, input আগে থেকেই আছে  |

---

## 📘 Recursion & Space Complexity

### 🔸Example 4: Recursive Fibonacci

```python
def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)
```

* প্রতিটি ফাংশন কল stack-এ যায়।
* খারাপ case: `O(2^n)` time, **O(n)** space (stack depth)

---

## 🔁 Iterative vs Recursive Space

| Approach  | Example                | Space Complexity   |
| --------- | ---------------------- | ------------------ |
| Iterative | Loop দিয়ে কাজ          | O(1)               |
| Recursive | Function নিজেকে কল করে | O(n) (stack এ জমে) |

---

## ✅ ছোট টিপস

| টিপ                                   | মানে                          |
| ------------------------------------- | ----------------------------- |
| ➕ নতুন লিস্ট, সেট, ডিকশনারি তৈরি করলে | O(n) বা তার বেশি হতে পারে     |
| ➖ শুধু গণনা বা চলক ব্যবহার করলে       | O(1)                          |
| 🌀 রিকার্শন ব্যবহার করলে              | Call Stack এর কথা ভুলে যেও না |

---

## 📌 Interview সময় মনে রাখবে:

| Input Size       | Extra Variables         | Return Structure | Space Complexity |
| ---------------- | ----------------------- | ---------------- | ---------------- |
| 1000 elements    | result list size = 1000 | → O(n)           | O(n)             |
| Just one counter | nothing else            | → O(1)           | O(1)             |

---

## 🔍 Summary Table:

| Code Example Type        | Space Complexity |
| ------------------------ | ---------------- |
| Only counters / sums     | O(1)             |
| New list/array of size n | O(n)             |
| 2D Matrix n×n            | O(n²)            |
| Recursive function depth | O(n)             |

---


