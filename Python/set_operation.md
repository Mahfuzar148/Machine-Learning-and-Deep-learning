
---

# 🧾 Python Set Operations (Mathematics Style)

Python's `set` data type directly supports many **mathematical set operations**, such as union, intersection, difference, and more.

---

## 🔹 1. Set Creation

```python
A = {1, 2, 3}
B = {3, 4, 5}
```

Or using constructor:

```python
A = set([1, 2, 3])
B = set([3, 4, 5])
```

---

## 🔹 2. Set Union (`A ∪ B`)

Combines elements from both sets (no duplicates).

```python
A | B
# or
A.union(B)
```

**Output:**

```python
{1, 2, 3, 4, 5}
```

---

## 🔹 3. Set Intersection (`A ∩ B`)

Common elements in both sets.

```python
A & B
# or
A.intersection(B)
```

**Output:**

```python
{3}
```

---

## 🔹 4. Set Difference (`A - B`)

Elements in `A` but not in `B`.

```python
A - B
# or
A.difference(B)
```

**Output:**

```python
{1, 2}
```

---

## 🔹 5. Symmetric Difference (`A Δ B`)

Elements in either `A` or `B` but not both.

```python
A ^ B
# or
A.symmetric_difference(B)
```

**Output:**

```python
{1, 2, 4, 5}
```

---

## 🔹 6. Subset & Superset

### A is subset of B (`A ⊆ B`)

```python
A.issubset(B)
```

### A is superset of B (`A ⊇ B`)

```python
A.issuperset(B)
```

### A equals B (`A == B`)

```python
A == B
```

---

## 🔹 7. Disjoint Sets

No elements in common.

```python
A.isdisjoint(B)
```

**Returns** `True` if intersection is empty.

---

## 🔹 8. Modifying Sets (in-place)

| Operation            | Method                             |
| -------------------- | ---------------------------------- |
| Union                | `A.update(B)`                      |
| Intersection         | `A.intersection_update(B)`         |
| Difference           | `A.difference_update(B)`           |
| Symmetric Difference | `A.symmetric_difference_update(B)` |

> These **modify** `A` directly.

---

## 🔹 9. Set Functions

| Function | Description         |
| -------- | ------------------- |
| `len(S)` | Number of elements  |
| `min(S)` | Minimum element     |
| `max(S)` | Maximum element     |
| `sum(S)` | Sum of all elements |

---

## 🔹 10. Set Conversion Examples

```python
list_to_set = set([1, 2, 2, 3])  # {1, 2, 3}
str_to_set = set("banana")      # {'a', 'b', 'n'}
```

---

## 🔹 11. Removing & Adding Elements

```python
s.add(x)       # Add element x
s.remove(x)    # Remove x, error if not found
s.discard(x)   # Remove x if present, no error
s.clear()      # Remove all elements
```

---

## 🔹 Example Summary Table

| Operation            | Symbol | Method                                 | Example Result    |                   |
| -------------------- | ------ | -------------------------------------- | ----------------- | ----------------- |
| Union                | ∪      | \`A                                    | B`or`A.union(B)\` | `{1, 2, 3, 4, 5}` |
| Intersection         | ∩      | `A & B` or `A.intersection(B)`         | `{3}`             |                   |
| Difference           | −      | `A - B` or `A.difference(B)`           | `{1, 2}`          |                   |
| Symmetric Difference | Δ      | `A ^ B` or `A.symmetric_difference(B)` | `{1, 2, 4, 5}`    |                   |
| Subset               | ⊆      | `A.issubset(B)`                        | `False` or `True` |                   |
| Superset             | ⊇      | `A.issuperset(B)`                      | `False` or `True` |                   |
| Disjoint             |        | `A.isdisjoint(B)`                      | `False` or `True` |                   |

---

## ✅ Final Notes

* Sets are **unordered** and contain **no duplicates**.
* Elements must be **immutable** (e.g., numbers, strings, tuples).
* Use `frozenset` if you need an **immutable set** (e.g., as dictionary keys).

---

