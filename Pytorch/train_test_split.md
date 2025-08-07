
---

## 📘 `train_test_split` – Documentation with Examples

### 🔹 Function:

```python
sklearn.model_selection.train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None)
```

### 🔹 Purpose:

Splits arrays or matrices into **random train and test subsets**.
Used to evaluate models effectively by separating data into **training** and **testing** sets.

---

## 🔹 Parameters:

| Parameter      | Type                  | Description                                                                               |
| -------------- | --------------------- | ----------------------------------------------------------------------------------------- |
| `*arrays`      | array-like            | Input data (features `X`, labels `y`, etc.) to be split.                                  |
| `test_size`    | float or int          | Proportion (e.g., `0.2`) or number (e.g., `200`) of samples to include in the test split. |
| `train_size`   | float or int          | Proportion or number of training samples. Optional if `test_size` is set.                 |
| `random_state` | int or None           | Controls shuffling for reproducibility. Same number = same split every time.              |
| `shuffle`      | bool (default `True`) | Whether or not to shuffle the data before splitting.                                      |
| `stratify`     | array-like or None    | Ensures the class distribution is preserved in train/test sets. Typically passed `y`.     |

---

## 🔹 Returns:

* Split arrays in the **same order** as input.
  Example:

```python
X_train, X_test, y_train, y_test
```

---

## 🔹 Example 1: Simple split

```python
from sklearn.model_selection import train_test_split

X = [[1], [2], [3], [4], [5], [6], [7], [8]]
y = [0, 1, 0, 1, 0, 1, 0, 1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print("X_train:", X_train)
print("X_test:", X_test)
```

**Output will always be same** every run due to `random_state=42`.

---

## 🔹 Example 2: Stratified split (maintaining class balance)

```python
X = [[1], [2], [3], [4], [5], [6]]
y = [0, 0, 0, 1, 1, 1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=0, stratify=y
)
```

Here, `stratify=y` ensures class balance is preserved in both sets.

---

## ✅ Notes:

* If you **don't set `random_state`**, each run will produce a **different split**.
* `train_size + test_size` must not exceed 1 (if using float).
* `stratify` is very useful for **classification problems**.

---

## 🛠 Use Case in ML Workflow:

```python
# Step 1: Load your data
X, y = load_data()

# Step 2: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train your model
model.fit(X_train, y_train)

# Step 4: Evaluate
model.evaluate(X_test, y_test)
```

---

