
---

## 🔶 1. **Importing PyTorch**

```python
import torch
```

* This imports the PyTorch library so you can use its functions.

---

## 🔶 2. **Creating Tensors**

### 🔸 Random tensor

```python
x = torch.rand(3)
print(x)
```

* Creates a 1D tensor of 3 elements with **random values between 0 and 1**.

### 🔸 Empty tensor

```python
x1 = torch.empty(3)
```

* Creates an uninitialized 1D tensor of size 3. Contents are **random garbage values** (whatever is in memory).

```python
x2 = torch.empty(2, 3)
x3 = torch.empty(2, 3, 4)
```

* Same idea but with different shapes:

  * `x2`: 2 rows, 3 columns
  * `x3`: 2 blocks, 3 rows, 4 columns

---

## 🔶 3. **Other Tensor Initializations**

```python
rand = torch.rand(2, 3, 4)
```

* 3D tensor with random values between 0 and 1

```python
zeros = torch.zeros(2, 3, 4)
```

* 3D tensor filled with **zeros**

```python
ones = torch.ones(2, 3, 4, dtype=torch.int)
```

* 3D tensor filled with **ones**, with **integer data type**

```python
print(ones.dtype)
```

* Prints data type: `torch.int32` or `torch.int64`, depending on system

### 🔸 Changing data types

```python
ones = torch.ones(2, 3, 4, dtype=torch.double)
print(ones.dtype)  # torch.float64
```

```python
ones = torch.ones(2, 3, 4, dtype=torch.float32)
print(ones.dtype)  # torch.float32
```

```python
print(ones.size())
```

* Prints the shape: `torch.Size([2, 3, 4])`

---

## 🔶 4. **Basic Tensor Math**

### 🔸 Element-wise addition

```python
x = torch.rand(2, 2)
y = torch.rand(2, 2)
print(x + y)
print(torch.add(x, y))
```

* `x + y` and `torch.add(x, y)` do the same thing: **element-wise addition**

### 🔸 In-place addition

```python
y.add_(x)
print(y)
```

* Adds `x` to `y` **in-place**, modifying `y` directly (notice the `_`)

### 🔸 Subtraction

```python
print(x - y)
print(torch.sub(x, y))
```

### 🔸 Multiplication

```python
print(x * y)
print(torch.mul(x, y))
```

### 🔸 Division

```python
print(x / y)
print(torch.div(x, y))
```

---

## 🔶 5. **In-Place Arithmetic**

```python
a = torch.rand(2, 3)
a1 = a  # `a1` and `a` point to the same memory
b = torch.rand(2, 3)
```

### 🔸 In-place subtraction

```python
a.sub_(b)
```

* `a` is updated: `a = a - b`

### 🔸 Reset `a` to original (still same as `a1`)

```python
a = a1
```

### 🔸 In-place multiplication

```python
a.mul_(b)
```

### 🔸 In-place division

```python
a.div_(b)
```

### 🔸 Safe division (returns new tensor)

```python
print(torch.div(a, b))
```

---

## 🧠 Summary of Important Concepts:

| Operation | Description                 | In-place Version |
| --------- | --------------------------- | ---------------- |
| `x + y`   | Element-wise addition       | `y.add_(x)`      |
| `x - y`   | Element-wise subtraction    | `x.sub_(y)`      |
| `x * y`   | Element-wise multiplication | `x.mul_(y)`      |
| `x / y`   | Element-wise division       | `x.div_(y)`      |


---

## 🔶 Code Overview: Tensor Indexing & Conversion

```python
import torch
x = torch.rand(3, 4)
print(x)
```

### ✅ Explanation:

* Creates a **2D tensor** with 3 rows and 4 columns filled with random values.
* Shape: `(3, 4)`

---

## 🔶 1. **Accessing All Rows and Columns**

```python
print(x[:, :])
```

### ✅ Explanation:

* This is **full slicing**: `:` means **"take everything"**.
* `x[:, :]` = All rows, all columns (same as just `x`).

> 🧠 Similar to Python NumPy slicing: `matrix[row_slice, column_slice]`

---

## 🔶 2. **Accessing a Specific Row**

```python
print(x[0, :])
```

### ✅ Explanation:

* First row (index 0), all columns.
* Shape: `(4,)` → a 1D tensor with 4 values.

```python
print(x[1, :])
```

* Second row (index 1), all columns.

---

## 🔶 3. **Accessing Specific Elements**

```python
print(x[0, 0])  # First row, first column
print(x[0, 1])  # First row, second column
print(x[1, 2])  # Second row, third column
```

### ✅ Explanation:

* Direct indexing for single scalar values inside the 2D tensor.
* Each returns a **0-dimensional tensor (scalar)**, not a number yet.

---

## 🔶 4. **Convert Scalar Tensor to Python Number**

```python
print(x[1, 2].item())
```

### ✅ Explanation:

* `.item()` converts a 0D tensor to a Python number (e.g., `float`).
* ✅ Use only for **scalar tensors** (i.e., single values), otherwise it will raise an error.

---

## 🔶 5. **Convert Tensor to NumPy Array**

```python
print(x.numpy())
```

### ✅ Explanation:

* Converts the entire tensor `x` to a NumPy array.
* Very useful when you need to use NumPy operations on PyTorch tensors.

> ⚠️ Works **only if the tensor is on CPU**, not on GPU.
> If `x` is on GPU, use `x.cpu().numpy()` instead.

---

## 🔶 6. **Get Tensor Shape**

```python
print(x.shape)
```

### ✅ Explanation:

* Returns the **dimensions** of the tensor.
* In this case: `torch.Size([3, 4])`
  Which means: 3 rows, 4 columns

> ✅ You can also use `x.size()` — same as `x.shape`.

---

## 🧠 Summary Table

| Expression              | Meaning                                | Output Type         |
| ----------------------- | -------------------------------------- | ------------------- |
| `x[:, :]`               | All rows, all columns                  | 2D tensor (3x4)     |
| `x[0, :]`               | First row                              | 1D tensor (4,)      |
| `x[1, 2]`               | Element at row 1, column 2             | Scalar tensor (0D)  |
| `x[1, 2].item()`        | Convert scalar tensor to Python number | float               |
| `x.numpy()`             | Convert tensor to NumPy array          | NumPy ndarray       |
| `x.shape` or `x.size()` | Shape of tensor                        | `torch.Size([3,4])` |

---


