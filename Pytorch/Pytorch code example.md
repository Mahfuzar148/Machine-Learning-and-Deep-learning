
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

---

## 🔷 Code Breakdown

```python
x = torch.rand(4, 4)
print(x)
```

* Creates a `4x4` tensor (2D) with random values.
* Shape: `(4, 4)` → Total elements = **16**

---

### 🔹 Reshape with `view(16)`

```python
y = x.view(16)
print(y)
```

✅ **Explanation**:

* `view(16)` reshapes the tensor to a **1D tensor** of 16 elements.
* Shape: `(16,)`
* ✅ No data is lost or added — just rearranged.

---

### 🔹 Reshape with `view(-1, 8)`

```python
y1 = x.view(-1, 8)
print(y1)
```

✅ **Explanation of `-1`**:

* `-1` tells PyTorch:
  **"Automatically calculate this dimension based on the other one."**

* `x` has 16 elements. You’re asking PyTorch to reshape it into shape `(_, 8)`.

* So PyTorch calculates:

  $$
  \frac{16}{8} = 2
  $$

  Thus, the shape becomes `(2, 8)`

✔️ Final shape: `torch.Size([2, 8])`

```python
print(y1.size())  # Outputs: torch.Size([2, 8])
```

---

## 🔶 📌 Summary: What does `-1` do in `view()`?

| Statement          | Resulting Shape | Why                      |
| ------------------ | --------------- | ------------------------ |
| `x.view(16)`       | `[16]`          | Flattens into 1D         |
| `x.view(-1, 8)`    | `[2, 8]`        | PyTorch infers 2         |
| `x.view(4, -1)`    | `[4, 4]`        | PyTorch infers 4 columns |
| `x.view(-1, 2, 2)` | `[4, 2, 2]`     | Nested reshape inferred  |

> ✅ **Rule**: Only one dimension can be `-1` in any `view()` call. Otherwise, PyTorch throws an error.

---

You're exploring **data sharing between PyTorch and NumPy**, which is a very important and subtle topic when working with both libraries. Let's break your example down and provide a detailed tutorial.

---

# 🔍 What This Tutorial Covers

1. **Convert PyTorch Tensor → NumPy Array**
2. **Convert NumPy Array → PyTorch Tensor**
3. **Memory Sharing between NumPy & PyTorch**
4. **When data is copied vs shared**
5. **In-place operations and side effects**

---

## 🧠 Key Concepts

| Conversion Direction | Function                  | Shares Memory? |
| -------------------- | ------------------------- | -------------- |
| PyTorch → NumPy      | `.numpy()`                | ✅ Yes          |
| NumPy → PyTorch      | `torch.from_numpy(array)` | ✅ Yes          |
| PyTorch → NumPy      | `tensor.clone().numpy()`  | ❌ No (copy)    |
| NumPy → PyTorch      | `torch.tensor(array)`     | ❌ No (copy)    |

---

## ✅ Step-by-Step Explanation

### 🔹 1. PyTorch Tensor to NumPy Array using `.numpy()`

```python
import torch

a = torch.ones(5)
b = a.numpy()
print(b)  # [1. 1. 1. 1. 1.]
print(type(b))  # <class 'numpy.ndarray'>
```

* `torch.ones(5)` creates a tensor: `[1, 1, 1, 1, 1]`
* `.numpy()` **returns a NumPy view**, not a copy.
* Both `a` and `b` point to the **same memory**.

### 🧪 Try modifying `a`

```python
a[0] = 2
print(b)  # [2. 1. 1. 1. 1.]
```

* Modifying `a` updates `b` as well. ✅ They share memory.

### 🧪 Try in-place addition on `a`

```python
a.add_(1)  # In-place addition: a = a + 1
print(a)  # [3. 2. 2. 2. 2.]
print(b)  # [3. 2. 2. 2. 2.]
```

* Since `add_()` is in-place, it also updates `b`.

---

### 🔹 2. NumPy Array to PyTorch Tensor using `torch.from_numpy()`

```python
import numpy as np

x = np.ones(5)
y = torch.from_numpy(x)
print(y)  # tensor([1., 1., 1., 1., 1.])
print(type(y))  # <class 'torch.Tensor'>
```

* `torch.from_numpy()` also **shares memory**.

### 🧪 Try modifying `x`

```python
x += 1
print(x)  # [2. 2. 2. 2. 2.]
print(y)  # tensor([2., 2., 2., 2., 2.])
```

* Modifying `x` affects `y` too. ✅ Shared memory.

---

## ❗Important: When They DON’T Share Memory

### 🔸 Using `torch.tensor()` (copies data)

```python
x = np.ones(5)
y = torch.tensor(x)  # makes a copy
x += 1
print(x)  # [2. 2. 2. 2. 2.]
print(y)  # tensor([1., 1., 1., 1., 1.])
```

* `torch.tensor()` copies data: no shared memory

### 🔸 Using `.clone()` on tensor

```python
a = torch.ones(5)
b = a.clone().numpy()
a[0] = 9
print(a)  # [9., 1., 1., 1., 1.]
print(b)  # [1., 1., 1., 1., 1.]
```

* `.clone()` ensures the NumPy array is **independent**.

---

## 📌 When to Use Which?

| Use Case                                       | Recommended Method             |
| ---------------------------------------------- | ------------------------------ |
| Want memory-efficient data sharing             | `.numpy()` or `from_numpy()`   |
| Want independent copy (safe from side effects) | `torch.tensor()` or `.clone()` |

---

## 🧠 In-Place vs Out-of-Place Operations

| Operation   | Description           | Affects Shared Memory |
| ----------- | --------------------- | --------------------- |
| `a.add(1)`  | Returns a new tensor  | ❌ No                  |
| `a.add_(1)` | In-place modification | ✅ Yes                 |

So if you use `add_()`, it affects the shared memory. If you use `add()`, it creates a new object and doesn’t affect the NumPy view.

---

## ✅ Summary Table

| Task                              | Code                              | Shared Memory? |
| --------------------------------- | --------------------------------- | -------------- |
| Tensor → NumPy                    | `tensor.numpy()`                  | ✅ Yes          |
| Tensor → NumPy (safe)             | `tensor.clone().numpy()`          | ❌ No           |
| NumPy → Tensor                    | `torch.from_numpy(array)`         | ✅ Yes          |
| NumPy → Tensor (safe)             | `torch.tensor(array)`             | ❌ No           |
| In-place op (`add_`, `mul_`, etc) | Changes NumPy array or tensor too | ✅ Yes          |
| Out-of-place (`add`, `mul`)       | Creates new object                | ❌ No           |

---

## 💡 Real World Tip

When debugging unexpected tensor or NumPy array changes, always check:

* Are you using `.from_numpy()` or `.numpy()`?
* Are you using in-place operations (`add_`, `mul_`, etc)?
* Do you need memory sharing or a copy?

---

* What `device = torch.device('cuda')` means
* Why `device=device` is written like that
* What `.to(device)` does
* What CUDA does in PyTorch
* CPU vs GPU tensor management

---

## ✅ Full Code with Comments:

```python
if torch.cuda.is_available():
    device = torch.device("cuda")  # ①
    x = torch.ones(5, device=device)  # ②
    y = torch.ones(5)  # ③
    y = y.to(device)   # ④

    z = x + y  # ⑤
    z = z.to("cpu")  # ⑥
    print(z)  # ⑦
```

---

## 🔍 Line-by-Line Explanation

### 🔹 `torch.cuda.is_available()`

* Checks whether a CUDA-compatible **GPU is available** on your machine.
* Returns `True` if PyTorch can use your GPU (e.g., NVIDIA GPU with proper drivers + CUDA installed).

---

### 🔹 `device = torch.device("cuda")`

* This creates a **PyTorch `device` object** representing the GPU.
* `"cuda"` is a keyword for GPU (like `"cpu"` is for processor).
* Now, `device` holds: `device(type='cuda')`

🔁 So when you write `device=device`, you're telling PyTorch **where to create or move the tensor**.

---

### 🔹 `x = torch.ones(5, device=device)`

* Creates a tensor `[1, 1, 1, 1, 1]`
* Puts it **directly on the GPU**
* Equivalent to: `torch.ones(5).to(device)`

---

### 🔹 `y = torch.ones(5)`

* Creates a tensor on **CPU** by default.

---

### 🔹 `y = y.to(device)`

* Moves the tensor `y` from **CPU to GPU**
* Tensors must be on the **same device (both CPU or both GPU)** to do operations like addition.

🔁 That’s why `y.to(device)` is necessary before adding with `x`.

---

### 🔹 `z = x + y`

* Adds two GPU tensors (`x`, `y`)
* Performs the addition **on GPU**

---

### 🔹 `z = z.to("cpu")`

* Moves the result back to **CPU**
* Useful when you want to print it or use it in a NumPy function (which only works on CPU tensors)

---

### 🔹 `print(z)`

* Displays the tensor, which is now back on the CPU.

---

## 🧠 Summary Table

| Concept                     | Meaning                                            |
| --------------------------- | -------------------------------------------------- |
| `torch.device("cuda")`      | Refers to the GPU device                           |
| `device=device`             | Specifies where to place the tensor                |
| `.to(device)`               | Moves tensor to the specified device               |
| `torch.cuda.is_available()` | Checks for GPU availability                        |
| GPU (CUDA)                  | Speeds up deep learning using parallel computation |
| `to("cpu")`                 | Moves tensor back to CPU for printing, NumPy, etc. |

---

## 📌 Output (if CUDA is available)

```python
tensor([2., 2., 2., 2., 2.])
```

Because:

* `x` is `[1, 1, 1, 1, 1]` on GPU
* `y` is also `[1, 1, 1, 1, 1]` moved to GPU
* `z = x + y` → `[2, 2, 2, 2, 2]`
* `z` is moved to CPU and printed

---

## 💡 Tip: Use `.to()` carefully

```python
tensor = tensor.to("cuda")  # GPU
tensor = tensor.to("cpu")   # back to CPU
```

You can also use:

```python
tensor = tensor.cuda()  # Shortcut to move to GPU
tensor = tensor.cpu()   # Shortcut to move to CPU
```

---

---

## ✅ Code

```python
import torch

x = torch.ones(5, requires_grad=True)
print(x)

y = x + 2
print(y)
```

---

## 🔍 Step-by-Step Explanation

### 🔹 `x = torch.ones(5, requires_grad=True)`

* `torch.ones(5)` creates a 1D tensor:

  ```python
  tensor([1., 1., 1., 1., 1.])
  ```

* `requires_grad=True` tells PyTorch to:

  > Track **all operations** on `x` so it can **automatically compute gradients** later via `.backward()`

📌 This is **essential** for training neural networks — it lets you compute **derivatives** of a loss with respect to weights.

### ✅ Output:

```python
tensor([1., 1., 1., 1., 1.], requires_grad=True)
```

This means:

* Values are `[1, 1, 1, 1, 1]`
* PyTorch is tracking the tensor for gradient calculation

---

### 🔹 `y = x + 2`

This adds 2 to every element of `x`.

So:

```python
y = tensor([1., 1., 1., 1., 1.]) + 2
  = tensor([3., 3., 3., 3., 3.])
```

BUT since `x` has `requires_grad=True`, PyTorch also **remembers this operation in a computation graph**.

📌 `y` is now a **computed tensor** that **remembers it was created from `x`**:

```python
y = x + 2
```

That means:

```python
y.requires_grad == True
```

Because it **depends on `x`** (which requires gradients).

✅ Output:

```python
tensor([3., 3., 3., 3., 3.], grad_fn=<AddBackward0>)
```

This shows:

* The values are `[3, 3, 3, 3, 3]`
* `grad_fn=<AddBackward0>` indicates this was **computed** from another tensor and has a **backward function** to compute gradients later.

---

## 🔁 Computation Graph

PyTorch builds a **computation graph** as you perform operations on tensors with `requires_grad=True`.

Here’s the structure in your example:

```
x (leaf node with requires_grad=True)
     |
     +--[ +2 ]--> y (has grad_fn=<AddBackward0>)
```

---

## 🧠 Why This Is Useful

In deep learning:

* You define weights (`x`) and loss function (`y`)
* Then call `y.backward()` to compute the gradient:
  `x.grad` will contain ∂y/∂x

---

## ✅ Next Steps: Try `.backward()`

```python
z = y.sum()  # A scalar
z.backward()  # Computes gradients
print(x.grad)  # Should be tensor([1., 1., 1., 1., 1.])
```

Explanation:

* `z = sum(y)` = sum(x + 2) = sum(x) + 10
* ∂z/∂xᵢ = 1 for all `i`, since z is linear in x

---

## 🔁 Summary Table

| Expression                     | Description                               |
| ------------------------------ | ----------------------------------------- |
| `requires_grad=True`           | Enables gradient tracking                 |
| `y = x + 2`                    | y is tracked and has a `grad_fn`          |
| `y.requires_grad`              | True (because it depends on x)            |
| `y.grad_fn`                    | `<AddBackward0>` (from the `+` operation) |
| `x.grad` (after `.backward()`) | Holds the gradients of x                  |

---



