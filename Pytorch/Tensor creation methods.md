
---

## 🧠 What Is a Tensor?

A **tensor** in PyTorch is a multi-dimensional array, similar to NumPy's `ndarray`, but with added support for GPU acceleration and automatic differentiation.

---

## 🔧 PyTorch Tensor Creation Methods

📚 Official Docs: [torch — Creation Ops](https://pytorch.org/docs/stable/torch.html#creation-ops)

---

### 1. **From Existing Data**

| Method           | Description                             | Example                   |
| ---------------- | --------------------------------------- | ------------------------- |
| `torch.tensor()` | Creates tensor from list or NumPy array | `torch.tensor([1, 2, 3])` |

```python
import torch

data = [1, 2, 3]
x = torch.tensor(data)
print(x)  # tensor([1, 2, 3])
```

---

### 2. **Create by Shape with Default Values**

| Method          | Description                     | Example                 |
| --------------- | ------------------------------- | ----------------------- |
| `torch.zeros()` | All elements 0                  | `torch.zeros(2, 3)`     |
| `torch.ones()`  | All elements 1                  | `torch.ones((2, 3))`    |
| `torch.full()`  | All elements set to given value | `torch.full((2, 2), 5)` |

```python
a = torch.zeros(2, 3)
b = torch.ones((2, 3))
c = torch.full((2, 2), 5)
print(a, b, c, sep="\n")
```

---

### 3. **Create with Ranges**

| Method             | Description          | Example                             |
| ------------------ | -------------------- | ----------------------------------- |
| `torch.arange()`   | Sequence of values   | `torch.arange(0, 10, 2)`            |
| `torch.linspace()` | Evenly spaced values | `torch.linspace(0, 1, steps=5)`     |
| `torch.logspace()` | Log-scale values     | `torch.logspace(0.1, 1.0, steps=5)` |

```python
ar = torch.arange(0, 10, 2)
ls = torch.linspace(0, 1, steps=5)
log = torch.logspace(0.1, 1.0, steps=5)
print(ar, ls, log, sep="\n")
```

---

### 4. **Randomized Tensor Creation**

| Method            | Description                   | Example                        |
| ----------------- | ----------------------------- | ------------------------------ |
| `torch.rand()`    | Uniform distribution \[0, 1)  | `torch.rand(2, 2)`             |
| `torch.randn()`   | Standard normal distribution  | `torch.randn(2, 2)`            |
| `torch.randint()` | Discrete uniform \[low, high) | `torch.randint(0, 10, (2, 2))` |

```python
r1 = torch.rand(2, 2)
r2 = torch.randn(2, 2)
r3 = torch.randint(0, 10, (2, 2))
print(r1, r2, r3, sep="\n")
```

---

### 5. **Identity and Diagonal**

| Method         | Description             | Example                             |
| -------------- | ----------------------- | ----------------------------------- |
| `torch.eye()`  | Identity matrix         | `torch.eye(3)`                      |
| `torch.diag()` | Diagonal from 1D tensor | `torch.diag(torch.tensor([1,2,3]))` |

```python
eye = torch.eye(3)
diag = torch.diag(torch.tensor([1, 2, 3]))
print(eye, diag, sep="\n")
```

---

### 6. **Based on Existing Tensor**

| Method               | Description                 | Example               |
| -------------------- | --------------------------- | --------------------- |
| `torch.zeros_like()` | All zeros, shape copied     | `torch.zeros_like(x)` |
| `torch.ones_like()`  | All ones, shape copied      | `torch.ones_like(x)`  |
| `torch.rand_like()`  | Random values, shape copied | `torch.rand_like(x)`  |

```python
x = torch.tensor([[1, 2], [3, 4]])
z_like = torch.zeros_like(x)
r_like = torch.rand_like(x)
print(z_like, r_like, sep="\n")
```

---

---

## 🧪 1. Official PyTorch Tutorial (“Introduction to PyTorch Tensors”)

From the official docs, updated Jan 29, 2025 (verified Nov 5, 2024):

* One of the simplest creation methods:

```python
import torch
x = torch.empty(3, 4)
print(x)  # Uninitialized values, memory allocated only
```

* Common initialization methods:

```python
zeros = torch.zeros(2, 3)
ones = torch.ones(2, 3)
torch.manual_seed(1729)
rand = torch.rand(2, 3)
print(zeros)
print(ones)
print(rand)
```

This shows default dtype (`float32`) and uninitialized memory when using `empty()`([GeeksforGeeks][1], [docs.pytorch.org][2]).

---

## 📰 2. freeCodeCamp Article: “How to Create Tensors in PyTorch”

Provides an overview of ten creation methods:

* Examples include:

```python
torch.eye(n)  # identity matrix, default size n×n
torch.complex(real, imag)  # real and imag must be tensors of same shape
```

Supports range, logspace, random, identity, and complex types([freecodecamp.org][3], [armanasq.github.io][4]).

---

## 📘 3. GeeksforGeeks: “Tensors in PyTorch” Tutorial

Clear usage examples of common creation functions:

```python
# From list
V = torch.tensor([1, 2, 3, 4, 5])

# Random normal
x = torch.randn((3,4,5))

# Zeros/ones/full/arange/linspace
zeros = torch.zeros(3,2, dtype=torch.int32)
ones = torch.ones((4,4,4))
ar = torch.arange(2, 20, 2)
ls = torch.linspace(1, 7.75, 4)
full_t = torch.full((3, 2), 3)
randint_t = torch.randint(10, 100, (2, 2))
eye = torch.eye(4, 4)
c = torch.complex(torch.rand(4, 5), torch.rand(4, 5))
```

This page also illustrates shape parameter rules and dtype flexibility([tutorialexample.com][5], [GeeksforGeeks][6], [GeeksforGeeks][1], [tutorialspoint.com][7]).

---

## 📌 Example Summary from Each Source

| Method            | Official Tutorial                             | freeCodeCamp Example              | GeeksforGeeks Example                       |
| ----------------- | --------------------------------------------- | --------------------------------- | ------------------------------------------- |
| `empty(shape)`    | `torch.empty(3,4)`                            | –                                 | –                                           |
| `zeros/ones`      | `torch.zeros(2,3)`, `torch.ones(2,3)`         | identity via `torch.eye()`        | See zeros/ones above                        |
| `rand/random`     | `torch.rand(2,3)` after `torch.manual_seed()` | uniform, normal, randint examples | `torch.randn((3,4,5))`, `torch.randint()`   |
| `arange/linspace` | –                                             | covered in range and logspace     | `torch.arange(2,20,2)` / `torch.linspace()` |
| `full`            | –                                             | `torch.full((…, value))`          | `torch.full((3,2), 3)`                      |
| `eye/diag`        | –                                             | `torch.eye(n)`                    | `torch.eye(4,4)`                            |
| `complex()`       | –                                             | mentioned among ten ways          | `torch.complex(...)`                        |

---

## ✅ Key Takeaways

1. **Official PyTorch tutorial** emphasizes low-level control (`empty()`), data types, seeding, and default behavior.
2. **freeCodeCamp** illustrates a broader variety of built-in creation functions, including niche ones like `complex()`.
3. **GeeksforGeeks** provides many example snippets for each function, helpful for quick reference.


[1]: https://www.geeksforgeeks.org/python/creating-a-tensor-in-pytorch/?utm_source=chatgpt.com "Creating a Tensor in Pytorch - GeeksforGeeks"
[2]: https://docs.pytorch.org/tutorials/beginner/introyt/tensors_deeper_tutorial.html?utm_source=chatgpt.com "Introduction to PyTorch Tensors"
[3]: https://www.freecodecamp.org/news/pytorch-tensor-methods/?utm_source=chatgpt.com "PyTorch Tensor Methods – How to Create Tensors in Python"
[4]: https://armanasq.github.io/Deep-Learning/PyTorch-Tensors/?utm_source=chatgpt.com "A Profound Journey into PyTorch Tensors: A Comprehensive Tutorial"
[5]: https://www.tutorialexample.com/4-methods-to-create-a-pytorch-tensor-pytorch-tutorial/?utm_source=chatgpt.com "4 Methods to Create a PyTorch Tensor - PyTorch Tutorial - Tutorial Example"
[6]: https://www.geeksforgeeks.org/python/tensors-in-pytorch/?utm_source=chatgpt.com "Tensors in Pytorch - GeeksforGeeks"
[7]: https://www.tutorialspoint.com/creating-a-tensor-in-pytorch?utm_source=chatgpt.com "Creating a Tensor in Pytorch - Online Tutorials Library"



---

## 🧠 1. Official PyTorch Tutorial (Tensors Deeper Tutorial)

**Key operations** include arithmetic with scalars and tensors, exponentiation, and chaining:

```python
import torch

ones = torch.zeros(2, 2) + 1
twos = torch.ones(2, 2) * 2
threes = (torch.ones(2, 2) * 7 - 1) / 2
fours = twos ** 2
sqrt2s = twos ** 0.5

print(ones)
print(twos)
print(threes)
print(fours)
print(sqrt2s)

powers2 = twos ** torch.tensor([[1, 2], [3, 4]])
fives = ones + fours
dozens = threes * fours

print(powers2)
print(fives)
print(dozens)
```

This demonstrates element-wise arithmetic: addition, subtraction, multiplication, division, power, and chaining operations. All tensors above have matching shapes.
([docs.pytorch.org][1])

---

## 2. GeeksforGeeks – Practical Examples for Operations

**Operations covered** include indexing, slicing, reshaping, matrix multiplication:

```python
import torch

tensor = torch.tensor([[1, 2], [3, 4], [5, 6]])
# Indexing
element = tensor[1, 0]
# Slicing
slice2 = tensor[:2, :]
# Reshape using view or reshape
reshaped = tensor.view(2, 3)

print(element)   # 3
print(slice2)    # tensor([[1, 2], [3, 4]])
print(reshaped)  # tensor([[1, 2, 3], [4, 5, 6]])
```

Other examples:

```python
a = torch.tensor([1.0, 2.0])
b = torch.tensor([3.0, 4.0])
print(a + b)                          # Add
print(torch.matmul(a.view(2,1), b.view(1,2)))  # Matrix multiplication

# Reshaping & transposing
t = torch.tensor([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9,10,11,12]])
print(t.reshape(6,2))
print(t.view(2,6))
print(t.transpose(0, 1))
```

([GeeksforGeeks][2], [GeeksforGeeks][3])

---

## 3. CodeSignal Learn – Element-wise, Broadcasting, and Matrix Ops

**Core coverage**: addition, multiplication (element-wise vs. matmul), and broadcasting:

```python
import torch

a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)
# Element-wise
add = a + b
mul = a * b

# Matrix multiplication
mat = torch.matmul(a, b)

# Broadcasting: add a scalar
scalar_add = a + 5

# Broadcasting: align shapes for add/mul
c = torch.tensor([1, 2])
broad_add = a + c        # c broadcasted to match a’s shape
broad_mul = a * c

print(add, mul, mat, scalar_add, broad_add, broad_mul)
```

These examples show how PyTorch handles shape alignment and broadcasting when dimensions differ.
([codesignal.com][4], [github.com][5], [GeeksforGeeks][6], [GeeksforGeeks][7], [GeeksforGeeks][3])

---

## ✅ Summary Table

| Operation                   | Official PyTorch Example                | GeeksforGeeks Example                   | CodeSignal Example                   |
| --------------------------- | --------------------------------------- | --------------------------------------- | ------------------------------------ |
| Arithmetic (add, mul, etc.) | Scalar and tensor arithmetic & chaining | a + b, torch.matmul(...)                | Element-wise & matmul                |
| Indexing & slicing          | —                                       | `tensor[1,0]`, slicing `:`, reshape     | —                                    |
| Reshaping & Transposing     | —                                       | `.view()`, `.reshape()`, `.transpose()` | —                                    |
| Broadcasting                | Implicit in scalar ops                  | —                                       | Scalar + tensor, vector + matrix ops |

---

## 🔧 Combined Example – In One Script

```python
import torch

# Arithmetic & chaining
ones = torch.zeros(2, 2) + 1
threes = (torch.ones(2, 2) * 7 - 1) / 2

# Broadcast
sum_broad = threes + torch.tensor([1, 2])   # shape (2,) broadcast to (2,2)

# Tensor operations
t = torch.tensor([[1, 2], [3, 4], [5, 6]])
elem = t[1, 0]
slice2 = t[:2, :]
resh = t.view(2, 3)
matmul_result = torch.matmul(ones, t[:2, :])  # using matching dims

print("Arithmetic:", ones, threes)
print("Broadcast sum:", sum_broad)
print("Indexed:", elem, "Sliced:", slice2, "Reshaped:", resh)
print("MatMul:", matmul_result)
```

---



[1]: https://docs.pytorch.org/tutorials/beginner/introyt/tensors_deeper_tutorial.html?utm_source=chatgpt.com "Introduction to PyTorch Tensors — PyTorch Tutorials 2.7.0+cu126 ..."
[2]: https://www.geeksforgeeks.org/deep-learning/pytorch-learn-with-examples/?utm_source=chatgpt.com "PyTorch Tutorial - GeeksforGeeks"
[3]: https://www.geeksforgeeks.org/deep-learning/getting-started-with-pytorch/?utm_source=chatgpt.com "What is PyTorch - GeeksforGeeks"
[4]: https://codesignal.com/learn/courses/introduction-to-pytorch-tensors/lessons/fundamental-tensor-operations-in-pytorch?utm_source=chatgpt.com "Fundamental Tensor Operations in PyTorch | CodeSignal Learn"
[5]: https://github.com/sedwna/PyTorch-Tensor-Basics/?utm_source=chatgpt.com "sedwna/PyTorch-Tensor-Basics - GitHub"
[6]: https://www.geeksforgeeks.org/python/how-to-perform-in-place-operations-in-pytorch/?utm_source=chatgpt.com "How to Perform in-place Operations in PyTorch? - GeeksforGeeks"
[7]: https://www.geeksforgeeks.org/python/pytorch-index-based-operation/?utm_source=chatgpt.com "Pytorch - Index-based Operation - GeeksforGeeks"
