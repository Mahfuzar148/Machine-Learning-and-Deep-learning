
---

# 📘 TensorFlow Documentation: **Tensor Basics**

---

## ✅ **1. What is a Tensor?**

A **Tensor** is a multi-dimensional array of numbers. It is the fundamental data structure in TensorFlow used for all computations.

---

## ✅ **2. Tensor Structure (Rank, Shape, Dtype)**

### **Rank**:

* Number of tensor dimensions.
* Examples:

  * Scalar (rank-0): `42`
  * Vector (rank-1): `[1, 2, 3]`
  * Matrix (rank-2): `[[1,2], [3,4]]`
  * Higher-dimensional tensors (3 or more dimensions).

### **Shape**:

* The length of each tensor dimension.
* Example:

  * Shape `[2, 3]` means a matrix with 2 rows and 3 columns.

### **Dtype**:

* The datatype of tensor elements (e.g., `tf.float32`, `tf.int32`, `tf.string`).

**Example:**

```python
import tensorflow as tf

tensor = tf.constant([[1, 2, 3], [4, 5, 6]])

print("Rank:", tensor.ndim)
print("Shape:", tensor.shape)
print("Dtype:", tensor.dtype)
```

**Output:**

```
Rank: 2
Shape: (2, 3)
Dtype: <dtype: 'int32'>
```

---

## ✅ **3. Creating Tensors**

### ✔️ **`tf.constant`** (immutable tensors with fixed values)

```python
tensor = tf.constant([1, 2, 3])
```

### ✔️ **`tf.zeros`** (creates tensor filled with zeros)

```python
tensor_zeros = tf.zeros([2, 3])
```

### ✔️ **`tf.ones`** (creates tensor filled with ones)

```python
tensor_ones = tf.ones([3, 2])
```

### ✔️ **`tf.fill`** (creates tensor filled with a specific value)

```python
tensor_fill = tf.fill([2, 2], 9)
```

### ✔️ **`tf.range`** (creates a sequence of numbers)

```python
tensor_range = tf.range(start=0, limit=10, delta=2)
```

### ✔️ **`tf.random`** (creates tensor with random values)

```python
tensor_random = tf.random.uniform([2,2], minval=0, maxval=5)
```

---

## ✅ **4. Types of Tensors**

### ✔️ **Dense Tensors**

* Regular tensors with values for every dimension.

```python
dense = tf.constant([[1, 2], [3, 4]])
```

### ✔️ **Sparse Tensors**

* Efficient storage for tensors with many zeros.

```python
sparse = tf.sparse.SparseTensor(indices=[[0, 1], [1, 0]],
                                values=[3, 4],
                                dense_shape=[2, 2])
dense_version = tf.sparse.to_dense(sparse)
```

### ✔️ **Ragged Tensors**

* Tensors with irregular shapes (varying lengths along a dimension).

```python
ragged = tf.ragged.constant([[1, 2, 3], [4, 5]])
```

### ✔️ **String Tensors**

* Tensors holding text data.

```python
str_tensor = tf.constant(["hello", "tensorflow"])
```

---

## ✅ **5. Tensor Attributes and Properties**

* **`.shape`**: gives tensor shape.
* **`.numpy()`**: converts tensor to NumPy array.
* **`.dtype`**: data type of tensor.

**Example:**

```python
tensor = tf.constant([[10, 20], [30, 40]])

print("Shape:", tensor.shape)
print("Data type:", tensor.dtype)
print("NumPy array:", tensor.numpy())
```

**Output:**

```
Shape: (2, 2)
Data type: <dtype: 'int32'>
NumPy array: [[10 20]
              [30 40]]
```

---

## ✅ **6. Broadcasting in Tensors**

* Allows operations between tensors with different shapes.
* Automatically expands smaller tensors to match larger ones.

**Example:**

```python
tensor1 = tf.constant([[1, 2, 3]])
tensor2 = tf.constant([[4], [5], [6]])

result = tensor1 + tensor2
print(result.numpy())
```

**Explanation:**
Tensor1 (`1x3`) broadcasts to match Tensor2 (`3x1`), producing a `3x3` tensor.

**Output:**

```
[[5 6 7]
 [6 7 8]
 [7 8 9]]
```

---

## 📚 **Summary Table: Common Tensor Operations**

| Method                | Purpose                           | Example usage                  |
| --------------------- | --------------------------------- | ------------------------------ |
| **tf.constant**       | Immutable tensor                  | `tf.constant([1,2,3])`         |
| **tf.zeros**          | Tensor filled with zeros          | `tf.zeros([2,2])`              |
| **tf.ones**           | Tensor filled with ones           | `tf.ones([2,2])`               |
| **tf.fill**           | Tensor filled with specific value | `tf.fill([2,2], 7)`            |
| **tf.range**          | Sequence of numbers               | `tf.range(0,10,2)`             |
| **tf.random.uniform** | Random tensor within range        | `tf.random.uniform([2,2],0,5)` |
| **tf.shape**          | Shape of tensor                   | `tf.shape(tensor)`             |
| **tf.cast**           | Change tensor datatype            | `tf.cast(tensor, tf.float32)`  |
| **tf.reshape**        | Reshape tensor                    | `tf.reshape(tensor, [2,3])`    |
| **tf.transpose**      | Transpose tensor                  | `tf.transpose(tensor)`         |

---

## 📖 **References & Further Reading:**

* [TensorFlow Tensors Guide](https://www.tensorflow.org/guide/tensor)
* [TensorFlow API Documentation](https://www.tensorflow.org/api_docs/python/tf)
* [TensorFlow Broadcasting Guide](https://www.tensorflow.org/xla/broadcasting)

---


---

## 📝 **Problem 1**

Create a constant tensor with values `[5, 10, 15]`.

**Solution:**

```python
import tensorflow as tf
tensor = tf.constant([5, 10, 15])
```

**Explanation:**
This creates an immutable 1-dimensional tensor containing the integers 5, 10, and 15.

---

## 📝 **Problem 2**

Generate a tensor of zeros with shape `(4, 4)`.

**Solution:**

```python
tensor_zeros = tf.zeros([4, 4])
```

**Explanation:**
This generates a 4x4 tensor filled entirely with zeros, useful for initialization.

---

## 📝 **Problem 3**

Create a tensor filled with the number `7`, shape `(2, 3)`.

**Solution:**

```python
tensor_seven = tf.fill([2, 3], 7)
```

**Explanation:**
This creates a tensor of shape `(2, 3)` where each element is the number 7.

---

## 📝 **Problem 4**

Generate numbers from `0` to `9` as a tensor.

**Solution:**

```python
tensor_range = tf.range(10)
```

**Explanation:**
This generates a 1-dimensional tensor containing numbers from `0` to `9`.

---

## 📝 **Problem 5**

Create a tensor of random values between `0` and `1` with shape `(3, 3)`.

**Solution:**

```python
random_tensor = tf.random.uniform([3, 3], minval=0, maxval=1)
```

**Explanation:**
This creates a 3x3 tensor filled with random floating-point numbers between `0` (inclusive) and `1` (exclusive).

---

## 📝 **Problem 6**

Transpose a tensor of shape `(2, 3)`.

**Solution:**

```python
original_tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
transposed_tensor = tf.transpose(original_tensor)
```

**Explanation:**
The tensor shape changes from `(2, 3)` to `(3, 2)`, swapping rows with columns.

---

## 📝 **Problem 7**

Create a string tensor containing the words `"Tensor"` and `"Flow"`.

**Solution:**

```python
string_tensor = tf.constant(["Tensor", "Flow"])
```

**Explanation:**
Creates a tensor specifically to hold strings `"Tensor"` and `"Flow"`.

---

## 📝 **Problem 8**

Convert a float tensor `[2.9, 3.7]` to an integer tensor.

**Solution:**

```python
float_tensor = tf.constant([2.9, 3.7])
int_tensor = tf.cast(float_tensor, tf.int32)
```

**Explanation:**
This converts float values to integers by truncating decimals, resulting in `[2, 3]`.

---

## 📝 **Problem 9**

Compute the sum of elements of tensor `[1, 2, 3, 4]`.

**Solution:**

```python
sum_tensor = tf.reduce_sum(tf.constant([1, 2, 3, 4]))
```

**Explanation:**
Calculates the total sum of tensor elements, resulting in `10`.

---

## 📝 **Problem 10**

Reshape tensor from `(3, 2)` to `(2, 3)`.

**Solution:**

```python
original = tf.constant([[1, 2], [3, 4], [5, 6]])
reshaped = tf.reshape(original, [2, 3])
```

**Explanation:**
Reshapes the tensor from a 3-row, 2-column tensor into a 2-row, 3-column tensor.

---

## 📝 **Problem 11**

Expand dimensions of tensor `[1, 2, 3]` to shape `(1, 3)`.

**Solution:**

```python
tensor = tf.constant([1, 2, 3])
expanded_tensor = tf.expand_dims(tensor, axis=0)
```

**Explanation:**
Adds an extra dimension at position `0`, converting shape from `(3,)` to `(1, 3)`.

---

## 📝 **Problem 12**

Create a sparse tensor with indices `[0, 1]`, `[1, 0]`, values `[5, 6]`, and shape `[2, 2]`.

**Solution:**

```python
sparse_tensor = tf.sparse.SparseTensor(indices=[[0,1],[1,0]], values=[5,6], dense_shape=[2,2])
```

**Explanation:**
Creates a tensor storing only non-zero elements at positions `[0,1]` and `[1,0]`.

---

## 📝 **Problem 13**

Create a ragged tensor with values `[[1, 2, 3], [4, 5]]`.

**Solution:**

```python
ragged_tensor = tf.ragged.constant([[1, 2, 3], [4, 5]])
```

**Explanation:**
Creates a tensor with rows of differing lengths, suitable for irregular data.

---

## 📝 **Problem 14**

Compute element-wise multiplication of tensors `[1, 2]` and `[3, 4]`.

**Solution:**

```python
result = tf.multiply(tf.constant([1, 2]), tf.constant([3, 4]))
```

**Explanation:**
Multiplies each corresponding element producing `[3, 8]`.

---

## 📝 **Problem 15**

Compute matrix multiplication of tensors `(1x2)` and `(2x1)`.

**Solution:**

```python
result = tf.matmul(tf.constant([[1, 2]]), tf.constant([[3], [4]]))
```

**Explanation:**
Calculates matrix multiplication yielding `[[11]]`.

---

## 📝 **Problem 16**

Flatten tensor of shape `(2, 2)` to 1-dimensional tensor.

**Solution:**

```python
flattened = tf.reshape(tf.constant([[1, 2], [3, 4]]), [-1])
```

**Explanation:**
Converts tensor shape from `(2, 2)` to a flat `(4,)` tensor `[1,2,3,4]`.

---

## 📝 **Problem 17**

Get datatype of tensor `[1.5, 2.5]`.

**Solution:**

```python
dtype = tf.constant([1.5, 2.5]).dtype
```

**Explanation:**
Returns the datatype `tf.float32`.

---

## 📝 **Problem 18**

Check the shape of tensor `[[10, 20, 30], [40, 50, 60]]`.

**Solution:**

```python
shape = tf.constant([[10, 20, 30], [40, 50, 60]]).shape
```

**Explanation:**
Returns the shape `(2, 3)`.

---

## 📝 **Problem 19**

Create a tensor of ones with shape `(5,)`.

**Solution:**

```python
ones_tensor = tf.ones([5])
```

**Explanation:**
Generates a 1D tensor with 5 ones: `[1, 1, 1, 1, 1]`.

---

## 📝 **Problem 20**

Broadcast tensor `[1, 2, 3]` to add to `[[10], [20], [30]]`.

**Solution:**

```python
result = tf.constant([1, 2, 3]) + tf.constant([[10], [20], [30]])
```

**Explanation:**
Automatically expands the tensors’ dimensions to perform addition, resulting in:

```
[[11,12,13],
 [21,22,23],
 [31,32,33]]
```

---

