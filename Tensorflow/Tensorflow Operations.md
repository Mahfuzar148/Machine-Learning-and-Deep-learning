## 📚 TensorFlow Documentation: Tensor Operations

### 1. Indexing and Slicing Tensors

Extract specific elements or slices from tensors.

```python
import tensorflow as tf
tensor = tf.constant([[1, 2, 3], [4, 5, 6]])

# Extract element [row 0, col 1]
print(tensor[0, 1].numpy())  # Output: 2

# Slice first row
print(tensor[0, :].numpy())  # Output: [1, 2, 3]
```

### 2. Reshaping Tensors

Modify tensor dimensions without changing data.

#### reshape:

```python
tensor = tf.range(6)
reshaped_tensor = tf.reshape(tensor, [2, 3])
print(reshaped_tensor.numpy())
```

#### expand\_dims:

```python
tensor = tf.constant([1, 2, 3])
expanded_tensor = tf.expand_dims(tensor, axis=1)
print(expanded_tensor.numpy())  # shape: (3,1)
```

#### squeeze:

```python
tensor = tf.constant([[[1], [2], [3]]])
squeezed_tensor = tf.squeeze(tensor)
print(squeezed_tensor.numpy())  # shape: (3,)
```

### 3. Math Operations

Perform mathematical computations.

#### Element-wise Operations:

```python
a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])

# Add
print(tf.add(a, b).numpy())  # [5, 7, 9]

# Multiply
print(tf.multiply(a, b).numpy())  # [4, 10, 18]

# Square
print(tf.square(a).numpy())  # [1, 4, 9]
```

### 4. Matrix Operations

Perform matrix algebra operations.

#### matmul:

```python
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[5, 6], [7, 8]])
print(tf.matmul(a, b).numpy())
```

#### transpose:

```python
tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
print(tf.transpose(tensor).numpy())
```

#### inverse:

```python
matrix = tf.constant([[1., 2.], [3., 4.]])
print(tf.linalg.inv(matrix).numpy())
```

### 5. Reduction Operations

Reduce tensor dimensions by computing aggregate values.

```python
tensor = tf.constant([[1, 2], [3, 4]])

# Sum
print(tf.reduce_sum(tensor).numpy())  # 10

# Mean
print(tf.reduce_mean(tensor).numpy())  # 2.5

# Max
print(tf.reduce_max(tensor).numpy())  # 4
```

### 6. Logical Operations

Apply logical tests element-wise.

#### tf.equal:

```python
a = tf.constant([1, 2, 3])
b = tf.constant([3, 2, 1])
print(tf.equal(a, b).numpy())  # [False, True, False]
```

#### tf.greater:

```python
print(tf.greater(a, b).numpy())  # [False, False, True]
```

#### tf.where:

```python
condition = tf.constant([True, False, True])
x = tf.constant([1, 2, 3])
y = tf.constant([4, 5, 6])
print(tf.where(condition, x, y).numpy())  # [1, 5, 3]
```

### 7. Type Casting

Change tensor data types.

```python
float_tensor = tf.constant([1.7, 2.3])
int_tensor = tf.cast(float_tensor, tf.int32)
print(int_tensor.numpy())  # [1, 2]
```
## 📚 TensorFlow Documentation: 20 Tensor Operations Problems with Solutions

### Problem 1: Extracting Last Column

**Solution:**

```python
import tensorflow as tf
tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
last_column = tensor[:, -1]
print(last_column.numpy())  # Output: [3, 6]
```

**Explanation:** Extracts the last column elements from each row.

### Problem 2: Reverse Tensor Rows

**Solution:**

```python
tensor = tf.constant([[1, 2], [3, 4], [5, 6]])
reversed_tensor = tensor[::-1]
print(reversed_tensor.numpy())  # [[5,6],[3,4],[1,2]]
```

**Explanation:** Reverses the order of rows.

### Problem 3: Increase Tensor Dimension

**Solution:**

```python
tensor = tf.constant([4, 5, 6])
new_tensor = tf.expand_dims(tensor, axis=-1)
print(new_tensor.numpy())  # shape: (3,1)
```

**Explanation:** Adds a new dimension at the end.

### Problem 4: Remove Single-dimensional Entries

**Solution:**

```python
tensor = tf.constant([[[10], [20]]])
squeezed = tf.squeeze(tensor)
print(squeezed.numpy())  # shape: (2,)
```

**Explanation:** Removes dimensions with size one.

### Problem 5: Element-wise Division

**Solution:**

```python
a = tf.constant([10, 20, 30])
b = tf.constant([2, 4, 6])
divided = tf.divide(a, b)
print(divided.numpy())  # [5,5,5]
```

**Explanation:** Divides corresponding elements.

### Problem 6: Element-wise Subtraction

**Solution:**

```python
a = tf.constant([10, 20, 30])
b = tf.constant([1, 2, 3])
subtracted = tf.subtract(a, b)
print(subtracted.numpy())  # [9,18,27]
```

**Explanation:** Subtracts elements of one tensor from another.

### Problem 7: Dot Product

**Solution:**

```python
a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])
dot_product = tf.tensordot(a, b, axes=1)
print(dot_product.numpy())  # 32
```

**Explanation:** Computes the dot product.

### Problem 8: Trace of Matrix

**Solution:**

```python
matrix = tf.constant([[1, 2], [3, 4]])
trace = tf.linalg.trace(matrix)
print(trace.numpy())  # 5
```

**Explanation:** Computes the sum of diagonal elements.

### Problem 9: Norm of a Vector

**Solution:**

```python
vector = tf.constant([3., 4.])
norm = tf.norm(vector)
print(norm.numpy())  # 5.0
```

**Explanation:** Calculates Euclidean norm.

### Problem 10: Finding Index of Maximum Value

**Solution:**

```python
tensor = tf.constant([1, 3, 2, 5, 4])
index = tf.argmax(tensor)
print(index.numpy())  # 3
```

**Explanation:** Finds the position of the maximum value.

### Problem 11: Logical AND Operation

**Solution:**

```python
a = tf.constant([True, False, True])
b = tf.constant([False, False, True])
result = tf.logical_and(a, b)
print(result.numpy())  # [False, False, True]
```

**Explanation:** Performs element-wise logical AND.

### Problem 12: Logical OR Operation

**Solution:**

```python
a = tf.constant([True, False, True])
b = tf.constant([False, True, False])
result = tf.logical_or(a, b)
print(result.numpy())  # [True, True, True]
```

**Explanation:** Performs element-wise logical OR.

### Problem 13: Conditional Selection

**Solution:**

```python
condition = tf.constant([False, True, False])
x = tf.constant([10, 20, 30])
y = tf.constant([40, 50, 60])
selected = tf.where(condition, x, y)
print(selected.numpy())  # [40,20,60]
```

**Explanation:** Selects elements based on condition.

### Problem 14: Reduce Product of Elements

**Solution:**

```python
tensor = tf.constant([1, 2, 3, 4])
product = tf.reduce_prod(tensor)
print(product.numpy())  # 24
```

**Explanation:** Multiplies all elements.

### Problem 15: Compute Mean Along Columns

**Solution:**

```python
tensor = tf.constant([[1, 2], [3, 4]])
mean = tf.reduce_mean(tensor, axis=0)
print(mean.numpy())  # [2., 3.]
```

**Explanation:** Calculates column-wise mean.

### Problem 16: Compute Variance

**Solution:**

```python
tensor = tf.constant([1., 2., 3.])
variance = tf.math.reduce_variance(tensor)
print(variance.numpy())  # ~0.6667
```

**Explanation:** Computes variance of tensor values.

### Problem 17: Rounding Tensor Values

**Solution:**

```python
tensor = tf.constant([1.1, 2.5, 3.9])
rounded = tf.round(tensor)
print(rounded.numpy())  # [1., 2., 4.]
```

**Explanation:** Rounds tensor values to nearest integer.

### Problem 18: Checking for Finite Values

**Solution:**

```python
tensor = tf.constant([1., float('inf'), 3.])
finite_check = tf.math.is_finite(tensor)
print(finite_check.numpy())  # [True, False, True]
```

**Explanation:** Checks if tensor elements are finite.

### Problem 19: Changing Tensor Type to Float64

**Solution:**

```python
tensor = tf.constant([1, 2, 3])
float_tensor = tf.cast(tensor, tf.float64)
print(float_tensor.numpy())  # [1.0, 2.0, 3.0]
```

**Explanation:** Casts integer tensor to float64.

### Problem 20: Reverse Elements in Tensor

**Solution:**

```python
tensor = tf.constant([1, 2, 3, 4, 5])
reversed_tensor = tf.reverse(tensor, axis=[0])
print(reversed_tensor.numpy())  # [5,4,3,2,1]
```

**Explanation:** Reverses tensor elements.
