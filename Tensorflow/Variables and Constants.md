## 📚 TensorFlow Documentation: Variables and Constants

### 1. tf.Variable vs tf.Tensor

**Explanation:**

* `tf.Variable`: Mutable, can be updated and reassigned during model training.
* `tf.Tensor`: Immutable, fixed value after creation.

**Example:**

```python
import tensorflow as tf
# Tensor (immutable)
tensor = tf.constant([1, 2, 3])

# Variable (mutable)
variable = tf.Variable([1, 2, 3])
variable.assign([4, 5, 6])
print(variable.numpy())  # Output: [4, 5, 6]
```

### 2. Initialization and Assignment

**Explanation:**
Variables need initialization and can be reassigned using methods like `.assign()`.

**Example:**

```python
var = tf.Variable(tf.zeros([2, 2]))
print(var.numpy())  # Initial values: [[0, 0], [0, 0]]

# Assign new values
var.assign([[1, 2], [3, 4]])
print(var.numpy())  # Updated values: [[1, 2], [3, 4]]
```

### 3. Trainable vs Non-trainable Variables

**Explanation:**

* Trainable variables are updated during training (e.g., weights, biases).
* Non-trainable variables remain static.

**Example:**

```python
trainable_var = tf.Variable([1.0, 2.0, 3.0], trainable=True)
non_trainable_var = tf.Variable([4.0, 5.0, 6.0], trainable=False)

print(trainable_var.trainable)       # Output: True
print(non_trainable_var.trainable)   # Output: False
```

### 4. Variable Scope (TF 1.x)

**Explanation:**
In TensorFlow 1.x, `tf.variable_scope` helps organize variables and reuse them within scopes.

**Example:**

```python
# TensorFlow 1.x example
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

with tf.variable_scope('scope1'):
    v1 = tf.get_variable("var", shape=[2], initializer=tf.zeros_initializer())

with tf.variable_scope('scope1', reuse=True):
    v2 = tf.get_variable("var")

print(v1 is v2)  # True, both references point to the same variable.
```




### 20 Problems with Solutions and Explanations

### Problem 1: Create and Initialize a Variable

**Solution:**

```python
import tensorflow as tf
var = tf.Variable([10, 20, 30])
print(var.numpy())
```

**Explanation:** Creates and initializes a TensorFlow variable.

### Problem 2: Reassign Values to a Variable

**Solution:**

```python
var = tf.Variable([1, 2, 3])
var.assign([4, 5, 6])
print(var.numpy())
```

**Explanation:** Uses `.assign()` to update variable values.

### Problem 3: Difference Between Variable and Tensor

**Solution:**

```python
tensor = tf.constant([1, 2])
variable = tf.Variable([1, 2])
# tensor.assign([3, 4]) # Error, tensors are immutable
variable.assign([3, 4]) # Correct
print(variable.numpy())
```

**Explanation:** Demonstrates mutability differences.

### Problem 4: Check if Variable is Trainable

**Solution:**

```python
var = tf.Variable([1.0, 2.0], trainable=True)
print(var.trainable)
```

**Explanation:** Confirms variable is trainable.

### Problem 5: Create a Non-trainable Variable

**Solution:**

```python
var = tf.Variable([1, 2, 3], trainable=False)
print(var.trainable)
```

**Explanation:** Creates and checks non-trainable variable.

### Problem 6: Initialize Variable with Zeros

**Solution:**

```python
var = tf.Variable(tf.zeros([3, 3]))
print(var.numpy())
```

**Explanation:** Initializes variable with zeros.

### Problem 7: Variable Arithmetic

**Solution:**

```python
var = tf.Variable([2, 4, 6])
result = var + 2
print(result.numpy())
```

**Explanation:** Performs arithmetic operations on variables.

### Problem 8: Increment Variable Values

**Solution:**

```python
var = tf.Variable(10)
var.assign_add(5)
print(var.numpy())  # 15
```

**Explanation:** Increments the variable's value using `assign_add`.

### Problem 9: Variable Scope in TF 1.x

**Solution:**

```python
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
with tf.variable_scope('scope1'):
    v1 = tf.get_variable("var", shape=[2], initializer=tf.ones_initializer())
print(v1)
```

**Explanation:** Demonstrates variable scope management in TF 1.x.

### Problem 10: Convert Variable to Tensor

**Solution:**

```python
var = tf.Variable([1, 2, 3])
tensor = tf.convert_to_tensor(var)
print(tensor.numpy())
```

**Explanation:** Converts a variable into a tensor for further operations.

### Problem 11: Initialize Variable with Ones

**Solution:**

```python
var = tf.Variable(tf.ones([2, 2]))
print(var.numpy())
```

**Explanation:** Initializes variable with ones.

### Problem 12: Multiply Variable by a Scalar

**Solution:**

```python
var = tf.Variable([1, 2, 3])
result = var * 3
print(result.numpy())
```

**Explanation:** Multiplies each element of a variable by a scalar.

### Problem 13: Subtract from Variable

**Solution:**

```python
var = tf.Variable([10, 20, 30])
var.assign_sub([1, 2, 3])
print(var.numpy())
```

**Explanation:** Subtracts values from a variable using `assign_sub`.

### Problem 14: Reshape a Variable

**Solution:**

```python
var = tf.Variable(tf.range(6))
reshaped = tf.reshape(var, [2, 3])
print(reshaped.numpy())
```

**Explanation:** Reshapes a variable to a different shape.

### Problem 15: Check Variable Shape

**Solution:**

```python
var = tf.Variable([[1, 2], [3, 4]])
print(var.shape)
```

**Explanation:** Retrieves the shape of a variable.

### Problem 16: Create Variable with Random Values

**Solution:**

```python
var = tf.Variable(tf.random.uniform([3]))
print(var.numpy())
```

**Explanation:** Initializes variable with random values.

### Problem 17: Create a Boolean Variable

**Solution:**

```python
var = tf.Variable([True, False, True])
print(var.numpy())
```

**Explanation:** Initializes a variable with boolean values.

### Problem 18: Check Data Type of Variable

**Solution:**

```python
var = tf.Variable([1, 2, 3])
print(var.dtype)
```

**Explanation:** Checks the data type of the variable.

### Problem 19: Create Variable from NumPy Array

**Solution:**

```python
import numpy as np
numpy_array = np.array([1, 2, 3])
var = tf.Variable(numpy_array)
print(var.numpy())
```

**Explanation:** Converts a NumPy array to a TensorFlow variable.

### Problem 20: Variable Initialization Check (TF 1.x)

**Solution:**

```python
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
var = tf.Variable([1, 2, 3])
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(var))
```

**Explanation:** Initializes variables in TensorFlow 1.x.
