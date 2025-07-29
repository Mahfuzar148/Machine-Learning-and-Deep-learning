
---

# 📘 1. Introduction to TensorFlow

---

## ✅ What is TensorFlow?

**TensorFlow** is an open-source, end-to-end machine learning (ML) and deep learning framework developed by the **Google Brain Team**. It provides a flexible ecosystem of tools, libraries, and community resources that help researchers and developers build and deploy ML-powered applications.

### Key Features:

* **Ecosystem**: Includes TensorFlow Lite, TensorFlow\.js, TFX (TensorFlow Extended), and TF-Agents.
* **Multiplatform**: Runs on CPU, GPU, TPU; supports mobile, web, edge, and cloud.
* **Language Support**: Primarily Python, with bindings for C++, JavaScript, Java, and Swift.

---

## 🕰️ History and Versions

| Version    | Release Year | Key Features                                                                                                                      |
| ---------- | ------------ | --------------------------------------------------------------------------------------------------------------------------------- |
| **TF 1.x** | 2015–2019    | Graph-based computation, static execution, verbose syntax. Used sessions (`tf.Session()`), manual control flow.                   |
| **TF 2.x** | 2019–present | Eager execution by default, intuitive APIs (e.g., Keras), tight integration with Python. Streamlined for research and production. |

### Why TF 2.x is Preferred:

* Simpler debugging with native Python control flow
* Built-in support for Keras
* Eager execution for dynamic computation
* `tf.function` decorator allows switching to graph mode for performance

---

## 🏗️ TensorFlow Architecture

TensorFlow's architecture consists of the following layers:

### 1. **Frontend API Layer**

* High-level APIs like `tf.keras`, `Estimators`, and `Functional API`
* Simplifies model building and experimentation

### 2. **TensorFlow Core (Execution Engine)**

* Defines and executes computational graphs
* Uses **Tensors** as the fundamental unit of data
* Manages device placement (CPU/GPU/TPU)

### 3. **Distributed Runtime**

* Manages distributed computing, parallelization
* Supports multiple devices across multiple machines

### 4. **Hardware Abstraction Layer**

* Allows TensorFlow to run on CPUs, GPUs, TPUs, mobile devices

---

## ⚙️ Installation and Setup

### ✅ 1. **Via pip (Recommended)**

```bash
pip install tensorflow
```

* Installs CPU version by default.
* For GPU version (ensure CUDA and cuDNN are compatible):

```bash
pip install tensorflow-gpu
```

### ✅ 2. **Via Conda**

```bash
conda create -n tf_env tensorflow
conda activate tf_env
```

### ✅ 3. **From Source (Advanced Users)**

* Clone the repository:

```bash
git clone https://github.com/tensorflow/tensorflow.git
```

* Build using Bazel (requires toolchain setup)

### ✅ 4. **System Requirements**

* Python 3.7–3.11
* Pip ≥ 19.0
* Optional GPU: CUDA 11.x + cuDNN

👉 Full setup guide: [https://www.tensorflow.org/install](https://www.tensorflow.org/install)

---

## ⚡ Eager Execution vs. Graph Execution

| Feature         | Eager Execution                 | Graph Execution                  |
| --------------- | ------------------------------- | -------------------------------- |
| **Definition**  | Operations executed immediately | Operations compiled into a graph |
| **Debugging**   | Easy (uses Python tools)        | Harder (requires graph tools)    |
| **Flexibility** | Dynamic computation             | Static computation               |
| **Performance** | Lower for large models          | Optimized for performance        |
| **Best Use**    | Prototyping, quick debugging    | Production training, deployment  |

### Example: Eager vs Graph

```python
import tensorflow as tf

# Eager execution (default in TF 2.x)
a = tf.constant(2.0)
b = tf.constant(3.0)
print(a + b)  # Direct output: tf.Tensor(5.0, shape=(), dtype=float32)

# Graph execution with tf.function
@tf.function
def add_tensors(x, y):
    return x + y

print(add_tensors(a, b))  # Also produces a tensor but through compiled graph
```

---

## 📎 Documentation Links

* [Official TensorFlow Introduction](https://www.tensorflow.org/overview)
* [Installing TensorFlow](https://www.tensorflow.org/install)
* [TensorFlow Guide: Eager Execution](https://www.tensorflow.org/guide/eager)
* [Differences: TF 1.x vs TF 2.x](https://www.tensorflow.org/guide/migrate)

---
Here's detailed **documentation** for **20 basic TensorFlow examples**, clearly explained with code snippets:

---

# 📘 **20 Basic TensorFlow Examples Documentation**

---

### ✅ **1. Importing TensorFlow**

TensorFlow must first be imported to use its functionalities.
**Code:**

```python
import tensorflow as tf
print(tf.__version__)
```

**Explanation:**
This imports TensorFlow as `tf` and prints the installed version.

---

### ✅ **2. Creating a Constant Tensor**

Creates immutable tensors with fixed values.
**Code:**

```python
tensor = tf.constant([[1, 2], [3, 4]])
print(tensor)
```

**Explanation:**
Creates a 2x2 tensor with predefined values.

---

### ✅ **3. Creating a Tensor of Zeros**

Initializes a tensor filled with zeros.
**Code:**

```python
zero_tensor = tf.zeros([2, 3])
print(zero_tensor)
```

**Explanation:**
Generates a tensor of shape `2x3` filled with zeros.

---

### ✅ **4. Creating a Tensor of Ones**

Generates tensors filled entirely with ones.
**Code:**

```python
ones_tensor = tf.ones([3, 2])
print(ones_tensor)
```

**Explanation:**
Creates a tensor with dimensions `3x2` containing all ones.

---

### ✅ **5. Random Tensor**

Creates a tensor populated with random numbers.
**Code:**

```python
rand_tensor = tf.random.uniform([2, 2], minval=0, maxval=10)
print(rand_tensor)
```

**Explanation:**
Produces a 2x2 tensor with random floating-point values between 0 and 10.

---

### ✅ **6. Tensor Addition**

Adds two tensors element-wise.
**Code:**

```python
a = tf.constant([1, 2])
b = tf.constant([3, 4])
print(tf.add(a, b))
```

**Explanation:**
Element-wise addition resulting in `[4, 6]`.

---

### ✅ **7. Tensor Multiplication**

Multiplies two tensors element-wise.
**Code:**

```python
a = tf.constant([1, 2])
b = tf.constant([3, 4])
print(tf.multiply(a, b))
```

**Explanation:**
Returns `[3, 8]`, which is the element-wise product.

---

### ✅ **8. Tensor Matrix Multiplication**

Performs matrix multiplication.
**Code:**

```python
m1 = tf.constant([[1, 2]])
m2 = tf.constant([[3], [4]])
print(tf.matmul(m1, m2))
```

**Explanation:**
Calculates matrix multiplication resulting in `[[11]]`.

---

### ✅ **9. Reshape Tensor**

Reshapes tensor dimensions.
**Code:**

```python
reshaped = tf.reshape(tf.range(9), [3, 3])
print(reshaped)
```

**Explanation:**
Transforms a 1D tensor of range 0-8 into a 3x3 matrix.

---

### ✅ **10. Flatten Tensor**

Flattens a tensor into 1D.
**Code:**

```python
flat = tf.reshape(reshaped, [-1])
print(flat)
```

**Explanation:**
Transforms 3x3 tensor back into a 1D array `[0,1,2,3,4,5,6,7,8]`.

---

### ✅ **11. Tensor Transpose**

Swaps rows and columns.
**Code:**

```python
transposed = tf.transpose(reshaped)
print(transposed)
```

**Explanation:**
Switches rows and columns of a matrix.

---

### ✅ **12. Expand Dimensions**

Adds an extra dimension to a tensor.
**Code:**

```python
tensor = tf.constant([1, 2, 3])
expanded = tf.expand_dims(tensor, 0)
print(expanded)
```

**Explanation:**
Transforms tensor shape from `[3]` to `[1, 3]`.

---

### ✅ **13. Squeeze Dimensions**

Removes single-dimensional entries.
**Code:**

```python
squeezed = tf.squeeze(expanded)
print(squeezed)
```

**Explanation:**
Removes dimensions of size 1, reverting `[1, 3]` back to `[3]`.

---

### ✅ **14. Tensor Slicing**

Extracts tensor subsets.
**Code:**

```python
tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
print(tensor[:, 1:])
```

**Explanation:**
Slices tensor to get second and third columns, resulting in `[[2,3],[5,6]]`.

---

### ✅ **15. String Tensor**

Creates tensors with strings.
**Code:**

```python
str_tensor = tf.constant(["hello", "tensorflow"])
print(str_tensor)
```

**Explanation:**
Forms a tensor with string elements.

---

### ✅ **16. Casting Tensor Types**

Changes tensor datatype.
**Code:**

```python
f_tensor = tf.constant([1.7, 2.8])
i_tensor = tf.cast(f_tensor, tf.int32)
print(i_tensor)
```

**Explanation:**
Casts float tensor to integers, giving `[1, 2]`.

---

### ✅ **17. Reduce Sum**

Sums all tensor elements.
**Code:**

```python
tensor = tf.constant([[1, 2], [3, 4]])
print(tf.reduce_sum(tensor))
```

**Explanation:**
Calculates total sum of elements, resulting in `10`.

---

### ✅ **18. Creating Variables**

Creates mutable tensor variables.
**Code:**

```python
var = tf.Variable([1, 2, 3])
print(var)
```

**Explanation:**
Defines a TensorFlow variable, allowing updates during training.

---

### ✅ **19. Using tf.function**

Optimizes Python functions via graph compilation.
**Code:**

```python
@tf.function
def add(a, b):
    return a + b

print(add(tf.constant(2), tf.constant(3)))
```

**Explanation:**
Defines a function compiled into TensorFlow's graph mode for efficiency.

---

### ✅ **20. Gradient Calculation**

Calculates gradients using automatic differentiation.
**Code:**

```python
x = tf.Variable(3.0)
with tf.GradientTape() as tape:
    y = x ** 2
grad = tape.gradient(y, x)
print(grad)
```

**Explanation:**
Computes derivative of y = x² at x=3, resulting in gradient `6.0`.

---

## 📚 **Additional Documentation and Resources:**

* [TensorFlow Official Guide](https://www.tensorflow.org/guide)
* [TensorFlow API Reference](https://www.tensorflow.org/api_docs/python/tf)
* [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)

---


