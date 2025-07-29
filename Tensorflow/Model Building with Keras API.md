## 📚 TensorFlow Documentation: Model Building with Keras API

### 1. Sequential Model

**Explanation:**
The Sequential API is straightforward, ideal for linear stacks of layers.

**Example:**

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1)
])
model.summary()
```

### 2. Functional API

**Explanation:**
The Functional API allows building complex models with multiple inputs/outputs and shared layers.

**Example:**

```python
inputs = tf.keras.Input(shape=(10,))
x = tf.keras.layers.Dense(32, activation='relu')(inputs)
outputs = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs, outputs)
model.summary()
```

### 3. Subclassing tf.keras.Model

**Explanation:**
Subclassing provides the highest flexibility, allowing the creation of custom layers and methods.

**Example:**

```python
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

model = MyModel()
model.build(input_shape=(None, 10))
model.summary()
```

### 4. Building Custom Layers (tf.keras.layers.Layer)

**Explanation:**
Custom layers help define new behavior or computations not available in built-in layers.

**Example:**

```python
class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, units=32):
        super(CustomLayer, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal', trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w)

inputs = tf.keras.Input(shape=(10,))
outputs = CustomLayer(32)(inputs)
model = tf.keras.Model(inputs, outputs)
model.summary()
```

### 5. Model Summary and Configuration

**Explanation:**
`model.summary()` provides details on the layers, output shapes, and parameter counts. `model.get_config()` returns the model configuration for reproducibility.

**Example:**

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1)
])

model.summary()
config = model.get_config()
print(config)
```
## 📚 TensorFlow Documentation: Model Building with Keras API

### 10 Problems with Solutions and Explanations

### Problem 1: Build a simple Sequential Model for binary classification

**Solution:**

```python
import tensorflow as tf
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(8,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.summary()
```

**Explanation:** Creates a sequential neural network suitable for binary classification tasks.

### Problem 2: Construct a Functional API Model with two inputs

**Solution:**

```python
input1 = tf.keras.Input(shape=(5,))
input2 = tf.keras.Input(shape=(5,))
x = tf.keras.layers.concatenate([input1, input2])
x = tf.keras.layers.Dense(10, activation='relu')(x)
output = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs=[input1, input2], outputs=output)
model.summary()
```

**Explanation:** Demonstrates combining multiple inputs using the Functional API.

### Problem 3: Subclassing tf.keras.Model for Regression

**Solution:**

```python
class RegressionModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs):
        return self.dense(inputs)

model = RegressionModel()
model.build(input_shape=(None, 3))
model.summary()
```

**Explanation:** Defines a simple regression model using subclassing.

### Problem 4: Creating Custom Layer with Bias

**Solution:**

```python
class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, units=32):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(self.units,), initializer='zeros', trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

inputs = tf.keras.Input(shape=(4,))
outputs = CustomLayer(8)(inputs)
model = tf.keras.Model(inputs, outputs)
model.summary()
```

**Explanation:** Adds a custom bias term to a layer, providing greater flexibility.

### Problem 5: Retrieve Model Configuration

**Solution:**

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(2)
])
config = model.get_config()
print(config)
```

**Explanation:** Retrieves and prints the model configuration details.

### Problem 6: Build and Compile Model for Multi-class Classification

**Solution:**

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
```

**Explanation:** Compiles the model to classify input into multiple classes.

### Problem 7: Create a Model using Functional API with shared layers

**Solution:**

```python
shared_layer = tf.keras.layers.Dense(8, activation='relu')
input_a = tf.keras.Input(shape=(4,))
input_b = tf.keras.Input(shape=(4,))
output_a = shared_layer(input_a)
output_b = shared_layer(input_b)
concat = tf.keras.layers.concatenate([output_a, output_b])
output = tf.keras.layers.Dense(1)(concat)
model = tf.keras.Model(inputs=[input_a, input_b], outputs=output)
model.summary()
```

**Explanation:** Demonstrates layer sharing using Functional API.

### Problem 8: Create a Custom Layer with Activation

**Solution:**

```python
class ActivationLayer(tf.keras.layers.Layer):
    def __init__(self, units=16, activation='relu'):
        super().__init__()
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer='random_normal')

    def call(self, inputs):
        return self.activation(tf.matmul(inputs, self.w))

inputs = tf.keras.Input(shape=(4,))
outputs = ActivationLayer()(inputs)
model = tf.keras.Model(inputs, outputs)
model.summary()
```

**Explanation:** Combines custom layer operations with built-in activation functions.

### Problem 9: Subclass tf.keras.Model with multiple outputs

**Solution:**

```python
class MultiOutputModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(16, activation='relu')
        self.out1 = tf.keras.layers.Dense(1)
        self.out2 = tf.keras.layers.Dense(2)

    def call(self, inputs):
        x = self.dense(inputs)
        return self.out1(x), self.out2(x)

model = MultiOutputModel()
model.build(input_shape=(None, 4))
model.summary()
```

**Explanation:** Implements a model that outputs multiple predictions simultaneously.

### Problem 10: Save and Load a Keras Model

**Solution:**

```python
model = tf.keras.Sequential([tf.keras.layers.Dense(2, input_shape=(4,))])
model.save('model.h5')
loaded_model = tf.keras.models.load_model('model.h5')
loaded_model.summary()
```

**Explanation:** Demonstrates saving and loading a Keras model to and from a file.

