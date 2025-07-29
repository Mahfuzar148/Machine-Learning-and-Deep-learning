
---
## 🔧 Custom Training and Autodiff in TensorFlow

### 1. Using `tf.GradientTape` for Automatic Differentiation

**Explanation:**
`tf.GradientTape` records operations for automatic gradient computation.

**Example:**

```python
x = tf.constant(3.0)
with tf.GradientTape() as tape:
    tape.watch(x)
    y = x * x

dy_dx = tape.gradient(y, x)
print("dy/dx:", dy_dx)
```

### 2. Custom Training Loop with `train_step`

**Explanation:**
Customize model training using a manual loop to define forward pass, loss, and gradient updates.

**Example:**

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
])

optimizer = tf.keras.optimizers.SGD()
loss_fn = tf.keras.losses.MeanSquaredError()

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x)
        loss = loss_fn(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

x_train = tf.random.normal((100, 1))
y_train = 3 * x_train + tf.random.normal((100, 1))
for epoch in range(5):
    loss = train_step(x_train, y_train)
    print(f"Epoch {epoch+1}, Loss: {loss.numpy()}")
```

### 3. Writing Custom Loss Functions and Metrics

**Explanation:**
Create specialized loss and metric functions tailored to your application.

**Example - Custom Loss:**

```python
def custom_mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))
```

**Example - Custom Metric:**

```python
class MeanPred(tf.keras.metrics.Metric):
    def __init__(self, name='mean_pred', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.total.assign_add(tf.reduce_sum(y_pred))
        self.count.assign_add(tf.cast(tf.size(y_pred), tf.float32))

    def result(self):
        return self.total / self.count
```

### 4. Gradient Clipping and Accumulation

**Explanation:**
Stabilize training by clipping gradients or accumulating them over multiple steps.

**Gradient Clipping Example:**

```python
grads = tape.gradient(loss, model.trainable_variables)
clipped_grads = [tf.clip_by_norm(g, 1.0) for g in grads]
optimizer.apply_gradients(zip(clipped_grads, model.trainable_variables))
```

**Gradient Accumulation Concept:**
Accumulate gradients over `n` steps before updating weights:

```python
gradients_accum = [tf.zeros_like(var) for var in model.trainable_variables]
for i in range(accum_steps):
    with tf.GradientTape() as tape:
        preds = model(x_batch)
        loss = loss_fn(y_batch, preds)
    gradients = tape.gradient(loss, model.trainable_variables)
    gradients_accum = [accum + grad for accum, grad in zip(gradients_accum, gradients)]
optimizer.apply_gradients(zip(gradients_accum, model.trainable_variables))
```

This section covers key tools in building low-level and flexible training routines in TensorFlow.



---


### Model 1: Regression with Functional API

```python
import tensorflow as tf
import numpy as np

x = np.linspace(0, 100, 1000)
y = 3 * x + 7 + np.random.randn(*x.shape) * 10
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

inputs = tf.keras.Input(shape=(1,))
x_layer = tf.keras.layers.Dense(16, activation='relu')(inputs)
outputs = tf.keras.layers.Dense(1)(x_layer)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(x, y, epochs=5)
model.evaluate(x, y)
```

### Model 2: Text Classification with Preprocessing Layer

```python
import tensorflow as tf

texts = ["great movie", "bad film", "nice acting", "worst direction"]
labels = [1, 0, 1, 0]

vectorizer = tf.keras.layers.TextVectorization(max_tokens=1000)
vectorizer.adapt(texts)

model = tf.keras.Sequential([
    vectorizer,
    tf.keras.layers.Embedding(1000, 64),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(tf.constant(texts), tf.constant(labels), epochs=5)
```

### Model 3: Image Classification with Preprocessing and Augmentation

```python
import tensorflow as tf

(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train[..., tf.newaxis] / 255.0

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(32)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1)
])

model = tf.keras.Sequential([
    data_augmentation,
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(dataset, epochs=5)
```

### Model 4: Multi-Input Model

```python
import tensorflow as tf

input_1 = tf.keras.Input(shape=(10,), name='wide_input')
input_2 = tf.keras.Input(shape=(5,), name='deep_input')

x1 = tf.keras.layers.Dense(16, activation='relu')(input_1)
x2 = tf.keras.layers.Dense(32, activation='relu')(input_2)
x2 = tf.keras.layers.Dense(16, activation='relu')(x2)

concat = tf.keras.layers.concatenate([x1, x2])
output = tf.keras.layers.Dense(1, activation='sigmoid')(concat)

model = tf.keras.Model(inputs=[input_1, input_2], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

import numpy as np
x1 = np.random.random((100, 10))
x2 = np.random.random((100, 5))
y = np.random.randint(2, size=(100, 1))

model.fit({'wide_input': x1, 'deep_input': x2}, y, epochs=5)
```

### Model 5: Custom Training Loop with Classification

```python
import tensorflow as tf

(x_train, y_train), _ = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.reshape(-1, 784) / 255.0

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(64)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        logits = model(images)
        loss = loss_fn(labels, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

for epoch in range(5):
    for images, labels in dataset:
        loss = train_step(images, labels)
    print(f"Epoch {epoch+1}, Loss: {loss.numpy()}")
```

These models explore additional use-cases with Functional API, custom training loops, multi-inputs, preprocessing, and different data types.
