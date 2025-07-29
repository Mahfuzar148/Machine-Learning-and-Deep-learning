## 📚 TensorFlow Documentation: 5 Full Model Examples with TensorBoard Integration

---

## 💾 Model Saving and Serialization

### Explanation:

TensorFlow/Keras provides options to save entire models (including architecture, weights, and optimizer state), only weights, or models in different formats such as HDF5 and SavedModel.

### 1. Saving Entire Model (SavedModel format)

```python
import tensorflow as tf
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.save("saved_model_format")  # SavedModel format (folder)
```

### 2. Saving Entire Model (HDF5 format)

```python
model.save("model_hdf5_format.h5")  # HDF5 file
```

### 3. Saving and Loading Only Weights

```python
model.save_weights("model_weights_only.h5")  # Save weights

# Rebuild model architecture to load weights into
new_model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(1)
])
new_model.load_weights("model_weights_only.h5")
```

### 4. Loading a Full Model from HDF5

```python
loaded_model = tf.keras.models.load_model("model_hdf5_format.h5")
loaded_model.summary()
```

### 5. Loading a Full Model from SavedModel

```python
loaded_model = tf.keras.models.load_model("saved_model_format")
loaded_model.summary()
```

### Summary of Formats

* `SavedModel` (default): stores a model as a directory with model configuration, weights, and optimizer state.
* `HDF5` (`.h5`): stores the model in a single file, compatible with older Keras.
* `model.save()`: can save in either format.
* `model.save_weights()`: saves only the layer weights.

These examples illustrate how to persist models during or after training for reuse, deployment, or sharing.


## 📚 TensorFlow Documentation: Full Model Examples by Topic

---

## 💾 Model Saving and Serialization — Model 1 (Regression Model with SavedModel)

```python
import tensorflow as tf
import numpy as np

x = np.random.rand(100, 5)
y = 3 * x[:, 0] + 2 * x[:, 1] + np.random.randn(100)
y = y.reshape(-1, 1)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=3)
model.save("saved_model_format")  # Save model in SavedModel format

# Load and evaluate the model
loaded_model = tf.keras.models.load_model("saved_model_format")
print("Evaluation on new data:", loaded_model.evaluate(x, y))
```

---

## 💾 Model Saving with HDF5 — Model 2 (Binary Classification)

```python
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784) / 255.0
y_train = (y_train % 2 == 0).astype(int)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=2, batch_size=128)
model.save("model_hdf5_format.h5")  # Save model in HDF5 format

# Load and evaluate
loaded_model = tf.keras.models.load_model("model_hdf5_format.h5")
print("Accuracy on training set:", loaded_model.evaluate(x_train, y_train))
```

---

## 💾 Saving and Loading Only Weights — Model 3 (Small Dense Network)

```python
x = tf.random.normal((30, 3))
y = tf.reduce_sum(x, axis=1, keepdims=True)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=2)
model.save_weights("model_weights_only.h5")  # Save only weights

# Load weights into identical model
new_model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(1)
])
new_model.compile(optimizer='adam', loss='mse')
new_model.load_weights("model_weights_only.h5")
print("Restored model evaluation:", new_model.evaluate(x, y))
```

---

## 💾 Loading from HDF5 — Model 4 (Reload and Predict)

```python
from tensorflow.keras.models import load_model
import numpy as np

loaded_model = load_model("model_hdf5_format.h5")
loaded_model.summary()
sample_input = np.random.rand(1, 784)
prediction = loaded_model.predict(sample_input)
print("Prediction from loaded HDF5 model:", prediction)
```

---

## 💾 Loading from SavedModel — Model 5 (Reload and Evaluate)

```python
from tensorflow.keras.models import load_model

loaded_model = load_model("saved_model_format")
loaded_model.summary()
test_data = tf.random.normal((10, 5))
test_output = loaded_model.predict(test_data)
print("Predicted Outputs from SavedModel:\n", test_output)
```

---

## Summary of Formats

* `SavedModel`: stores model in directory with config, weights, and optimizer state.
* `HDF5`: stores model in a single `.h5` file.
* `model.save()`: use with path to determine format.
* `model.save_weights()`: saves only model weights.

Each model above demonstrates a different method of saving or restoring model components, with loading and evaluation included to complete the serialization workflow.

---
