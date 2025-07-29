
---

## 🎯 1. Built-in **Loss Functions**

Loss functions guide model training by penalizing incorrect predictions.

### 🔸 1.1 `categorical_crossentropy`

Used for **multi-class classification** with **one-hot encoded** labels.

```python
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

> ✅ Use this when `y_train` is one-hot encoded:

```python
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train, num_classes=10)
```

---

### 🔸 1.2 `mean_squared_error` (MSE)

Used for **regression** tasks to measure squared differences.

```python
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
```

---

### 🔸 1.3 `huber_loss`

Combines MSE and MAE — less sensitive to outliers than MSE.

```python
model.compile(loss='huber', optimizer='adam', metrics=['mae'])
```

Or explicitly:

```python
from tensorflow.keras.losses import Huber

model.compile(loss=Huber(delta=1.0), optimizer='adam')
```

---

## ✅ 2. Built-in **Metrics**

Metrics help evaluate model performance (do not affect training).

### 🔹 2.1 `accuracy`

Works with both classification and binary tasks.

```python
metrics=['accuracy']
```

---

### 🔹 2.2 `precision`, `recall`, and `AUC`

These are useful in **imbalanced classification** or when **false positives/negatives** matter.

```python
from tensorflow.keras.metrics import Precision, Recall, AUC

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy', Precision(), Recall(), AUC()]
)
```

---

## 🧪 3. Custom Loss Functions

Define your own loss when you need more control.

### Example: Custom MSE + L1 combo loss

```python
import tensorflow.keras.backend as K

def custom_loss(y_true, y_pred):
    mse = K.mean(K.square(y_true - y_pred))
    l1 = K.mean(K.abs(y_true - y_pred))
    return mse + 0.5 * l1

model.compile(optimizer='adam', loss=custom_loss)
```

---

### Example: Focal Loss (for class imbalance)

```python
import tensorflow as tf

def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1. - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.math.pow(1 - y_pred, gamma)
        return tf.reduce_mean(weight * cross_entropy)
    return loss

model.compile(optimizer='adam', loss=focal_loss())
```

---

## 🧾 4. Custom Metrics

### Example: Custom F1 Score Metric

```python
from tensorflow.keras.metrics import Metric

class F1Score(Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = Precision()
        self.recall = Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred)
        self.recall.update_state(y_true, y_pred)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + K.epsilon()))

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[F1Score()])
```

---

## 🧠 Summary Table

| Type   | Name                       | Use Case                   | Code                         |
| ------ | -------------------------- | -------------------------- | ---------------------------- |
| Loss   | `categorical_crossentropy` | Multi-class classification | `'categorical_crossentropy'` |
| Loss   | `mean_squared_error`       | Regression                 | `'mean_squared_error'`       |
| Loss   | `Huber`                    | Regression w/ outliers     | `Huber(delta=1.0)`           |
| Metric | `accuracy`                 | General performance        | `'accuracy'`                 |
| Metric | `Precision`, `Recall`      | Class imbalance            | `Precision(), Recall()`      |
| Metric | `AUC`                      | Binary classification      | `AUC()`                      |
| Custom | `custom_loss`, `F1Score`   | Special use cases          | `def ...` or `class ...`     |

---

---

## 🧠 **Example 1: MNIST + `categorical_crossentropy` + `accuracy`**

```python
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Load and prepare data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train & Evaluate
model.fit(x_train, y_train, epochs=5, validation_split=0.2)
model.evaluate(x_test, y_test)
```

---

## 🧮 **Example 2: Boston Housing + `mse` + `mae`**

```python
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError

# Load data
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

# Build model
model = Sequential([
    Dense(64, activation='relu', input_shape=(13,)),
    Dense(1)
])

# Compile
model.compile(optimizer='adam', loss=MeanSquaredError(), metrics=[MeanAbsoluteError()])

# Train & Evaluate
model.fit(x_train, y_train, epochs=30, validation_split=0.2, verbose=0)
model.evaluate(x_test, y_test)
```

---

## ⚖️ **Example 3: Binary Classifier + `huber` + `precision` + `recall`**

```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import Huber
from tensorflow.keras.metrics import Precision, Recall

# Generate binary classification data
x_train = np.random.rand(1000, 10)
y_train = (np.sum(x_train, axis=1) > 5).astype(int)

x_test = np.random.rand(200, 10)
y_test = (np.sum(x_test, axis=1) > 5).astype(int)

# Build model
model = Sequential([
    Dense(32, activation='relu', input_shape=(10,)),
    Dense(1, activation='sigmoid')
])

# Compile
model.compile(optimizer='adam', loss=Huber(), metrics=['accuracy', Precision(), Recall()])

# Train & Evaluate
model.fit(x_train, y_train, epochs=10, validation_split=0.2, verbose=1)
model.evaluate(x_test, y_test)
```

---

## 🔍 **Example 4: Custom Loss Function (MSE + L1)**

```python
import tensorflow.keras.backend as K

# Custom loss function
def mse_l1_loss(y_true, y_pred):
    mse = K.mean(K.square(y_true - y_pred))
    l1 = K.mean(K.abs(y_true - y_pred))
    return mse + 0.5 * l1

# Use Boston housing again
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

model = Sequential([
    Dense(64, activation='relu', input_shape=(13,)),
    Dense(1)
])

model.compile(optimizer='adam', loss=mse_l1_loss, metrics=['mae'])

model.fit(x_train, y_train, epochs=25, validation_split=0.2)
model.evaluate(x_test, y_test)
```

---

## 🧪 **Example 5: Custom Metric (F1 Score)**

```python
from tensorflow.keras.metrics import Metric
from tensorflow.keras.metrics import Precision, Recall

class F1Score(Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = Precision()
        self.recall = Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred)
        self.recall.update_state(y_true, y_pred)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + K.epsilon()))

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()

# Synthetic binary data
x_train = np.random.rand(1000, 10)
y_train = (np.sum(x_train, axis=1) > 5).astype(int)

x_test = np.random.rand(200, 10)
y_test = (np.sum(x_test, axis=1) > 5).astype(int)

# Model
model = Sequential([
    Dense(32, activation='relu', input_shape=(10,)),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', F1Score()])

model.fit(x_train, y_train, epochs=10, validation_split=0.2)
model.evaluate(x_test, y_test)
```

---

## ✅ Summary

| Example | Dataset          | Loss Function              | Metrics               | Extra Feature         |
| ------- | ---------------- | -------------------------- | --------------------- | --------------------- |
| 1       | MNIST            | `categorical_crossentropy` | `accuracy`            | Softmax output        |
| 2       | Boston Housing   | `mse`                      | `mae`                 | Regression            |
| 3       | Synthetic Binary | `huber`                    | `precision`, `recall` | Binary classification |
| 4       | Boston Housing   | `custom mse + l1`          | `mae`                 | Custom loss           |
| 5       | Synthetic Binary | `binary_crossentropy`      | `custom F1Score`      | Custom metric         |


