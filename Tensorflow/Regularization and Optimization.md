
---

## 🔒 1. Dropout

**Dropout** is a regularization method where a fraction of neurons is randomly "dropped" (set to 0) during training to prevent overfitting.

### Code Example

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dropout(0.5),  # 50% of the neurons are dropped
    Dense(64, activation='relu'),
    Dropout(0.3),  # 30% dropped
    Dense(10, activation='softmax')
])
```

---

## 🧰 2. L1 and L2 Regularization

These add penalties to the loss function to discourage complex models:

* **L1** (Lasso): adds absolute value of weights.
* **L2** (Ridge): adds squared weights.

### Code Example

```python
from tensorflow.keras import regularizers

model = Sequential([
    Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=(784,)),
    Dense(64, activation='relu', kernel_regularizer=regularizers.l1(0.01)),
    Dense(10, activation='softmax')
])
```

---

## ⛔ 3. Early Stopping

**EarlyStopping** stops training once the validation loss stops improving, avoiding overfitting.

### Code Example

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model.fit(x_train, y_train, validation_split=0.2, epochs=50, callbacks=[early_stop])
```

---

## ⚖️ 4. Batch Normalization

**Batch Normalization** normalizes inputs across the batch, speeding up training and stabilizing learning.

### Code Example

```python
from tensorflow.keras.layers import BatchNormalization

model = Sequential([
    Dense(128, input_shape=(784,)),
    BatchNormalization(),
    Activation('relu'),
    Dense(64),
    BatchNormalization(),
    Activation('relu'),
    Dense(10, activation='softmax')
])
```

---

## 📉 5. Learning Rate Scheduling

**Learning Rate Schedulers** adjust the learning rate during training to improve convergence.

### Code Example

```python
from tensorflow.keras.callbacks import LearningRateScheduler

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * 0.95  # reduce lr by 5% after 10 epochs

lr_schedule = LearningRateScheduler(scheduler)

model.fit(x_train, y_train, epochs=30, callbacks=[lr_schedule])
```

Or use built-in:

```python
from tensorflow.keras.optimizers.schedules import ExponentialDecay

lr_schedule = ExponentialDecay(initial_learning_rate=0.01,
                               decay_steps=10000,
                               decay_rate=0.9)
```

---

## ⚙️ 6. Custom Optimizers

Custom optimizers allow for modifying how gradients are applied. You can use built-in ones like Adam, RMSprop, or create your own.

### Example using a built-in custom optimizer:

```python
from tensorflow.keras.optimizers import Adam

optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
```

### Writing a custom optimizer (advanced):

```python
import tensorflow as tf

class MySGD(tf.keras.optimizers.Optimizer):
    def __init__(self, learning_rate=0.01, name="MySGD", **kwargs):
        super().__init__(name, **kwargs)
        self._learning_rate = learning_rate

    def _resource_apply_dense(self, grad, var):
        var.assign_sub(self._learning_rate * grad)

    def get_config(self):
        return {"learning_rate": self._learning_rate}

model.compile(optimizer=MySGD(learning_rate=0.01), loss='mse')
```

---

## ✅ Summary Table

| Technique                | Purpose                                       | Code Element                            |
| ------------------------ | --------------------------------------------- | --------------------------------------- |
| Dropout                  | Prevent overfitting by random deactivation    | `Dropout(rate)`                         |
| L1/L2 Regularization     | Penalize large weights                        | `kernel_regularizer=regularizers.l1/l2` |
| Early Stopping           | Stop training when validation stops improving | `EarlyStopping()`                       |
| Batch Normalization      | Normalize layer inputs                        | `BatchNormalization()`                  |
| Learning Rate Scheduling | Dynamically adjust learning rate              | `LearningRateScheduler()`               |
| Custom Optimizers        | Custom gradient update logic                  | `Custom class + model.compile()`        |

---
Here are **5 complete neural network models** using **TensorFlow/Keras** that showcase combinations of **regularization and optimization techniques** like Dropout, L1/L2 regularization, Batch Normalization, Early Stopping, Learning Rate Scheduling, and Custom Optimizers.

---

## 🧠 **Model 1: Dropout + Adam Optimizer (MNIST Classifier)**

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, validation_split=0.2)
```

---

## 🧰 **Model 2: L1/L2 Regularization + Batch Normalization**

```python
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    BatchNormalization(),
    Dense(64, activation='relu', kernel_regularizer=regularizers.l1(0.005)),
    BatchNormalization(),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=15, validation_split=0.2)
```

---

## ⛔ **Model 3: Early Stopping + Learning Rate Scheduler**

```python
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

# Learning rate scheduler function
def lr_schedule(epoch, lr):
    return lr * 0.95 if epoch > 5 else lr

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=30, validation_split=0.2,
          callbacks=[early_stop, LearningRateScheduler(lr_schedule)])
```

---

## ⚙️ **Model 4: Custom Optimizer + Batch Normalization**

```python
import tensorflow as tf

# Custom SGD optimizer
class CustomSGD(tf.keras.optimizers.Optimizer):
    def __init__(self, learning_rate=0.01, name="CustomSGD", **kwargs):
        super().__init__(name, **kwargs)
        self._learning_rate = learning_rate

    def _resource_apply_dense(self, grad, var):
        var.assign_sub(self._learning_rate * grad)

    def get_config(self):
        return {"learning_rate": self._learning_rate}

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128),
    BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    Dense(64),
    BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer=CustomSGD(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=15, validation_split=0.2)
```

---

## 🧪 **Model 5: All Techniques Combined (Dropout, L2, BatchNorm, Early Stopping, LR Schedule)**

```python
from tensorflow.keras.optimizers.schedules import ExponentialDecay

# Learning rate decay schedule
lr_schedule = ExponentialDecay(initial_learning_rate=0.01,
                               decay_steps=1000,
                               decay_rate=0.9)

early_stop = EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True)

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(256, kernel_regularizer=regularizers.l2(0.01)),
    BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    Dropout(0.5),
    Dense(128, kernel_regularizer=regularizers.l2(0.01)),
    BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=30, validation_split=0.2, callbacks=[early_stop])
```

---

## ✅ Summary

| Model | Techniques Used                                 |
| ----- | ----------------------------------------------- |
| 1     | Dropout, Adam Optimizer                         |
| 2     | L1/L2 Regularization, Batch Normalization       |
| 3     | Early Stopping, Learning Rate Scheduler         |
| 4     | Custom Optimizer, Batch Normalization           |
| 5     | Dropout, L2, BatchNorm, EarlyStopping, LR Decay |


