## 📚 TensorFlow Documentation: Training and Evaluation

### Compiling a Model

**Explanation:**
Compiling configures the learning process by specifying the optimizer, loss function, and evaluation metrics.

**Example:**

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
```

### Fitting a Model

**Explanation:**
Training the model using the `model.fit()` method with data and specifying training parameters such as epochs and validation splits.

**Example:**

```python
history = model.fit(x_train, y_train, epochs=10, validation_split=0.2)
```

### Evaluating a Model

**Explanation:**
Evaluates the model's performance on test data.

**Example:**

```python
loss, mae = model.evaluate(x_test, y_test)
print(f'Test loss: {loss}, Test MAE: {mae}')
```

### Predicting with a Model

**Explanation:**
Using a trained model to make predictions on new data.

**Example:**

```python
predictions = model.predict(x_new)
print(predictions)
```

### Using Callbacks

**Explanation:**
Callbacks are utilities called at certain points during training (e.g., end of an epoch) to manage the training process, including saving models, adjusting learning rates, or early stopping.

**Example:**

```python
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3),
    tf.keras.callbacks.ModelCheckpoint(filepath='best_model.h5', save_best_only=True),
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2)
]

model.fit(x_train, y_train, epochs=20, validation_split=0.2, callbacks=callbacks)
```

These examples illustrate the process of compiling, fitting, evaluating, and using callbacks effectively in TensorFlow with the Keras API.



### 20 Problems with Solutions and Explanations

### Problem 1: Compile a model for binary classification

**Solution:**

```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

**Explanation:** Sets up the model for binary classification tasks.

### Problem 2: Train a model with validation data

**Solution:**

```python
model.fit(x_train, y_train, epochs=15, validation_data=(x_val, y_val))
```

**Explanation:** Trains the model and validates on a separate dataset.

### Problem 3: Evaluate a regression model

**Solution:**

```python
loss, mae = model.evaluate(x_test, y_test)
print(f'Loss: {loss}, MAE: {mae}')
```

**Explanation:** Evaluates regression model performance on test data.

### Problem 4: Predict class probabilities

**Solution:**

```python
probabilities = model.predict(x_new)
print(probabilities)
```

**Explanation:** Predicts the class probabilities of new data samples.

### Problem 5: Implement early stopping

**Solution:**

```python
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
model.fit(x_train, y_train, epochs=50, validation_split=0.2, callbacks=[early_stop])
```

**Explanation:** Stops training if validation loss doesn't improve.

### Problem 6: Save best model during training

**Solution:**

```python
checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
model.fit(x_train, y_train, epochs=20, validation_split=0.2, callbacks=[checkpoint])
```

**Explanation:** Saves the model with the lowest validation loss.

### Problem 7: Visualize training with TensorBoard

**Solution:**

```python
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./logs')
model.fit(x_train, y_train, epochs=10, callbacks=[tensorboard])
```

**Explanation:** Records logs for visualization with TensorBoard.

### Problem 8: Adjust learning rate dynamically

**Solution:**

```python
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
model.fit(x_train, y_train, epochs=30, validation_split=0.2, callbacks=[reduce_lr])
```

**Explanation:** Reduces the learning rate when validation loss plateaus.

### Problem 9: Compile model for multi-class classification

**Solution:**

```python
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
```

**Explanation:** Configures the model for multi-class tasks.

### Problem 10: Train with batch size

**Solution:**

```python
model.fit(x_train, y_train, epochs=10, batch_size=32)
```

**Explanation:** Specifies the number of samples per gradient update.

### Problem 11: Evaluate model accuracy

**Solution:**

```python
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Accuracy: {accuracy}')
```

**Explanation:** Evaluates the classification accuracy on test data.

### Problem 12: Predict specific class labels

**Solution:**

```python
predictions = model.predict(x_new)
class_labels = predictions.argmax(axis=-1)
print(class_labels)
```

**Explanation:** Predicts class labels for new samples.

### Problem 13: Use multiple callbacks

**Solution:**

```python
callbacks = [tf.keras.callbacks.EarlyStopping(patience=2), tf.keras.callbacks.ModelCheckpoint('model.h5')]
model.fit(x_train, y_train, epochs=15, validation_split=0.3, callbacks=callbacks)
```

**Explanation:** Utilizes early stopping and checkpointing simultaneously.

### Problem 14: Compile model using mean absolute error

**Solution:**

```python
model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mae'])
```

**Explanation:** Uses MAE as loss for regression tasks.

### Problem 15: Validate model with validation\_split

**Solution:**

```python
model.fit(x_train, y_train, epochs=12, validation_split=0.25)
```

**Explanation:** Splits training data into training and validation sets automatically.

### Problem 16: Model prediction on single input

**Solution:**

```python
single_input = x_new[0:1]
prediction = model.predict(single_input)
print(prediction)
```

**Explanation:** Predicts the output for a single data sample.

### Problem 17: Set initial learning rate

**Solution:**

```python
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='mse')
```

**Explanation:** Specifies a custom learning rate for optimization.

### Problem 18: Evaluate model with custom metrics

**Solution:**

```python
model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
```

**Explanation:** Adds custom metrics to model evaluation.

### Problem 19: Predict with batch prediction

**Solution:**

```python
predictions = model.predict(x_new, batch_size=64)
print(predictions)
```

**Explanation:** Predicts outputs for new data in batches.

### Problem 20: Log detailed training progress

**Solution:**

```python
history = model.fit(x_train, y_train, epochs=10, verbose=2)
```

**Explanation:** Provides detailed logs during training to track progress.






### 8 Comprehensive Problems with Full Model, Solution, and Explanation

### Problem 1: Full Binary Classification Model

**Solution:**

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(8,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20, validation_split=0.2)
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {accuracy}')
```

**Explanation:** Builds, compiles, trains, and evaluates a binary classification model.

### Problem 2: Multi-Class Classification with Functional API

**Solution:**

```python
inputs = tf.keras.Input(shape=(10,))
x = tf.keras.layers.Dense(32, activation='relu')(inputs)
x = tf.keras.layers.Dense(16, activation='relu')(x)
outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=15, validation_split=0.2)
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {accuracy}')
```

**Explanation:** Utilizes the functional API to create a model for multi-class classification tasks.

### Problem 3: Regression Model with Early Stopping and Checkpointing

**Solution:**

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3),
    tf.keras.callbacks.ModelCheckpoint('best_regression_model.h5', save_best_only=True)
]
model.fit(x_train, y_train, epochs=30, validation_split=0.2, callbacks=callbacks)
loss, mae = model.evaluate(x_test, y_test)
print(f'Test MAE: {mae}')
```

**Explanation:** Builds a regression model, applies early stopping, and saves the best model.

### Problem 4: Model Prediction with Custom Metrics

**Solution:**

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
model.fit(x_train, y_train, epochs=20, validation_split=0.3)
predictions = model.predict(x_new)
print(predictions)
```

**Explanation:** Trains a model with custom metrics and generates predictions for new data.

### Problem 5: Training Visualization with TensorBoard

**Solution:**

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(20,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs')
model.fit(x_train, y_train, epochs=10, validation_split=0.2, callbacks=[tensorboard_callback])
```

**Explanation:** Trains a multi-class model and logs training details for visualization in TensorBoard.

### Problem 6: Convolutional Neural Network (CNN) for Image Classification

**Solution:**

```python
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, validation_split=0.2)
```

**Explanation:** Implements a CNN for image classification tasks.

### Problem 7: LSTM Model for Sequence Prediction

**Solution:**

```python
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, activation='relu', input_shape=(10, 1)),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(x_train, y_train, epochs=20, validation_split=0.2)
```

**Explanation:** Utilizes LSTM layers for predicting sequences.

### Problem 8: Autoencoder for Dimensionality Reduction

**Solution:**

```python
input_layer = tf.keras.layers.Input(shape=(784,))
encoded = tf.keras.layers.Dense(32, activation='relu')(input_layer)
decoded = tf.keras.layers.Dense(784, activation='sigmoid')(encoded)
autoencoder = tf.keras.Model(input_layer, decoded)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train, epochs=20, validation_split=0.2)
```

**Explanation:** Constructs an autoencoder model to perform dimensionality reduction.
