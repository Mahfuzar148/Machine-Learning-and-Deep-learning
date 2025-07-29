## 📚 TensorFlow Documentation: TensorBoard and Visualization

### Logging Metrics, Weights, and Images

**Explanation:**
TensorBoard provides visualization of metrics, weights, and images during training.

**Example:**

```python
import tensorflow as tf

log_dir = "logs/fit/"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, callbacks=[tensorboard_callback])
```

### Visualizing Training Curves

**Explanation:**
Training curves illustrate model performance across epochs, showing metrics like accuracy and loss.

**Example:**

```python
# Run TensorBoard in terminal
%tensorboard --logdir logs/fit
```

### Embedding Projector

**Explanation:**
The Embedding Projector in TensorBoard visualizes high-dimensional embedding vectors.

**Example:**

```python
# Save embeddings for visualization
embedding_var = tf.Variable(embeddings, name='word_embeddings')
checkpoint = tf.train.Checkpoint(embedding=embedding_var)
checkpoint.save('logs/embed/embedding.ckpt')

config = tf.compat.v1.ConfigProto()
sess = tf.compat.v1.Session(config=config)
saver = tf.compat.v1.train.Saver([embedding_var])
saver.save(sess, 'logs/embed/embedding.ckpt')
```

### Custom TensorBoard Summaries

**Explanation:**
Create custom summaries for specific data visualization requirements.

**Example:**

```python
summary_writer = tf.summary.create_file_writer('logs/custom')

with summary_writer.as_default():
    tf.summary.scalar('loss', 0.25, step=1)
    tf.summary.image('sample_image', images, step=1)
```

These detailed examples enable comprehensive TensorBoard usage, enhancing training visualization and analysis.



## 📚 TensorFlow Documentation: 5 Full Model Examples with TensorBoard Integration

### Model 1: Binary Classification with TensorBoard

```python
import tensorflow as tf

(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784) / 255.0
y_train = (y_train % 2 == 0).astype(int)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs/model1")
model.fit(x_train, y_train, epochs=5, callbacks=[tensorboard_callback])
```

### Model 2: Multi-Class Classification with TensorBoard

```python
(x_train, y_train), _ = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.reshape(-1, 784) / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs/model2")
model.fit(x_train, y_train, epochs=5, callbacks=[tensorboard_callback])
```

### Model 3: CNN for Image Classification with TensorBoard

```python
(x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
x_train = x_train / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs/model3")
model.fit(x_train, y_train, epochs=5, callbacks=[tensorboard_callback])
```

### Model 4: LSTM for Text Classification with TensorBoard

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = ["TensorFlow is great", "Keras makes it easy", "Machine learning is fascinating"]
labels = [1, 1, 0]

tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences)
x_train = pad_sequences(tokenizer.texts_to_sequences(sentences), padding='post')

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=100, output_dim=64),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs/model4")
model.fit(x_train, labels, epochs=5, callbacks=[tensorboard_callback])
```

### Model 5: Autoencoder for Dimensionality Reduction with TensorBoard

```python
(x_train, _), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784) / 255.0

input_layer = tf.keras.layers.Input(shape=(784,))
encoded = tf.keras.layers.Dense(64, activation='relu')(input_layer)
decoded = tf.keras.layers.Dense(784, activation='sigmoid')(encoded)
autoencoder = tf.keras.Model(input_layer, decoded)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs/model5")
autoencoder.fit(x_train, x_train, epochs=5, callbacks=[tensorboard_callback])
```

These complete examples illustrate the integration of TensorBoard for comprehensive model monitoring and analysis.
