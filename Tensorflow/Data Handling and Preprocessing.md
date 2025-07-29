## 📚 TensorFlow Documentation: Data Handling and Preprocessing

### tf.data.Dataset API

**Explanation:**
The `tf.data.Dataset` API efficiently manages input pipelines by loading, batching, shuffling, and preprocessing data.

**Example:**

```python
import tensorflow as tf

dataset = tf.data.Dataset.from_tensor_slices((features, labels))
dataset = dataset.shuffle(buffer_size=1024).batch(32).prefetch(tf.data.AUTOTUNE)
```

### Working with CSV

**Explanation:**
Load data from CSV files directly into TensorFlow.

**Example:**

```python
dataset = tf.data.experimental.make_csv_dataset(
    'data.csv', batch_size=32, label_name='label', num_epochs=1, shuffle=True)
```

### Working with Images

**Explanation:**
Efficiently load and preprocess image datasets.

**Example:**

```python
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    'path/to/images', batch_size=32, image_size=(224, 224))
```

### Working with Text

**Explanation:**
Manage text data loading and preprocessing.

**Example:**

```python
texts = ['example text', 'another example']
dataset = tf.data.Dataset.from_tensor_slices(texts)
```

### Working with TFRecords

**Explanation:**
Store and read data efficiently with TFRecord format.

**Example:**

```python
def parse_example(serialized_example):
    features = tf.io.parse_single_example(serialized_example, feature_description)
    return features

raw_dataset = tf.data.TFRecordDataset('data.tfrecord')
parsed_dataset = raw_dataset.map(parse_example)
```

### Data Augmentation (tf.image)

**Explanation:**
Increase dataset variability using data augmentation techniques.

**Example:**

```python
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
])
```

### Preprocessing Layers

**Normalization:**

```python
normalization_layer = tf.keras.layers.Normalization()
normalization_layer.adapt(data)
```

**Rescaling:**

```python
rescaling_layer = tf.keras.layers.Rescaling(1./255)
```

**TextVectorization:**

```python
text_vectorizer = tf.keras.layers.TextVectorization(max_tokens=1000)
text_vectorizer.adapt(text_dataset)
```

### Feature Columns

**Explanation:**
Use feature columns for structured data input.

**Example:**

```python
feature_columns = [
    tf.feature_column.numeric_column('numeric_feature'),
    tf.feature_column.categorical_column_with_vocabulary_list('categorical_feature', ['A', 'B', 'C'])
]
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
```

These examples illustrate essential data handling and preprocessing techniques using TensorFlow effectively.


### Model 1: Simple Sequential Model (Binary Classification)

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
```

### Model 2: Multi-Class Classification with Functional API

```python
inputs = tf.keras.Input(shape=(10,))
x = tf.keras.layers.Dense(64, activation='relu')(inputs)
x = tf.keras.layers.Dense(32, activation='relu')(x)
outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
```

### Model 3: Convolutional Neural Network (CNN)

```python
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
```

### Model 4: LSTM for Sequence Prediction

```python
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, activation='relu', input_shape=(10, 1)),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.summary()
```

### Model 5: Autoencoder

```python
input_layer = tf.keras.layers.Input(shape=(784,))
encoded = tf.keras.layers.Dense(64, activation='relu')(input_layer)
decoded = tf.keras.layers.Dense(784, activation='sigmoid')(encoded)
autoencoder = tf.keras.Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.summary()
```

### Model 6: GAN (Generative Adversarial Network)

```python
generator = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(100,)),
    tf.keras.layers.Dense(784, activation='sigmoid')
])
discriminator = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile models separately
```

### Model 7: Transformer Model for NLP

```python
encoder_input = tf.keras.Input(shape=(None,), dtype='int64')
embedding_layer = tf.keras.layers.Embedding(input_dim=1000, output_dim=64)(encoder_input)
transformer_block = tf.keras.layers.TransformerEncoder(num_heads=2, intermediate_dim=128)(embedding_layer)
output = tf.keras.layers.Dense(1, activation='sigmoid')(transformer_block[:, 0, :])
model = tf.keras.Model(encoder_input, output)
model.compile(optimizer='adam', loss='binary_crossentropy')
model.summary()
```

### Model 8: Reinforcement Learning Model

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(24, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(2, activation='linear')
])
model.compile(optimizer='adam', loss='mse')
model.summary()
```

### Model 9: Time Series Prediction (RNN)

```python
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(32, activation='relu', input_shape=(None, 1)),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.summary()
```

### Model 10: Text Classification with Embedding

```python
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=5000, output_dim=64, input_length=100),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
```



## 📚 TensorFlow Documentation: 10 Comprehensive Full Model Examples with Data Handling and Preprocessing

### Model 1: Simple Sequential Model (Binary Classification)

```python
import tensorflow as tf

dataset = tf.data.Dataset.from_tensor_slices((features, labels)).batch(32)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(dataset, epochs=10)
```

### Model 2: Multi-Class Classification with Functional API and CSV data

```python
dataset = tf.data.experimental.make_csv_dataset(
    'data.csv', batch_size=32, label_name='label', num_epochs=1, shuffle=True)

inputs = tf.keras.Input(shape=(10,))
x = tf.keras.layers.Dense(64, activation='relu')(inputs)
x = tf.keras.layers.Dense(32, activation='relu')(x)
outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(dataset, epochs=10)
```

### Model 3: CNN with Image Data Augmentation

```python
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'path/to/images', batch_size=32, image_size=(64, 64))

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2)
])

model = tf.keras.Sequential([
    data_augmentation,
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_ds, epochs=10)
```

### Model 4: LSTM for Sequence Prediction with TFRecords

```python
def parse_example(serialized_example):
    features = tf.io.parse_single_example(serialized_example, feature_description)
    return features['sequence'], features['label']

dataset = tf.data.TFRecordDataset('data.tfrecord').map(parse_example).batch(32)

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(dataset, epochs=10)
```

### Model 5: Autoencoder with Normalization Layer

```python
dataset = tf.data.Dataset.from_tensor_slices(features).batch(32)
normalization_layer = tf.keras.layers.Normalization()
normalization_layer.adapt(features)

input_layer = tf.keras.layers.Input(shape=(784,))
x = normalization_layer(input_layer)
encoded = tf.keras.layers.Dense(64, activation='relu')(x)
decoded = tf.keras.layers.Dense(784, activation='sigmoid')(encoded)
autoencoder = tf.keras.Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(dataset, epochs=10)
```

### Model 6: GAN with structured data

```python
feature_columns = [tf.feature_column.numeric_column('features', shape=(100,))]
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

# Define generator and discriminator
```

### Model 7: Transformer NLP with TextVectorization

```python
text_vectorizer = tf.keras.layers.TextVectorization(max_tokens=1000)
text_vectorizer.adapt(text_dataset)

inputs = tf.keras.Input(shape=(None,), dtype=tf.string)
x = text_vectorizer(inputs)
x = tf.keras.layers.Embedding(1000, 64)(x)
x = tf.keras.layers.GlobalAveragePooling1D()(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs, outputs)
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(text_dataset.batch(32), epochs=10)
```

### Model 8: Reinforcement Learning Model

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(24, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(2, activation='linear')
])
model.compile(optimizer='adam', loss='mse')
```

### Model 9: Time Series Prediction with RNN and Prefetch

```python
dataset = tf.data.Dataset.from_tensor_slices((features, labels)).batch(32).prefetch(tf.data.AUTOTUNE)
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(dataset, epochs=10)
```

### Model 10: Text Classification with Embedding and Rescaling

```python
train_ds = tf.keras.preprocessing.text_dataset_from_directory('path/to/texts', batch_size=32)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=5000, output_dim=64),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_ds, epochs=10)
```



### Model 1: Simple Sequential Model (Binary Classification)

```python
import tensorflow as tf

features, labels = tf.random.normal([100, 10]), tf.random.uniform([100, 1], maxval=2, dtype=tf.int32)
dataset = tf.data.Dataset.from_tensor_slices((features, labels)).batch(32)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(dataset, epochs=10)
model.save('model1.h5')
```

### Model 2: Multi-Class Classification with CSV Data

```python
dataset = tf.data.experimental.make_csv_dataset(
    'data.csv', batch_size=32, label_name='label', num_epochs=1, shuffle=True)

inputs = tf.keras.Input(shape=(10,))
x = tf.keras.layers.Dense(64, activation='relu')(inputs)
x = tf.keras.layers.Dense(32, activation='relu')(x)
outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(dataset, epochs=10)
model.save('model2.h5')
```

### Model 3: CNN with Image Data Augmentation

```python
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'path/to/images', batch_size=32, image_size=(64, 64))

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2)
])

model = tf.keras.Sequential([
    data_augmentation,
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_ds, epochs=10)
model.save('model3.h5')
```

### Model 4: LSTM for Sequence Prediction with TFRecords

```python
feature_description = {'sequence': tf.io.FixedLenFeature([10], tf.float32), 'label': tf.io.FixedLenFeature([], tf.float32)}

def parse_example(serialized_example):
    features = tf.io.parse_single_example(serialized_example, feature_description)
    return features['sequence'], features['label']

dataset = tf.data.TFRecordDataset('data.tfrecord').map(parse_example).batch(32)

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(dataset, epochs=10)
model.save('model4.h5')
```

### Model 5: Autoencoder with Normalization Layer

```python
features = tf.random.normal([100, 784])
dataset = tf.data.Dataset.from_tensor_slices(features).batch(32)
normalization_layer = tf.keras.layers.Normalization()
normalization_layer.adapt(features)

input_layer = tf.keras.layers.Input(shape=(784,))
x = normalization_layer(input_layer)
encoded = tf.keras.layers.Dense(64, activation='relu')(x)
decoded = tf.keras.layers.Dense(784, activation='sigmoid')(encoded)
autoencoder = tf.keras.Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(dataset, epochs=10)
autoencoder.save('model5.h5')
```

### Model 6: Transformer NLP with TextVectorization

```python
text_dataset = tf.data.Dataset.from_tensor_slices((['text example']*100, [0]*100)).batch(32)
text_vectorizer = tf.keras.layers.TextVectorization(max_tokens=1000)
text_vectorizer.adapt(text_dataset.map(lambda x, y: x))

inputs = tf.keras.Input(shape=(1,), dtype=tf.string)
x = text_vectorizer(inputs)
x = tf.keras.layers.Embedding(1000, 64)(x)
x = tf.keras.layers.GlobalAveragePooling1D()(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs, outputs)
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(text_dataset, epochs=10)
model.save('model6.h5')
```

### Model 7: Reinforcement Learning Model

```python
features, labels = tf.random.normal([100, 4]), tf.random.normal([100, 2])
dataset = tf.data.Dataset.from_tensor_slices((features, labels)).batch(32)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(24, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(2, activation='linear')
])
model.compile(optimizer='adam', loss='mse')
model.fit(dataset, epochs=10)
model.save('model7.h5')
```

### Model 8: Time Series Prediction with RNN

```python
features, labels = tf.random.normal([100, 10, 1]), tf.random.normal([100, 1])
dataset = tf.data.Dataset.from_tensor_slices((features, labels)).batch(32)

model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(dataset, epochs=10)
model.save('model8.h5')
```


### Model 1: Simple Sequential Model (Binary Classification)

```python
import tensorflow as tf

features, labels = tf.random.normal([100, 10]), tf.random.uniform([100, 1], maxval=2, dtype=tf.int32)
dataset = tf.data.Dataset.from_tensor_slices((features, labels)).batch(32)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(dataset, epochs=10)
loss, accuracy = model.evaluate(dataset)
print(f'Test Accuracy: {accuracy}')
```

### Model 2: Multi-Class Classification with CSV Data

```python
dataset = tf.data.experimental.make_csv_dataset(
    'data.csv', batch_size=32, label_name='label', num_epochs=1, shuffle=True)

inputs = tf.keras.Input(shape=(10,))
x = tf.keras.layers.Dense(64, activation='relu')(inputs)
x = tf.keras.layers.Dense(32, activation='relu')(x)
outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(dataset, epochs=10)
loss, accuracy = model.evaluate(dataset)
print(f'Test Accuracy: {accuracy}')
```

### Model 3: CNN with Image Data Augmentation

```python
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'path/to/images', batch_size=32, image_size=(64, 64))

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2)
])

model = tf.keras.Sequential([
    data_augmentation,
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_ds, epochs=10)
loss, accuracy = model.evaluate(train_ds)
print(f'Test Accuracy: {accuracy}')
```

### Model 4: LSTM for Sequence Prediction with TFRecords

```python
feature_description = {'sequence': tf.io.FixedLenFeature([10], tf.float32), 'label': tf.io.FixedLenFeature([], tf.float32)}

def parse_example(serialized_example):
    features = tf.io.parse_single_example(serialized_example, feature_description)
    return features['sequence'], features['label']

dataset = tf.data.TFRecordDataset('data.tfrecord').map(parse_example).batch(32)

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(dataset, epochs=10)
loss, mae = model.evaluate(dataset)
print(f'Test MAE: {mae}')
```

### Model 5: Autoencoder with Normalization Layer

```python
features = tf.random.normal([100, 784])
dataset = tf.data.Dataset.from_tensor_slices(features).batch(32)
normalization_layer = tf.keras.layers.Normalization()
normalization_layer.adapt(features)

input_layer = tf.keras.layers.Input(shape=(784,))
x = normalization_layer(input_layer)
encoded = tf.keras.layers.Dense(64, activation='relu')(x)
decoded = tf.keras.layers.Dense(784, activation='sigmoid')(encoded)
autoencoder = tf.keras.Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(dataset, epochs=10)
loss = autoencoder.evaluate(dataset)
print(f'Test Loss: {loss}')
```


