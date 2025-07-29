
---

## 🧠 Advanced Models

Below are five advanced modeling techniques in TensorFlow, along with **use cases (কাজ), reasons (ক্যানো), and recommended scenarios (কখন)** for adopting them.

---

### 1. Custom Layers with Multiple Inputs/Outputs

**কাজ (Use-case):**

* When your model needs to process two or more inputs together and produce multiple related outputs in one pass. E.g., computing both sum and difference, or joint image-text features.

**ক্যানো (Why):**

* Encapsulates complex logic in a reusable component.
* Keeps model graph clear by isolating multi-input/output operations.

**কখন (When to use):**

* Building multi-modal networks (e.g., image + text).
* Creating specialized transformations that return several tensors.

**Example:**

```python
import tensorflow as tf

class MultiIO(tf.keras.layers.Layer):
    def call(self, inputs):
        x, y = inputs
        return {"sum": x + y, "diff": x - y}

in1 = tf.keras.Input(shape=(4,))
in2 = tf.keras.Input(shape=(4,))
iodata = MultiIO()([in1, in2])
model = tf.keras.Model(inputs=[in1, in2], outputs=[iodata['sum'], iodata['diff']])
model.summary()
```

---

### 2. Shared Layers

**কাজ (Use-case):**

* Applying the same transformation to different inputs, e.g., parallel feature extractors for different data streams.

**ক্যানো (Why):**

* Reduces number of parameters by weight sharing, improving generalization.
* Ensures consistent feature extraction across branches.

**কখন (When to use):**

* Siamese networks for similarity learning.
* Multi-branch models with identical sub-networks.

**Example:**

```python
shared = tf.keras.layers.Dense(8, activation='relu')

iA = tf.keras.Input(shape=(16,))
iB = tf.keras.Input(shape=(16,))
outA = shared(iA)
outB = shared(iB)
merged = tf.keras.layers.concatenate([outA, outB])
output = tf.keras.layers.Dense(1)(merged)
model = tf.keras.Model(inputs=[iA, iB], outputs=output)
model.summary()
```

---

### 3. Multiple Loss Functions

**কাজ (Use-case):**

* Training models with more than one objective, e.g., regression + classification simultaneously.

**ক্যানো (Why):**

* Allows balancing different learning goals in a unified model.
* Optimizes shared layers for multiple tasks, improving multi-task performance.

**কখন (When to use):**

* Multi-task learning scenarios (e.g., predict age and gender from image).
* Models that require auxiliary outputs to guide training.

**Example:**

```python
inp = tf.keras.Input(shape=(4,))
h = tf.keras.layers.Dense(3, activation='relu')(inp)
out_reg = tf.keras.layers.Dense(1, name='regression')(h)
out_clf = tf.keras.layers.Dense(5, activation='softmax', name='classification')(h)

model = tf.keras.Model(inputs=inp, outputs=[out_reg, out_clf])
model.compile(
    optimizer='adam',
    loss={'regression':'mse', 'classification':'categorical_crossentropy'},
    metrics={'regression':'mae', 'classification':'accuracy'}
)
model.summary()
```

---

### 4. Model Ensembling

**কাজ (Use-case):**

* Combining predictions from multiple submodels to reduce variance and improve robustness.

**ক্যানো (Why):**

* Captures diverse modeling perspectives, leading to stronger overall performance.
* Mitigates overfitting of individual models.

**কখন (When to use):**

* Competitive settings where marginal gains matter.
* Final stage of pipeline to combine different architectures.

**Example:**

```python
inp = tf.keras.Input(shape=(10,))
b1 = tf.keras.layers.Dense(32, activation='relu')(inp)
out1 = tf.keras.layers.Dense(1)(b1)
b2 = tf.keras.layers.Dense(16, activation='relu')(inp)
out2 = tf.keras.layers.Dense(1)(b2)
avg = tf.keras.layers.average([out1, out2])
model = tf.keras.Model(inputs=inp, outputs=avg)
model.compile(optimizer='adam', loss='mse')
model.summary()
```

---

### 5. Attention Mechanism & Transformer Layers

**কাজ (Use-case):**

* Learning dependencies between sequence elements, e.g., in NLP, time series, or graph data.

**ক্যানো (Why):**

* Dynamically focuses on relevant parts of input.
* Scales better than recurrence for long-range dependencies.

**কখন (When to use):**

* Sequence-to-sequence tasks (translation, summarization).
* Any problem requiring contextual weighting of inputs.

**Example:**

```python
import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention, Embedding, GlobalAveragePooling1D

inputs = tf.keras.Input(shape=(None,), dtype='int64')
x = Embedding(input_dim=10000, output_dim=64)(inputs)
attn = MultiHeadAttention(num_heads=2, key_dim=64)(x, x)
x = GlobalAveragePooling1D()(attn)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
```

---
