
---

## 🧠 1. Convolutional Layers

Convolutional layers extract features like edges, textures, shapes from images.

### Key Layers:

* `Conv2D`: Applies convolution operation.
* `MaxPooling2D`: Downsamples feature maps.

### 📦 Code Example

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')  # 10 classes
])
```

---

## 🏷️ 2. Image Classification

Classifies an image into one of several categories.

### Example: CIFAR-10 Image Classifier

```python
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, validation_split=0.2)
```

---

## 🧩 3. Image Segmentation

Assigns a class label to each pixel (semantic segmentation).

### Example: U-Net Architecture (Simplified)

```python
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model

def simple_unet(input_shape):
    inputs = Input(input_shape)
    
    c1 = Conv2D(16, 3, activation='relu', padding='same')(inputs)
    p1 = MaxPooling2D()(c1)
    
    c2 = Conv2D(32, 3, activation='relu', padding='same')(p1)
    p2 = MaxPooling2D()(c2)
    
    c3 = Conv2D(64, 3, activation='relu', padding='same')(p2)
    
    u1 = UpSampling2D()(c3)
    merge1 = concatenate([u1, c2])
    c4 = Conv2D(32, 3, activation='relu', padding='same')(merge1)
    
    u2 = UpSampling2D()(c4)
    merge2 = concatenate([u2, c1])
    c5 = Conv2D(16, 3, activation='relu', padding='same')(merge2)
    
    outputs = Conv2D(1, 1, activation='sigmoid')(c5)
    
    return Model(inputs, outputs)

model = simple_unet((128, 128, 3))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

---

## 🧿 4. Object Detection (Overview)

Object detection locates and classifies **multiple objects** in an image. TensorFlow provides models via the [TF Hub Object Detection API](https://tfhub.dev/s?module-type=image-object-detection).

### Example: TF Hub Pretrained Model

```python
import tensorflow_hub as hub
import tensorflow as tf

model = hub.load("https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1")

# Preprocess input image
image = tf.io.read_file("image.jpg")
image = tf.image.decode_jpeg(image, channels=3)
image = tf.image.resize(image, (640, 640))
image = tf.expand_dims(image, 0) / 255.0

# Run detection
result = model(image)
print(result["detection_boxes"], result["detection_class_entities"])
```

> 📝 For production, use [TF Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection).

---

## 🔁 5. Transfer Learning with Pretrained Models

### Examples: MobileNet, ResNet, VGG16

```python
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze base

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu')(x)
output = Dense(5, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

> ✅ Transfer learning enables training with fewer samples & faster convergence.

---

## 🧪 6. Data Augmentation for Vision Tasks

### Using `tf.keras.layers.Random*` layers:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom

data_augmentation = Sequential([
    RandomFlip('horizontal'),
    RandomRotation(0.2),
    RandomZoom(0.1)
])
```

Apply in your model:

```python
model = Sequential([
    data_augmentation,
    Conv2D(32, 3, activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(),
    Flatten(),
    Dense(10, activation='softmax')
])
```

---

## 📂 7. Image Loading Pipelines with `image_dataset_from_directory`

Loads images from directories structured as:

```
/dataset/
  /cats/
  /dogs/
```

### Code:

```python
from tensorflow.keras.utils import image_dataset_from_directory

train_ds = image_dataset_from_directory(
    "dataset/",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(128, 128),
    batch_size=32
)

val_ds = image_dataset_from_directory(
    "dataset/",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(128, 128),
    batch_size=32
)
```

### With Prefetch:

```python
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
```

---

## ✅ Summary Table

| Feature              | Method / Layer                       | Purpose                           |
| -------------------- | ------------------------------------ | --------------------------------- |
| Convolution          | `Conv2D`, `MaxPooling2D`             | Extract features                  |
| Classification       | `Dense` + `softmax`                  | Classify images                   |
| Segmentation         | U-Net, `Conv2D`, `UpSampling2D`      | Pixel-wise classification         |
| Object Detection     | TF Hub / TF Object Detection API     | Detect multiple objects per image |
| Transfer Learning    | `MobileNetV2`, `ResNet50`, etc.      | Leverage pretrained models        |
| Augmentation         | `RandomFlip`, `RandomRotation`, etc. | Improve generalization            |
| Image Input Pipeline | `image_dataset_from_directory()`     | Load structured image data        |

---

Below is the **full code** for both models including:

* Model building
* Training
* Prediction with a test input
* Visualization of **input image**, **actual label**, and **predicted label**

---

## ✅ **MODEL 1: Basic CNN — MNIST**

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
y_train = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)  # keep for evaluation

# Build model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, validation_split=0.1)

# Evaluate on test set
loss, acc = model.evaluate(x_test, y_test_cat)
print("Test Accuracy:", acc)

# Predict one sample
sample = x_test[0].reshape(1, 28, 28, 1)
prediction = model.predict(sample)
predicted_class = np.argmax(prediction)

# Show prediction
plt.imshow(x_test[0].reshape(28, 28), cmap='gray')
plt.title(f"Predicted: {predicted_class} | Actual: {y_test[0]}")
plt.axis('off')
plt.show()
```

---

## ✅ **MODEL 2: CNN + Data Augmentation — CIFAR-10**

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Data augmentation
datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1,
                             height_shift_range=0.1, horizontal_flip=True)
datagen.fit(x_train)

# Build model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(datagen.flow(x_train, y_train_cat, batch_size=64), epochs=10,
          validation_data=(x_test, y_test_cat))

# Evaluate
loss, acc = model.evaluate(x_test, y_test_cat)
print("Test Accuracy:", acc)

# Predict one sample
idx = 10
sample = x_test[idx].reshape(1, 32, 32, 3)
prediction = model.predict(sample)
predicted_class = np.argmax(prediction)

# Show prediction
plt.imshow(x_test[idx])
plt.title(f"Predicted: {class_names[predicted_class]} | Actual: {class_names[int(y_test[idx])]}")
plt.axis('off')
plt.show()
```

---

These scripts do **everything**:

* Train models
* Evaluate accuracy
* Take an input image
* Predict class
* Show image with predicted and actual label


Here's the full implementation of **Model 3: Transfer Learning with MobileNetV2**, which:

* Uses a pretrained MobileNetV2 model
* Accepts an image file path from keyboard input
* Preprocesses and predicts the class
* Displays the input image and predicted label

---

## ✅ **Model 3: Transfer Learning — MobileNetV2**

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import os

# Load pretrained MobileNetV2
model = MobileNetV2(weights='imagenet')

# Ask for input path from user
img_path = input("Enter path to an image (224x224 or larger): ")

# Check if the path is valid
if not os.path.exists(img_path):
    print("Invalid image path. Exiting.")
else:
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_array_expanded)

    # Predict
    preds = model.predict(img_preprocessed)
    decoded = decode_predictions(preds, top=1)[0][0]

    # Display
    plt.imshow(img)
    plt.title(f"Predicted: {decoded[1]} ({decoded[2]*100:.2f}%)")
    plt.axis('off')
    plt.show()
```

---

### 📥 **How to Use**

1. Run the code in your Python environment (e.g., Jupyter, VS Code).
2. When prompted:
   ➤ Enter an image path like:

   ```
   C:/Users/yourname/Pictures/cat.jpg
   ```
3. The model will display:

   * The **image**
   * The **predicted class label** and **confidence**

---

Here is the full implementation of **Model 4: Transfer Learning with ResNet50**, which:

* Loads the pre-trained **ResNet50** model
* Accepts image path via **keyboard input**
* Preprocesses the image
* Predicts and displays the result with confidence score

---

## ✅ **Model 4: Transfer Learning — ResNet50**

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import os

# Load ResNet50 model with ImageNet weights
model = ResNet50(weights='imagenet')

# Ask user for image path
img_path = input("Enter path to an image (224x224 or larger): ")

if not os.path.exists(img_path):
    print("Invalid image path.")
else:
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)

    # Predict
    preds = model.predict(img_preprocessed)
    decoded = decode_predictions(preds, top=1)[0][0]

    # Display image and prediction
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Predicted: {decoded[1]} ({decoded[2]*100:.2f}%)")
    plt.show()
```

---

### 📌 Output Example

If you input an image of a **Labrador retriever**, you’ll see:

```
Predicted: Labrador_retriever (94.56%)
```

And the image displayed.

---

Here is the full implementation of **Model 5: U-Net for Semantic Segmentation**, which:

* Defines a simplified **U-Net** architecture
* Loads a grayscale image and its corresponding mask from file paths entered via **keyboard input**
* Predicts a segmentation mask
* Displays the original image, ground truth mask, and predicted mask

---

## ✅ **Model 5: U-Net for Semantic Segmentation**

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

# Build U-Net model
def build_unet(input_shape):
    inputs = Input(input_shape)

    # Encoder
    c1 = Conv2D(16, 3, activation='relu', padding='same')(inputs)
    p1 = MaxPooling2D()(c1)

    c2 = Conv2D(32, 3, activation='relu', padding='same')(p1)
    p2 = MaxPooling2D()(c2)

    # Bottleneck
    c3 = Conv2D(64, 3, activation='relu', padding='same')(p2)

    # Decoder
    u1 = UpSampling2D()(c3)
    concat1 = concatenate([u1, c2])
    c4 = Conv2D(32, 3, activation='relu', padding='same')(concat1)

    u2 = UpSampling2D()(c4)
    concat2 = concatenate([u2, c1])
    c5 = Conv2D(16, 3, activation='relu', padding='same')(concat2)

    outputs = Conv2D(1, 1, activation='sigmoid')(c5)
    model = Model(inputs, outputs)
    return model

# Create U-Net
model = build_unet((128, 128, 1))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load image and mask from user input
img_path = input("Enter path to grayscale image (128x128): ")
mask_path = input("Enter path to mask image (128x128): ")

if not os.path.exists(img_path) or not os.path.exists(mask_path):
    print("Invalid path(s)")
else:
    img = load_img(img_path, color_mode='grayscale', target_size=(128, 128))
    mask = load_img(mask_path, color_mode='grayscale', target_size=(128, 128))

    img_array = img_to_array(img) / 255.0
    mask_array = img_to_array(mask) / 255.0

    img_input = np.expand_dims(img_array, axis=0)
    mask_input = np.expand_dims(mask_array, axis=0)

    # Train (optional, minimal example)
    model.fit(img_input, mask_input, epochs=10, verbose=1)

    # Predict
    pred_mask = model.predict(img_input)[0].squeeze()

    # Display
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.title("Input Image")
    plt.imshow(img_array.squeeze(), cmap='gray')
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.title("Ground Truth Mask")
    plt.imshow(mask_array.squeeze(), cmap='gray')
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.title("Predicted Mask")
    plt.imshow(pred_mask, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
```

---

### 🧪 Input Instructions

* Prepare a **grayscale image** and its corresponding **binary mask**, both sized `128x128`
* Provide their **full file paths** when prompted

---

Here is the full implementation of **Model 6: Image Classification using VGG16**, which:

* Uses the pre-trained **VGG16** model
* Accepts an image path from **keyboard input**
* Predicts the class and displays the result with confidence
* Shows the input image

---

## ✅ **Model 6: Transfer Learning — VGG16**

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import os

# Load pre-trained VGG16 model with ImageNet weights
model = VGG16(weights='imagenet')

# Ask for input image path
img_path = input("Enter path to an image (224x224 or larger): ")

if not os.path.exists(img_path):
    print("Invalid image path.")
else:
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_array)

    # Predict
    preds = model.predict(img_preprocessed)
    decoded = decode_predictions(preds, top=1)[0][0]

    # Display result
    plt.imshow(img)
    plt.title(f"Predicted: {decoded[1]} ({decoded[2]*100:.2f}%)")
    plt.axis('off')
    plt.show()
```

---

### 📝 Example

If you enter an image of an elephant, the output might show:

```
Predicted: African_elephant (97.12%)
```

And the image will be displayed.

---


Here is the full implementation of **Model 7: Binary Classification with Huber Loss**, which:

* Builds a simple **binary image classifier**
* Uses **Huber loss** for training (robust to outliers)
* Accepts an image path from **keyboard input**
* Predicts whether it belongs to class `1` or `0`
* Shows the input image and prediction

---

## ✅ **Model 7: Binary Classifier with Huber Loss**

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing import image
from tensorflow.keras.losses import Huber
import os

# Dummy model for binary classification (input: 64x64x3)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile using Huber loss
model.compile(optimizer='adam', loss=Huber(), metrics=['accuracy'])

# Dummy training for demonstration
# Normally you would train on real binary labeled data
x_dummy = np.random.rand(10, 64, 64, 3)
y_dummy = np.random.randint(0, 2, 10)
model.fit(x_dummy, y_dummy, epochs=3)

# Load image from user
img_path = input("Enter image path (64x64): ")
if not os.path.exists(img_path):
    print("Invalid image path.")
else:
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img) / 255.0
    img_input = np.expand_dims(img_array, axis=0)

    # Predict
    pred = model.predict(img_input)[0][0]
    result = 1 if pred >= 0.5 else 0

    # Show image and prediction
    plt.imshow(img)
    plt.title(f"Predicted Class: {result} (Confidence: {pred:.2f})")
    plt.axis('off')
    plt.show()
```

---

### 🔍 Notes:

* **Model Input**: RGB image resized to **64×64**
* **Output**: Binary class (0 or 1)
* Model uses `sigmoid` output + `Huber()` loss for regression-like robustness

---

Here is the full implementation of **Model 8: Focal Loss for Imbalanced Binary Classification**, which:

* Builds a simple CNN model
* Uses **Focal Loss** (custom loss) for class imbalance handling
* Accepts an image from **keyboard input**
* Predicts class 0 or 1 with confidence
* Displays the image and prediction result

---

## ✅ **Model 8: Focal Loss Binary Classifier**

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing import image
import os

# Define focal loss
def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)
        return tf.reduce_mean(weight * cross_entropy)
    return loss

# Build model (input 64x64 RGB image)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile with focal loss
model.compile(optimizer='adam', loss=focal_loss(), metrics=['accuracy'])

# Dummy training (replace with real imbalanced dataset)
x_dummy = np.random.rand(20, 64, 64, 3)
y_dummy = np.random.randint(0, 2, 20)
model.fit(x_dummy, y_dummy, epochs=3)

# Load test image from user
img_path = input("Enter path to an image (64x64): ")
if not os.path.exists(img_path):
    print("Invalid image path.")
else:
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img) / 255.0
    img_input = np.expand_dims(img_array, axis=0)

    # Predict
    pred = model.predict(img_input)[0][0]
    result = 1 if pred >= 0.5 else 0

    # Show result
    plt.imshow(img)
    plt.title(f"Predicted Class: {result} (Confidence: {pred:.2f})")
    plt.axis('off')
    plt.show()
```

---

### 📝 What’s special?

* Focal loss is excellent for imbalanced datasets like:

  * Tumor vs. non-tumor
  * Defect vs. normal
* Focuses more on **hard-to-classify** examples

---

Here is the full implementation of **Model 9: Object Detection using TensorFlow Hub**, which:

* Loads a **pre-trained object detection model**
* Accepts an image path from **keyboard input**
* Predicts **bounding boxes and class labels**
* Displays the original image with **boxes and labels**

---

## ✅ **Model 9: Object Detection with TF Hub**

```python
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import cv2

# Load detection model from TF Hub
detector = hub.load("https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1")

# Load image path from user
img_path = input("Enter path to an image: ")
if not os.path.exists(img_path):
    print("Invalid path.")
else:
    # Load image
    img = Image.open(img_path).convert("RGB")
    img_resized = img.resize((640, 640))
    img_np = np.array(img_resized) / 255.0
    input_tensor = tf.convert_to_tensor([img_np], dtype=tf.float32)

    # Run detection
    result = detector(input_tensor)
    result = {key: value.numpy() for key, value in result.items()}

    boxes = result["detection_boxes"]
    classes = result["detection_class_entities"]
    scores = result["detection_scores"]

    # Draw boxes
    img_cv = np.array(img_resized)
    h, w, _ = img_cv.shape
    for i in range(min(5, len(scores))):
        if scores[i] < 0.5:
            continue
        box = boxes[i]
        y1, x1, y2, x2 = int(box[0]*h), int(box[1]*w), int(box[2]*h), int(box[3]*w)
        label = f"{classes[i].decode('utf-8')} ({scores[i]*100:.1f}%)"
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img_cv, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

    # Show image
    plt.figure(figsize=(10, 8))
    plt.imshow(img_cv)
    plt.axis('off')
    plt.title("Detected Objects")
    plt.show()
```

---

### ✅ Output

* Draws top 5 detected bounding boxes with labels (if score ≥ 0.5)
* Example: `Person (92.3%)`, `Dog (88.1%)`
  

Here is the full implementation of **Model 10: Image Classifier using `image_dataset_from_directory`**, which:

* Loads images from folders using TensorFlow’s **`image_dataset_from_directory`**
* Trains a CNN on them
* Accepts an image path from **keyboard input**
* Predicts and displays the image’s class with confidence

---

## ✅ **Model 10: Custom Dataset Loader (Binary/Multiclass Classifier)**

### 🗂 Folder Structure Required:

```
dataset/
  ├── class_0/
  │   ├── img1.jpg
  │   └── ...
  └── class_1/
      ├── img2.jpg
      └── ...
```

### 🧠 Code:

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
import os

# Load dataset from folders
dataset_path = input("Enter path to dataset folder (with subfolders per class): ")
img_size = (128, 128)
batch_size = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    image_size=img_size,
    batch_size=batch_size,
    validation_split=0.2,
    subset="training",
    seed=123
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    image_size=img_size,
    batch_size=batch_size,
    validation_split=0.2,
    subset="validation",
    seed=123
)

# Get class names
class_names = train_ds.class_names
num_classes = len(class_names)

# Optimize pipeline
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# Build CNN model
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(128, 128, 3)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compile and train
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_ds, validation_data=val_ds, epochs=5)

# Load test image from user
img_path = input("Enter path to test image (128x128 or larger): ")
if not os.path.exists(img_path):
    print("Invalid image path.")
else:
    test_img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(test_img) / 255.0
    img_input = np.expand_dims(img_array, axis=0)

    # Predict
    pred = model.predict(img_input)
    pred_class = class_names[np.argmax(pred)]
    confidence = np.max(pred) * 100

    # Show result
    plt.imshow(test_img)
    plt.title(f"Predicted: {pred_class} ({confidence:.2f}%)")
    plt.axis('off')
    plt.show()
```

---

### 📝 Features:

* Works for **binary or multi-class** datasets
* Automatically maps folder names to class labels
* Trains a small CNN and performs **live prediction**

---

✅ You now have **10 full models** complete with:

* Training
* Keyboard input
* Image processing
* Live prediction with output display

