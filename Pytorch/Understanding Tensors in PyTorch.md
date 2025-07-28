
---

# 🧠 Understanding **Tensors** in PyTorch

---

## 📌 What is a Tensor?

A **tensor** is a multi-dimensional array, similar to a NumPy array, but with additional capabilities optimized for deep learning. Tensors are the **basic building blocks** of PyTorch.

> 🔸 Think of tensors as generalizations of scalars, vectors, and matrices to **n-dimensions**.

| Tensor Type | PyTorch Shape                               | Description         |
| ----------- | ------------------------------------------- | ------------------- |
| Scalar      | `torch.tensor(5)` → `[]`                    | Single number       |
| Vector      | `torch.tensor([1, 2, 3])` → `[3]`           | 1D tensor           |
| Matrix      | `torch.tensor([[1, 2], [3, 4]])` → `[2, 2]` | 2D tensor           |
| n-D Tensor  | Higher dimensions                           | 3D, 4D, ... tensors |

---

## 🔧 Creating Tensors

### 1. **From Python Lists**

```python
import torch

# 1D Tensor (vector)
x = torch.tensor([1, 2, 3])
# 2D Tensor (matrix)
y = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
```

### 2. **With Initial Values**

```python
torch.zeros((2, 3))      # 2x3 tensor filled with zeros
torch.ones((2, 2))       # 2x2 tensor filled with ones
torch.full((2, 2), 7)    # tensor filled with 7s
torch.eye(3)             # identity matrix
```

### 3. **With Random Values**

```python
torch.rand(2, 3)         # Uniform distribution [0,1)
torch.randn(2, 3)        # Normal distribution (mean=0, std=1)
```

### 4. **From NumPy**

```python
import numpy as np
arr = np.array([[1, 2], [3, 4]])
tensor = torch.from_numpy(arr)
```

---

## 🔄 Tensor Operations

### ➤ Basic Math

```python
a = torch.tensor([1, 2])
b = torch.tensor([3, 4])
print(a + b)       # tensor([4, 6])
print(a * b)       # tensor([3, 8])
```

### ➤ Matrix Multiplication

```python
x = torch.tensor([[1, 2], [3, 4]])
y = torch.tensor([[5, 6], [7, 8]])
print(torch.mm(x, y))
```

### ➤ Reshaping

```python
x = torch.arange(9)       # tensor([0, 1, ..., 8])
x = x.reshape(3, 3)
```

### ➤ Indexing and Slicing

```python
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
x[0]         # tensor([1, 2, 3])
x[:, 1]      # column index 1 → tensor([2, 5])
```

---

## 💻 Working with GPU

Move a tensor to GPU (if available):

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.tensor([1.0, 2.0, 3.0], device=device)
```

---

## 🔁 Gradients Support (for ML)

Make a tensor require gradients (for autograd):

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)  # Should print 4.0
```

---

## 🔄 Tensors vs NumPy Arrays

| Feature       | `torch.Tensor`                | `numpy.ndarray` |
| ------------- | ----------------------------- | --------------- |
| GPU support   | ✅ (with `.cuda()`)            | ❌               |
| Auto-diff     | ✅ (with `autograd`)           | ❌               |
| Deep learning | ✅ (PyTorch-native)            | ❌               |
| Interoperable | ✅ (`from_numpy()`, `numpy()`) | ✅               |

---

## 🧪 Examples to Try Yourself

```python
# Element-wise operations
a = torch.tensor([2.0, 3.0])
b = torch.tensor([4.0, 5.0])
print(torch.pow(a, 2))         # tensor([4., 9.])
print(torch.sqrt(b))           # tensor([2.0, 2.2361])

# Broadcasting
a = torch.tensor([[1], [2]])
b = torch.tensor([3, 4])
print(a + b)                   # shape becomes [2,2] due to broadcasting
```

---

## 📚 Summary

| Concept         | Example                    |
| --------------- | -------------------------- |
| Create          | `torch.tensor([1,2])`      |
| Shape           | `x.shape`                  |
| Device transfer | `x.to("cuda")`             |
| Gradients       | `requires_grad=True`       |
| Operations      | `+`, `*`, `.matmul()` etc. |

---

---

## 🌟 ২০টি PyTorch Tensor উদাহরণ 

1. **ফ্যাক্টরি creation** (zeros, ones, rand, full ইত্যাদি)

   ```python
   torch.zeros(2,3); torch.ones(2,2); torch.rand(2,3); torch.full((2,2), 7)
   ```

   ([docs.pytorch.org][1])

2. **torch.empty() ব্যবহার ও uninitialized memory প্রদর্শন**

   ```python
   x = torch.empty(3,4); print(x)
   ```

   ([docs.pytorch.org][1])

3. **element-wise multiplication, sum ও indexing**

   ```python
   a = torch.randn(2,3); b = torch.randn(2,3)
   print(a * b); print(a.sum()); print(a[1,2])
   ```

   ([Wikipedia][2])

4. **লিনস্পেস ও ট্রিগ সাইন ফাংশন fitting**

   ```python
   x = torch.linspace(-math.pi, math.pi, 2000)
   y = torch.sin(x)
   y_pred = a + b*x + c*x**2 + d*x**3
   loss = (y_pred - y).pow(2).sum()
   ```

   ([docs.pytorch.org][3], [h-huang.github.io][4])

5. **টেনসর ডিভাইস ব্যবহার (CPU/GPU)**

   ```python
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   x = torch.tensor([1.0,2.0,3.0], device=device)
   ```

   ([docs.pytorch.org][3], [Wikipedia][2])

6. **NumPy থেকে টেনসর কনভার্শন**

   ```python
   arr = np.array([[1,2],[3,4]])
   tensor = torch.from_numpy(arr)
   ```

   ([GitHub][5])

7. **reshape ও transpose**

   ```python
   x = torch.arange(9).reshape(3,3)
   y = x.t()
   ```

   ([docs.pytorch.org][1])

8. **broadcasting উদাহরণ**

   ```python
   a = torch.tensor([[1],[2]]); b = torch.tensor([3,4])
   print(a + b)  # shape becomes [2,2]
   ```

   ([armanasq.github.io][6])

9. **logical reductions ও axis-wise reductions**

   ```python
   t.sum(dim=0); t.mean(dim=1); t.min(), t.max()
   ```

   ([armanasq.github.io][6])

10. **one-hot encoding utility**

    ```python
    idx = [2,0,1]
    onehot = torch.nn.functional.one_hot(torch.tensor(idx), num_classes=3)
    ```

    ([GitHub][7])

11. **column normalization (mean‑std normalization)**

    ```python
    x = torch.rand(4,4)
    norm = (x - x.mean(dim=0)) / x.std(dim=0)
    ```

    ([GitHub][7])

12. **batched matrix multiplication**

    ```python
    x = torch.rand(10,3,4); y = torch.rand(10,4,5)
    out = torch.bmm(x,y)
    ```

    ([GitHub][7])

13. **slice indexing ও modifying tensor**

    ```python
    t = torch.arange(16).reshape(4,4)
    t[:2, :2] = -1
    ```

    ([GitHub][7])

14. **reverse rows ও shuffle columns**

    ```python
    rev = t.flip(0); shuffled = t[:, torch.randperm(t.size(1))]
    ```

    ([GitHub][7])

15. **scatter‑gather বা masked operations**

    ```python
    mask = (x > 0.5)
    x[mask] = 1.0
    ```

    ([armanasq.github.io][6])

16. **einsum উদাহরণ (advanced broadcasting)**

    ```python
    out = torch.einsum('ij,jk->ik', mat1, mat2)
    ```

    ([armanasq.github.io][6])

17. **detaching tensors from autograd**

    ```python
    y = x.detach()
    ```

    ([armanasq.github.io][6])

18. **requires\_grad ও gradient calculation**

    ```python
    x = torch.tensor(2.0, requires_grad=True)
    y = x**3 + 2*x
    y.backward()
    print(x.grad)
    ```

    ([armanasq.github.io][6])

19. **tensor.to(dtype=torch.float16) mixed precision**

    ```python
    x = x.half()
    ```

    ([armanasq.github.io][6])

20. **saving ও loading tensor**

    ```python
    torch.save(t, 'tensor.pt'); t2 = torch.load('tensor.pt')
    ```

    ([GitHub][8])

---

## 🧠 প্রত্যেক উদাহরণের মূল লক্ষ্য:

* **Tensor creation**, **indexing**, **reshaping**, **broadcasting**, **reductions**, **device handling**, **autograd**, এবং **utilities**—সবগুলো গুরুত্বপূর্ণ ধারণা।
* ঐতিহাসিক ও শিক্ষামূলক উদ্দেশ্যে অফিসিয়াল টিউটোরিয়াল, Gist এবং GitHub রিপোজিটরি থেকে নেয়া উদাহরণগুলো ব্যবহার করা হয়েছে।

---

## ✅ আপনার শেখার জন্য পরবর্তী পদক্ষেপ:

* প্রত্যেক উদাহরণ নিজে টাইপ করে চালান ও আউটপুট পর্যবেক্ষণ করুন।
* `.shape`, `.dtype`, `.device`, `.grad_fn`, `.requires_grad` ইত্যাদি পরীক্ষা করে বুঝুন।
* ছোট ছোট প্রজেক্ট তৈরি করুন: polynomial fitting, matrix ops, broadcasting challenges ইত্যাদি।

---


[1]: https://docs.pytorch.org/tutorials/beginner/introyt/tensors_deeper_tutorial.html?utm_source=chatgpt.com "Introduction to PyTorch Tensors — PyTorch Tutorials 2.7.0+cu126 ..."
[2]: https://en.wikipedia.org/wiki/PyTorch?utm_source=chatgpt.com "PyTorch"
[3]: https://docs.pytorch.org/tutorials/beginner/pytorch_with_examples?utm_source=chatgpt.com "Learning PyTorch with Examples — PyTorch Tutorials 2.7.0+cu126 ..."
[4]: https://h-huang.github.io/tutorials/beginner/pytorch_with_examples.html?utm_source=chatgpt.com "Learning PyTorch with Examples — PyTorch Tutorials 1.8.1+cu102 ..."
[5]: https://github.com/ml-dev-world/pytorch-fundamentals?utm_source=chatgpt.com "ml-dev-world/pytorch-fundamentals - GitHub"
[6]: https://armanasq.github.io/Deep-Learning/PyTorch-Tensors/?utm_source=chatgpt.com "A Profound Journey into PyTorch Tensors: A Comprehensive Tutorial"
[7]: https://github.com/AnmolGulati6/PyTorch-101-Tensor-Operations-and-Utilities?utm_source=chatgpt.com "PyTorch-101-Tensor-Operations-and-Utilities - GitHub"
[8]: https://github.com/sedwna/PyTorch-Tensor-Basics/?utm_source=chatgpt.com "GitHub - sedwna/PyTorch-Tensor-Basics: This repository contains ..."

---

## ১. Polynomial Fitting to Sine Function

**উদ্দেশ্য**: একটি ৩য়‑অর্ডার পলিনোমিয়াল ফাংশন fit করা কোনও dataset‑এ (যেমন sine wave)
**কোড**:

```python
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)
a,b,c,d = torch.randn((),requires_grad=True), ...
for t in range(2000):
    y_pred = a + b*x + c*x**2 + d*x**3
    loss = (y_pred - y).pow(2).sum()
    loss.backward()
    # optimizer.step() style update
```

এখানে tensor arithmetic, `.pow()`, `.sum()`, gradient tracking ব্যবহৃত হয়েছে ([docs.pytorch.org][1], [Python Mania][2])।

---

## ২. Image Classification (MNIST বা CIFAR‑10)

**উদ্দেশ্য**: টেনসর দিয়ে input data, CNN, loss, optimizer training নিয়ে কাজ করা
**কোড**:

```python
images = images.to(device)
labels = labels.to(device)
outputs = model(images)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
```

Tensor batch operations, `.to(device)`, gradient update ইত্যাদি ব্যবহৃত। এই কাঠামোটি standard vision models‑এ ব্যবহৃত হয় ([myscale.com][3], [Placement Preparation][4])।

---

## ৩. Text Classification using LSTM

**উদ্দেশ্য**: টেক্সটকে টোকেন করে LSTM ভিত্তিক sentiment classification করা
**কোড**:

```python
embedded = embedding(input_seq)
output, (h_n, c_n) = lstm(embedded)
logits = classifier(h_n[-1])
```

টেনসর shape manipulation, embeddings & RNN ফিড করা হয় ([GitHub][5])।

---

## ৪. Transfer Learning with Pretrained ResNet

**উদ্দেশ্য**: pretrained CNN model ফাইন‑টিউন করা
**কোড**:

```python
model = torchvision.models.resnet18(pretrained=True)
for param in model.parameters(): param.requires_grad = False
model.fc = nn.Linear(512, num_classes)
```

টেনসর forward/backward বাসি operations সহ structure modification করা হয় ([Placement Preparation][4], [ProjectPro][6])।

---

## ৫. DCGAN Image Generation

**উদ্দেশ্য**: GAN‑based image generation, tensors এর মাধ্যমে noise vector & generated images ব্যবহৃত
**কোড**:

```python
noise = torch.randn(batch_size, nz, 1, 1, device=device)
fake = netG(noise)
```

টেনসর initialization, backward on generator loss, ও discriminator loss–এ gradient tracking করা হয় ([docs.pytorch.org][1])।

---

## ৬. Plant Disease Detection (Image Segmentation)

**উদ্দেশ্য**: Leaf image থেকে segmentation mask predict করা U‑Net model দিয়ে
**কোড**:

```python
pred = model(input_image)
loss = criterion(pred, mask)
loss.backward()
```

টেনসর dimension handling, BCE/softmax loss ব্যাবহার করে segmentation output এ gradient tracking করা হয় ([Placement Preparation][4])।

---

## ৭. Emotion Recognition from Video (Audio + Vision tensors)

**উদ্দেশ্য**: ভিডিও বা audio থেকে emotion classify করানো
**কোড**:

```python
audio_feat = torchaudio.transforms.MFCC()(waveform)
video_feat = vision_backbone(frames)
combined = torch.cat((audio_feat, video_feat), dim=1)
out = classifier(combined)
```

টেনসর concatenation, multi‑modal fusion দেখানো হয় ([Omdena][7])।

---

## ৮. Stock Price Prediction using LSTM Time Series

**উদ্দেশ্য**: past price series predict করা LSTM দিয়ে
**কোড**:

```python
out, _ = lstm(seq_batch)
pred = linear(out[:, -1, :])
```

Time‑series টেনসর shapes, sequence batch processing, tensor gradients ব্যবহৃত হয় ([cognitiveclass.ai][8], [ProjectPro][6])।

---

## ৯. Siamese Network for Image Similarity

**উদ্দেশ্য**: Siamese twin network দিয়ে two-image embeddings compare করা
**কোড**:

```python
out1 = net(img1)
out2 = net(img2)
dist = F.pairwise_distance(out1, out2)
loss = contrastive_loss(dist, label)
loss.backward()
```

Tensor distance, embedding generation, contrastive loss tracking দেখা যায় ([myscale.com][3])।

---

## 🔟 Image Segmentation with U‑Net on Medical Data

**উদ্দেশ্য**: Retina বা MRI image segmentation
**কোড**:

```python
pred_mask = model(image_tensor)
loss = dice_loss(pred_mask, mask_tensor)
loss.backward()
optimizer.step()
```

Tensor operations, element-wise dice coefficient, training loop integration—all shown here ([cognitiveclass.ai][8], [ProjectPro][6])।

---

### 🧭 সারসংক্ষেপ:

| প্রজেক্ট             | মূল টেনসর কনসেপ্ট                     |
| -------------------- | ------------------------------------- |
| Polynomial fitting   | tensor math, autograd                 |
| CNN Classifier       | batch tensors, loss, optimizer        |
| LSTM Text            | embeddings, seq‑tensor                |
| Transfer Learning    | pretrained weights, requires\_grad    |
| GAN                  | noise tensor, generator/discriminator |
| Image Segmentation   | multi-dimensional input/output        |
| Multi-modal fusion   | concat, feature tensors               |
| Time Series LSTM     | sequence tensors, prediction          |
| Siamese Network      | embedding distance tensors            |
| Medical Segmentation | mask tensors, dice loss               |

---



[1]: https://docs.pytorch.org/tutorials/beginner/pytorch_with_examples.html?utm_source=chatgpt.com "Learning PyTorch with Examples"
[2]: https://pythonmania.org/how-to-use-pytorch-the-ultimate-guide-case-study-example/?utm_source=chatgpt.com "How to Use PyTorch: The Ultimate Guide + Case Study + Example"
[3]: https://myscale.com/blog/master-pytorch-example-projects-hands-on-guide-learning/?utm_source=chatgpt.com "Master PyTorch Example Projects: Your Hands-On Guide"
[4]: https://www.placementpreparation.io/blog/pytorch-project-ideas-for-beginners/?utm_source=chatgpt.com "10 Best PyTorch Project Ideas for Beginners [With Source Code]"
[5]: https://github.com/es-OmarHani/pytorch-projects?utm_source=chatgpt.com "PyTorch Projects and Tutorials - GitHub"
[6]: https://www.projectpro.io/projects/data-science-projects/pytorch?utm_source=chatgpt.com "5+ PyTorch Projects for Beginners with Source Code to Practice"
[7]: https://www.omdena.com/blog/best-pytorch-projects?utm_source=chatgpt.com "Best PyTorch Projects for Beginners in 2024 - Omdena"
[8]: https://cognitiveclass.ai/learn/pytorch-projects?utm_source=chatgpt.com "PyTorch Projects - cognitiveclass.ai"

# PyTorch Tensor Documentation with Examples

## Introduction to PyTorch Tensors

Tensors are the central data structure in PyTorch, similar to NumPy arrays but with additional GPU acceleration and automatic differentiation capabilities.

---

## Creating Tensors

```python
import torch
import math
```

### 1. Empty Tensor

```python
x = torch.empty(3, 4)
```

* Creates a 3x4 uninitialized tensor

### 2. Zeros, Ones, Random

```python
zeros = torch.zeros(2, 3)
ones = torch.ones(2, 3)
torch.manual_seed(1729)
random = torch.rand(2, 3)
```

### 3. Random Seed Reproducibility

```python
torch.manual_seed(1729)
random1 = torch.rand(2, 3)
random2 = torch.rand(2, 3)
torch.manual_seed(1729)
random3 = torch.rand(2, 3)
```

### 4. Tensor Like

```python
x = torch.empty(2, 2, 3)
empty_like_x = torch.empty_like(x)
zeros_like_x = torch.zeros_like(x)
ones_like_x = torch.ones_like(x)
rand_like_x = torch.rand_like(x)
```

### 5. From Python Collections

```python
some_constants = torch.tensor([[3.1415926, 2.71828], [1.61803, 0.0072897]])
some_integers = torch.tensor((2, 3, 5, 7, 11, 13, 17, 19))
```

---

## Tensor Data Types

```python
a = torch.ones((2, 3), dtype=torch.int16)
b = torch.rand((2, 3), dtype=torch.float64) * 20.
c = b.to(torch.int32)
```

---

## Math & Logic Operations

### Scalar Arithmetic

```python
ones = torch.zeros(2, 2) + 1
twos = torch.ones(2, 2) * 2
threes = (torch.ones(2, 2) * 7 - 1) / 2
fours = twos ** 2
sqrt2s = twos ** 0.5
```

### Tensor-Tensor Arithmetic

```python
fives = ones + fours
dozens = threes * fours
```

### Broadcasting

```python
rand = torch.rand(2, 4)
doubled = rand * (torch.ones(1, 4) * 2)
```

---

## More Math Operations

```python
a = torch.rand(2, 4) * 2 - 1
print(torch.abs(a))
print(torch.ceil(a))
print(torch.floor(a))
print(torch.clamp(a, -0.5, 0.5))
```

### Trigonometry

```python
angles = torch.tensor([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
sines = torch.sin(angles)
inverses = torch.asin(sines)
```

### Bitwise

```python
b = torch.tensor([1, 5, 11])
c = torch.tensor([2, 7, 10])
print(torch.bitwise_xor(b, c))
```

### Comparisons and Reductions

```python
d = torch.tensor([[1., 2.], [3., 4.]])
e = torch.ones(1, 2)
print(torch.eq(d, e))
print(torch.max(d))
print(torch.mean(d))
print(torch.std(d))
```

---

## Vectors & Matrices

```python
v1 = torch.tensor([1., 0., 0.])
v2 = torch.tensor([0., 1., 0.])
m1 = torch.rand(2, 2)
m2 = torch.tensor([[3., 0.], [0., 3.]])
m3 = torch.linalg.matmul(m1, m2)
print(torch.linalg.svd(m3))
```

---

## In-place Operations

```python
a = torch.tensor([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
print(torch.sin(a))

b = torch.tensor([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
print(torch.sin_(b))
```

---

## Memory-Efficient Operations

```python
a = torch.rand(2, 2)
b = torch.rand(2, 2)
c = torch.zeros(2, 2)
torch.matmul(a, b, out=c)
```

---

## Copying and Cloning Tensors

```python
a = torch.ones(2, 2)
b = a.clone()
c = a.detach().clone()
```

---

## Moving to Accelerators (GPU)

```python
my_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
x = torch.rand(2, 2, device=my_device)
y = x.to(my_device)
```

---

## Manipulating Tensor Shapes

### Unsqueeze / Squeeze

```python
a = torch.rand(3, 226, 226)
b = a.unsqueeze(0)
c = b.squeeze(0)
```

### Reshape

```python
output3d = torch.rand(6, 20, 20)
input1d = output3d.reshape(6 * 20 * 20)
```

---

## NumPy Interoperability

```python
import numpy as np
numpy_array = np.ones((2, 3))
torch_tensor = torch.from_numpy(numpy_array)

pytorch_rand = torch.rand(2, 3)
numpy_rand = pytorch_rand.numpy()
```

### Shared Memory

```python
numpy_array[1, 1] = 23
print(torch_tensor)

pytorch_rand[1, 1] = 17
print(numpy_rand)
```

---


