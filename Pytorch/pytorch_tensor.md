**টেনসর (Tensor) কী?**
ছোট করে: টেনসর হলো **বহুমাত্রিক (multi-dimensional) অ্যারে**—যেটা স্কেলার/ভেক্টর/ম্যাট্রিক্সের সাধারণীকরণ।

* **স্কেলার**: একটিই সংখ্যা → আকার/shape: `()`
* **ভেক্টর**: সংখ্যার লিস্ট (১-ডি) → shape: `(n,)`
* **ম্যাট্রিক্স**: সারি-কলামের টেবিল (২-ডি) → shape: `(m, n)`
* **টেনসর**: ৩-ডি, ৪-ডি, … যেকোনো ডাইমেনশন → উদাহরণ: `(N, C, H, W)`

PyTorch-এ টেনসর মানে হলো **ডেটা + টাইপ (dtype) + শেপ + ডিভাইস (CPU/GPU) + গ্র্যাড সেটিং**—যার ওপর দ্রুত গাণিতিক অপারেশন, অটো-গ্র্যাড, ব্রডকাস্টিং ইত্যাদি করা যায়।

### PyTorch উদাহরণ

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0])     # ভেক্টর (1-D)
M = torch.randn(2, 3)                 # ম্যাট্রিক্স (2-D), নরমাল ডিস্ট্রিবিউশন
img = torch.randn(3, 224, 224)        # ইমেজ টেনসর (C,H,W) = (RGB, উচ্চতা, প্রস্থ)

print(x.shape, x.ndim)   # torch.Size([3])  1
print(M.shape, M.ndim)   # torch.Size([2, 3])  2
print(img.shape, img.ndim)# torch.Size([3, 224, 224])  3
```

### কেন দরকার?

* **ডেটা প্রতিনিধিত্ব**: ছবি, অডিও, টেক্সট, ব্যাচ ডেটা—সবই টেনসরে রাখা যায়।
* **দ্রুত হিসাব**: GPU/CPU-তে ভেক্টরাইজড অপারেশন।
* **অটো-গ্র্যাড**: নিউরাল নেটওয়ার্ক ট্রেনিংয়ে গ্র্যাডিয়েন্ট স্বয়ংক্রিয়ভাবে বের করে।



---

# 1) টেনসর বানানো (creation)

```python
import torch

# Shape ও dtype সহ শূন্য/এক/র‍্যান্ডম
a = torch.zeros((2,3), dtype=torch.float32)
b = torch.ones(4, device="cpu")
c = torch.rand(3, 3)           # [0,1) uniform
d = torch.randn(2, 5)          # normal(0,1)

# লিস্ট/NumPy থেকে
e = torch.tensor([[1,2],[3,4]])
import numpy as np
f = torch.from_numpy(np.array([5,6,7]))  # শেয়ার্ড মেমরি
g = torch.arange(0, 10, step=2)          # 0,2,4,6,8
h = torch.linspace(0, 1, steps=5)        # 0.,0.25,...,1.

# dtype, device সেট
x = torch.empty(2,3, dtype=torch.float64, device="cpu")
```

> টিপ: `torch.from_numpy` করলে NumPy ও Tensor একই মেমরি শেয়ার করে; কপি চাইলে `.clone()` নিন।

---

# 2) শেপ/রেশেপ/ডাইমেনশন অপস

```python
t = torch.arange(12)           # shape [12]
t = t.reshape(3,4)             # ভিউ: shape [3,4]
t2 = t.view(-1, 2)             # view: contiguous হলে দ্রুত
t3 = t.permute(1,0)            # ডাইমেনশন swap
t4 = t.transpose(0,1)          # 2D transpose
t5 = t.unsqueeze(0)            # dim যোগ: [1,3,4]
t6 = t.squeeze()               # সাইজ-১ ডিমেনশন বাদ
cat = torch.cat([t, t], dim=0) # কেটেনেট
stk = torch.stack([t, t], 0)   # নতুন dim যোগ করে স্ট্যাক
```

> টিপ: `view` contiguous মেমরি চাই; না হলে `contiguous().view(...)` ব্যবহার করুন।

---

# 3) ইনডেক্সিং/স্লাইসিং/মাস্কিং

```python
A = torch.arange(1,13).reshape(3,4)
row = A[1]                 # 2nd row
col = A[:, 2]              # 3rd column
sub = A[0:2, 1:3]          # slicing

mask = A % 2 == 0
even = A[mask]             # boolean indexing

where_res = torch.where(A>5, A, torch.zeros_like(A))
gathered = torch.gather(A, dim=1, index=torch.tensor([[0,3,1,2],
                                                      [1,2,3,0],
                                                      [3,2,1,0]]))
```

---

# 4) ব্রডকাস্টিং ও বেসিক ম্যাথ

```python
x = torch.randn(3,1)
y = torch.randn(1,4)
z = x + y                   # broadcasting -> [3,4]

u = torch.tensor([1.,2.,3.])
v = torch.tensor([4.,5.,6.])
dot = (u*v).sum()           # ডট প্রোডাক্ট
hadamard = u * v            # elementwise
p = torch.pow(u, 2)
absu = torch.abs(u)
clamped = torch.clamp(u, min=1.5, max=2.5)
```

**রিডাকশন:**
`sum`, `mean`, `std`, `var`, `min`, `max`, `argmin`, `argmax`, `prod`, `logsumexp`—সবগুলোই `dim=` ও `keepdim=` সাপোর্ট করে।

```python
M = torch.randn(4,5)
col_mean = M.mean(dim=0)     # shape [5]
row_sum  = M.sum(dim=1, keepdim=True)  # shape [4,1]
```

---

# 5) মেট্রিক্স/লিনিয়ার অ্যালজেব্রা

```python
A = torch.randn(3,3)
B = torch.randn(3,3)
matmul = A @ B                       # বা torch.matmul(A,B)
tA = A.T                             # transpose
eye = torch.eye(3)

# লিনিয়ার সলভ/ডিকম্প
x = torch.linalg.solve(A, torch.randn(3,1))
U, S, Vh = torch.linalg.svd(A)
Q, R = torch.linalg.qr(A)
eigvals, eigvecs = torch.linalg.eig(A)
det = torch.linalg.det(A)
inv = torch.linalg.inv(A)
```

> টিপ: নিউরাল নেটে সাধারণত `@`/`matmul`/`mm` ব্যবহারই যথেষ্ট; `linalg.*` হলো নিউমেরিক্স কাজের জন্য।

---

# 6) র‍্যান্ডম, সিড, ডিস্ট্রিবিউশন

```python
torch.manual_seed(42)
r = torch.rand(2,2)
n = torch.randn(3,3)

# ডিস্ট্রিবিউশন অবজেক্ট
dist = torch.distributions.Normal(loc=0., scale=1.)
samples = dist.sample((5,))          # 5টি স্যাম্পল
logp = dist.log_prob(torch.tensor([0.0, 1.0]))
```

---

# 7) টাইপ/ডিভাইস/কাস্টিং

```python
x = torch.tensor([1,2,3], dtype=torch.int32)
xf = x.float()                # to float32
xd = x.to(dtype=torch.float64, device="cpu")

# CPU <-> GPU
if torch.cuda.is_available():
    gpu = xf.cuda()           # বা .to("cuda")
    back = gpu.cpu()
```

> টিপ: dtype promotion রুল আছে; মিক্সড dtype অপে PyTorch উপযুক্ত dtype বেছে নেয় (কিছু ক্ষেত্রে কাস্ট নিজে নিয়ন্ত্রণ করুন)।

---

# 8) অটোগ্র্যাড (গ্র্যাডিয়েন্ট ট্র্যাকিং)

```python
# requires_grad = True হলেই গ্রাফ ট্র্যাক করবে
w = torch.randn(3, requires_grad=True)
b = torch.randn(3, requires_grad=True)

x = torch.randn(3)
y = (w * x + b).sum()     # scalar loss-এর মত

y.backward()              # dy/dw, dy/db হিসাব
print(w.grad, b.grad)

# গ্র্যাড ছাড়া ব্লক
with torch.no_grad():
    z = (w * x + b).sum()

# টেনসরকে গ্রাফ থেকে আলাদা
y_detached = y.detach()
```

**ইন-প্লেস অপস সতর্কতা**: `tensor.add_()` টাইপের ইন-প্লেস অপস গ্র্যাড গ্রাফ ভেঙে দিতে পারে—মডেল ট্রেনিংয়ে সাবধানে।

---

# 9) বাছাই/সর্টিং/টপ-k/ইউনিক

```python
t = torch.tensor([3,1,2,2,5])
sorted_vals, indices = torch.sort(t)    # ascending
topv, topi = torch.topk(t, k=3)
uniq = torch.unique(t, sorted=True, return_counts=True)
args = torch.argsort(t, descending=True)
```

---

# 10) অ্যাডভান্সড গ্যাদার/স্ক্যাটার/ওয়ান-হট

```python
# one_hot
labels = torch.tensor([0,2,1])
onehot = torch.nn.functional.one_hot(labels, num_classes=3)  # [3,3], int64

# scatter: index অনুযায়ী মান বসান
src = torch.tensor([[10,20,30]])
idx = torch.tensor([[0,2,1]])
out = torch.zeros(1,3).scatter_(1, idx, src)  # -> [[10,30,20]]

# gather: index অনুযায়ী মান তুলুন
g = torch.gather(out, 1, idx)  # -> [[10,20,30]]
```

---

# 11) নর্মস/স্ট্যাবিলিটি/NaN-Inf হ্যান্ডলিং

```python
v = torch.tensor([1., -2., 3.])
l2 = torch.linalg.vector_norm(v)          # ||v||
finite = torch.isfinite(v)                # ~tensor([True, True, True])
cln = torch.nan_to_num(torch.tensor([float('nan'), float('inf'), -float('inf')]),
                       nan=0.0, posinf=1e9, neginf=-1e9)
```

---

# 12) বেটচিং/ব্যাচড অপারেশন ও `einsum`

```python
# Batched matmul: [B,M,K] @ [B,K,N] -> [B,M,N]
A = torch.randn(8, 16, 32)
B = torch.randn(8, 32, 10)
C = torch.matmul(A, B)

# einsum: ভেরসাটাইল টেন্সর কন্ট্রাকশন
x = torch.randn(32, 128)
w = torch.randn(128, 64)
y = torch.einsum('bi,ij->bj', x, w)
```

---

# 13) NumPy ইন্টারঅপ

```python
t = torch.randn(2,3)
arr = t.numpy()        # CPU টেনসর হলে ভিউ শেয়ার করে
t2 = torch.from_numpy(arr)
arr[0,0] = 999         # t-তেও প্রতিফলিত হবে (শেয়ার্ড!)
```

---

# 14) সিরিয়ালাইজেশন (টেনসর সেভ/লোড)

```python
x = torch.randn(4,4)
torch.save(x, "tensor.pt")
y = torch.load("tensor.pt")  # একই dtype/device বজায় রাখতে পারে
```

---

# 15) পারফরম্যান্স টিপস

* **GPU ব্যবহার**: `.to("cuda")`/`.cuda()` দিয়ে টেনসর/মডেল GPU-তে নিন।
* **পিনড মেমরি**: DataLoader-এ `pin_memory=True` CPU→GPU কপি দ্রুত করে।
* **অটোগ্র্যাড কন্ট্রোল**: ইভালুয়েশনে `with torch.no_grad():` ব্লক ব্যবহার করুন।
* **মেমরি**: বড় টেনসরের কপি এড়াতে `view/reshape/permute` (ভিউ) বোঝেন; প্রয়োজনে `.clone()` নিন।
* **মিক্সড প্রিসিশন**: ট্রেনিংয়ে `torch.cuda.amp.autocast()` ও `GradScaler` ব্যবহার করলে স্পিড/মেমরি সাশ্রয়।

---

# 16) ছোট্ট এন্ড-টু-এন্ড উদাহরণ (গ্র্যাডিয়েন্টসহ)

```python
import torch
torch.manual_seed(0)

# Fake data: y = xW + b + noise
X = torch.randn(64, 10)
true_W = torch.randn(10, 1)
true_b = torch.randn(1)
Y = X @ true_W + true_b + 0.1*torch.randn(64,1)

# Params
W = torch.randn(10,1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

lr = 0.1
for step in range(200):
    pred = X @ W + b             # [64,1]
    loss = torch.mean((pred - Y)**2)

    loss.backward()
    with torch.no_grad():
        W -= lr * W.grad
        b -= lr * b.grad
        W.grad.zero_()
        b.grad.zero_()

# শেখা প্যারামিটার দেখুন
print("W head:", W[:3].ravel())
print("b:", b.item())
```

---

# 17) প্রায়-সবচেয়ে-ব্যবহৃত অপারেশনগুলোর চিটশিট

* **Creation**: `tensor`, `zeros`, `ones`, `empty`, `arange`, `linspace`, `rand`, `randn`, `eye`, `from_numpy`
* **Shape**: `shape`, `reshape/view`, `permute/transpose`, `unsqueeze/squeeze`, `contiguous`, `cat/stack`, `repeat/expand`
* **Indexing**: standard indexing, boolean mask, `where`, `gather/scatter`
* **Math**: `+ - * /`, `pow`, `exp/log`, `sin/cos`, `clamp`, `round/floor/ceil`
* **Reduction**: `sum/mean/std/var/min/max/arg*`, `logsumexp`
* **Linalg**: `matmul/@`, `mm`, `svd`, `qr`, `eig`, `det`, `inv`, `solve`, `norm`
* **Random**: `manual_seed`, `rand`, `randn`, `distributions.*`
* **Autograd**: `requires_grad`, `.backward()`, `.grad`, `no_grad`, `detach`
* **Utils**: `isfinite`, `nan_to_num`, `topk/sort/argsort`, `unique`, `one_hot`
* **Interop**: `.numpy()`, `from_numpy()`
* **I/O**: `torch.save`, `torch.load`
* **Device**: `.to("cuda"/"cpu")`, `.cuda()`, `.cpu()`

---

