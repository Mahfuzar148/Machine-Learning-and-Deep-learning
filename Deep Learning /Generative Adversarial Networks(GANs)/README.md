


---

# 📘 Documentation: Generative Adversarial Networks (GANs)


---

![GANs Diagram](https://github.com/Mahfuzar148/Machine-Learning-and-Deep-learning/blob/main/Deep%20Learning%20/Generative%20Adversarial%20Networks\(GANs\)/GANs%20diagram.png?raw=true)

---

---

## 1. Introduction

Generative Adversarial Networks (GANs) হলো একটি **deep learning framework** যা প্রথম Ian Goodfellow 2014 সালে প্রস্তাব করেন।
এটি ব্যবহার করা হয় **new, synthetic data instances** তৈরি করার জন্য যা আসল ডেটার মতো দেখায়।

GAN মূলত **game-theoretic** সেটআপে কাজ করে—দুইটি neural network (Generator এবং Discriminator) একে অপরের বিপক্ষে লড়াই করে:

* Generator (G) → নকল data বানায়।
* Discriminator (D) → বলে দেয় data আসল না নকল।

ফলাফল: Generator এমন data বানাতে শেখে যেটা আসল data থেকে আলাদা করা কঠিন।

---

## 2. Components of a GAN

### (a) Generator (G)

* Input: Random noise vector $z$ (latent space থেকে)
* Output: Fake data sample (image, audio ইত্যাদি)
* Goal: Discriminator কে ভুলাতে realistic data তৈরি করা।
* সাধারণত **deconvolutional / transpose convolution layers** ব্যবহার করে।

### (b) Discriminator (D)

* Input: Real data (dataset থেকে) অথবা Fake data (Generator থেকে)
* Output: Probability (0–1), কতটা আসল মনে হচ্ছে।
* Goal: আসল এবং নকল data আলাদা করা।
* সাধারণত **convolutional network**।

---

## 3. Training Process (Adversarial Game)

GAN ট্রেনিং হয় দুই ধাপে (বারবার loop করে):

1. **Train Discriminator (D):**

   * আসল data (label=1) এবং নকল data (label=0) নিয়ে train করাও।
   * Discriminator-এর কাজ: আসল/নকল ঠিকভাবে classify করা।

2. **Train Generator (G):**

   * Noise থেকে fake data বানায়।
   * Discriminator-কে ভুলাতে চায়।
   * Generator loss = কতটা Discriminator ভুল হয়েছে।

3. **Repeat Adversarial Training:**

   * এক পর্যায়ে G এত ভালো fake বানায় যে D বিভ্রান্ত হয়।
   * Equilibrium-এ D real/fake ধরতে পারবে না (accuracy \~50%)।

---

## 4. Mathematical Formulation

GAN objective হলো Minimax optimization problem:

$$
\min_G \max_D V(D,G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]
$$

* $D(x)$: Probability যে x আসল।
* $G(z)$: Generator output from noise $z$।
* Discriminator maximize করতে চায় (real ভালোভাবে চেনা, fake ধরে ফেলা)।
* Generator minimize করতে চায় (D যেন বিভ্রান্ত হয়)।

---

## 5. Example (Wine analogy from diagram)

* **Generator = Forger** → নকল ওয়াইন বানায়।
* **Discriminator = Shop owner** → বোতল চেখে দেখে আসল না নকল।
* ট্রেনিং চলতে চলতে Forger শেখে আরও ভালো ওয়াইন বানাতে, Shop owner শেখে ভালোভাবে ধরতে।
* শেষ পর্যন্ত Forger এমন ওয়াইন বানায় যা আসল/নকল আলাদা করা যায় না।

---

## 6. Practical Example (MNIST Digit Generation)

* Dataset: আসল হাতে লেখা digit (0–9)।
* Generator: Noise থেকে নতুন digit ইমেজ বানায়।
* Discriminator: আসল digit বনাম নকল digit classify করে।
* ফলাফল: Generator আসল digit-এর মতো নকল digit বানাতে শেখে।

---

## 7. Applications of GANs

1. **Image Generation**

   * Human faces (StyleGAN)
   * Objects, scenes

2. **Image-to-Image Translation**

   * Day ↔ Night (CycleGAN)
   * Sketch → Photo

3. **Super-Resolution**

   * Low-res image থেকে high-res বানানো (SRGAN)

4. **Deepfake Creation**

   * মুখ/কণ্ঠস্বর পরিবর্তন

5. **Data Augmentation**

   * Medical images, rare class samples synthetic বানানো

6. **Art and Creativity**

   * AI-generated art, music, design

---

## 8. Advantages

* Highly realistic synthetic data generate করতে পারে।
* Works in unsupervised setting (label না লাগলেও চলে)।
* Flexible (image, video, audio, text — সব ধরনের data)।

---

## 9. Limitations

* Training খুব sensitive (mode collapse, instability)।
* Generator vs Discriminator balance রাখা কঠিন।
* Computationally heavy।
* Ethical issues (deepfakes, misinformation)।

---

## 10. Summary

GAN = **Generator + Discriminator adversarial training**

* Generator → Fake বানায়
* Discriminator → Fake ধরতে শেখে
* Adversarial game-এর ফলে Generator realistic sample বানাতে শেখে।
* Used in: image synthesis, deepfake, data augmentation, art, super-resolution ইত্যাদি।

---



---

## Step 0: Setup (imports, device, folders)

**Process:** প্রয়োজনীয় লাইব্রেরি ইমপোর্ট, ডিভাইস সিলেক্ট, আউটপুট ফোল্ডার তৈরি।

```python
# Step 0 — Setup
import os, torch, torch.nn as nn, torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("samples", exist_ok=True)
```

---

## Step 1: Hyperparameters

**Process:** ইমেজ সাইজ/চ্যানেল, noise dim, লার্নিং রেট, ইপোক, ব্যাচ সাইজ ইত্যাদি সেট করা।

```python
# Step 1 — Hyperparameters
img_size   = 28       # dataset অনুযায়ী বদলাও (e.g., 64 for CelebA)
channels   = 1        # grayscale=1, RGB=3
z_dim      = 100
g_width    = 128
d_width    = 128
batch_size = 128
lr         = 2e-4
betas      = (0.5, 0.999)
epochs     = 20
```

---

## Step 2: Data pipeline

**Process:** ডেটাসেট লোড, টেন্সরে কনভার্ট, Normalize → `[-1, 1]` (Generator-এর `tanh` আউটপুটের সাথে মেলে)।

```python
# Step 2 — Data
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*channels, [0.5]*channels),
])
trainset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
loader   = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
```

---

## Step 3: Define Generator

**Process:** Noise z → আপস্যাম্পল করে ইমেজ বানায়; শেষে `tanh()` দিয়ে আউটপুট `[-1,1]`।

```python
# Step 3 — Generator
class Generator(nn.Module):
    def __init__(self, z_dim=100, width=128, out_ch=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, width*4, 7, 1, 0, bias=False),  # 1x1 -> 7x7
            nn.BatchNorm2d(width*4), nn.ReLU(True),

            nn.ConvTranspose2d(width*4, width*2, 4, 2, 1, bias=False), # 7->14
            nn.BatchNorm2d(width*2), nn.ReLU(True),

            nn.ConvTranspose2d(width*2, width, 4, 2, 1, bias=False),   # 14->28
            nn.BatchNorm2d(width), nn.ReLU(True),

            nn.Conv2d(width, out_ch, 3, 1, 1),
            nn.Tanh(),
        )
    def forward(self, z):
        return self.net(z.view(z.size(0), z.size(1), 1, 1))
```

---

## Step 4: Define Discriminator

**Process:** ইমেজ → ডাউনস্যাম্পল → সিগময়েড প্রোবাবিলিটি (real/fake)।

```python
# Step 4 — Discriminator
class Discriminator(nn.Module):
    def __init__(self, in_ch=1, width=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, width, 4, 2, 1),         # 28->14
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(width, width*2, 4, 2, 1, bias=False),  # 14->7
            nn.BatchNorm2d(width*2),
            nn.LeakyReLU(0.2, True),

            nn.Flatten(),
            nn.Linear(width*2*7*7, 1),
            nn.Sigmoid(),
        )
    def forward(self, x): return self.net(x)
```

---

## Step 5: Init models & weights

**Process:** DCGAN-স্টাইল ইনিশিয়ালাইজেশন (স্টেবল ট্রেনিং) + ডিভাইসে পাঠানো।

```python
# Step 5 — Init
def dcgan_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if getattr(m, "bias", None) is not None: nn.init.zeros_(m.bias)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02); nn.init.zeros_(m.bias)

G = Generator(z_dim, g_width, channels).to(device).apply(dcgan_init)
D = Discriminator(channels, d_width).to(device).apply(dcgan_init)
```

---

## Step 6: Loss, Optimizers, Fixed noise

**Process:** BCE loss (real/fake), Adam অপ্টিমাইজার, ফিক্সড noise → প্রগ্রেস দেখতে স্যাম্পল জেনারেট।

```python
# Step 6 — Loss & Optim
criterion = nn.BCELoss()
opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=betas)
opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=betas)
fixed_z = torch.randn(64, z_dim, device=device)
```

---

## Step 7: Discriminator step (per batch)

**Process:** D-কে real=1, fake=0 দিয়ে ট্রেন; fake তৈরিতে G-এর গ্র্যাডিয়েন্ট বন্ধ (detach)।

```python
# Step 7 — D-step (inside training loop)
def train_discriminator(real_batch):
    b = real_batch.size(0)
    opt_D.zero_grad()

    # Real
    pred_real = D(real_batch)
    loss_real = criterion(pred_real, torch.ones(b, 1, device=device))

    # Fake (stop grad to G)
    z    = torch.randn(b, z_dim, device=device)
    fake = G(z).detach()
    pred_fake = D(fake)
    loss_fake = criterion(pred_fake, torch.zeros(b, 1, device=device))

    loss_D = loss_real + loss_fake
    loss_D.backward()
    opt_D.step()
    return loss_D.item(), pred_real.mean().item(), pred_fake.mean().item()
```

---

## Step 8: Generator step (per batch)

**Process:** G চায় D(fake) → 1 হোক, তাই target=1 দিয়ে BCE; G-র গ্র্যাডিয়েন্ট আপডেট।

```python
# Step 8 — G-step (inside training loop)
def train_generator(b):
    opt_G.zero_grad()
    z = torch.randn(b, z_dim, device=device)
    gen = G(z)
    pred = D(gen)
    loss_G = criterion(pred, torch.ones(b, 1, device=device))  # fool D
    loss_G.backward()
    opt_G.step()
    return loss_G.item()
```

---

## Step 9: Training loop + logging/saving

**Process:** প্রতি ব্যাচে D-step → G-step; নির্দিষ্ট স্টেপে স্যাম্পল সেভ।

```python
# Step 9 — Full training loop
step = 0
for epoch in range(epochs):
    for real, _ in loader:
        real = real.to(device)
        b = real.size(0)

        d_loss, d_real, d_fake = train_discriminator(real)
        g_loss = train_generator(b)

        if step % 200 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Step {step} | "
                  f"D: {d_loss:.4f} (Dx={d_real:.3f}, Dg={d_fake:.3f}) | G: {g_loss:.4f}")
            with torch.no_grad():
                samp = (G(fixed_z).cpu() + 1) / 2  # [-1,1] -> [0,1]
                grid = make_grid(samp, nrow=8, padding=2)
                save_image(grid, f"samples/ep{epoch:03d}_st{step:06d}.png")
        step += 1

print("Done! Check ./samples for generated grids.")
```

---

## Step 10 (optional): Inference / sample-only

**Process:** ট্রেনিং ছাড়াই জেনারেটর থেকে নতুন ইমেজ।

```python
# Step 10 — Inference (after training)
with torch.no_grad():
    z = torch.randn(64, z_dim, device=device)
    imgs = (G(z).cpu() + 1) / 2
    save_image(make_grid(imgs, nrow=8), "samples/final.png")
```

---

### ব্যবহার টিপস

* **RGB ডেটা** হলে `channels=3`, Normalize-এ `[0.5]*3` দিন, এবং Discriminator/Generator conv width একটু বাড়ান।
* **বড় ইমেজ (64×64)** নিলে TransposeConv/Conv ব্লকগুলো এক লেভেল বেশি দিন (7→8→16→32→64 টাইপ)।
* ট্রেনিং অনস্টেবল হলে: `lr` কমাও, `betas=(0.0,0.9)` ট্রাই করো, label smoothing বা one-sided label noise (real=0.9) ইউজ করতে পারো।


---

# 🔹 GAN Training Steps (Summary)

### **Step 0 — Setup**

* লাইব্রেরি ইমপোর্ট (PyTorch, torchvision)
* ডিভাইস সিলেক্ট (CPU/GPU)
* আউটপুট ফোল্ডার বানানো

---

### **Step 1 — Hyperparameters**

* ইমেজ সাইজ, চ্যানেল সংখ্যা (1=Gray, 3=RGB)
* Latent/noise dimension (z\_dim)
* লার্নিং রেট, ব্যাচ সাইজ, ইপোক সংখ্যা

---

### **Step 2 — Data Pipeline**

* Dataset লোড (যেমন MNIST)
* টেন্সরে কনভার্ট
* Normalize `[-1,1]` (Generator-এর `tanh` আউটপুটের সাথে মেলাতে)

---

### **Step 3 — Generator (G)**

* Noise z → আপস্যাম্পল (TransposeConv/Linear+Reshape)
* আউটপুট: ইমেজ
* Activation: **tanh()**

---

### **Step 4 — Discriminator (D)**

* ইমেজ → Convolution দিয়ে ডাউনস্যাম্পল
* আউটপুট: Probability (real/fake)
* Activation: **sigmoid()**

---

### **Step 5 — Initialization**

* DCGAN স্টাইল weight init (Normal mean=0, std=0.02)
* Models ডিভাইসে পাঠানো

---

### **Step 6 — Loss & Optimizers**

* BCE Loss (binary cross-entropy)
* Optimizer: Adam (betas = (0.5,0.999))
* Fixed noise → স্যাম্পল ভিজ্যুয়ালাইজেশনের জন্য

---

### **Step 7 — Train Discriminator (per batch)**

* Real data → label = 1
* Fake data (G থেকে) → label = 0
* Loss\_D = CE(D(real),1) + CE(D(fake),0)
* Backprop + update D

---

### **Step 8 — Train Generator (per batch)**

* Fake data বানাও
* Target = 1 (Generator চাই D কে বোকা বানাক)
* Loss\_G = CE(D(G(z)), 1)
* Backprop + update G

---

### **Step 9 — Training Loop**

* প্রতিটি batch এ: D-step → G-step
* Periodically → Loss print + Fixed noise থেকে sample save

---

### **Step 10 — Inference (after training)**

* Random noise → G
* Output ইমেজ save/show

---

⚡ **এক লাইনে:**
Noise → Generator → Fake image → Discriminator (with Real+Fake) → Loss update (D, then G) → Repeat until Generator realistic data বানাতে শেখে।

---


