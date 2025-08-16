
---

# ✅ Full, runnable PyTorch code (with step-by-step comments)

```python
# =========================
# 0) Imports & setup
# =========================
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Results directory
os.makedirs("gan_samples", exist_ok=True)

# =========================
# 1) Hyperparameters
# =========================
img_size     = 28          # MNIST 28x28
channels     = 1           # grayscale
z_dim        = 100         # noise/latent size
g_feat       = 128         # generator base width
d_feat       = 128         # discriminator base width
batch_size   = 128
lr           = 2e-4
betas        = (0.5, 0.999)
num_epochs   = 20

# =========================
# 2) Dataset & DataLoader
# =========================
# Normalize to [-1, 1] so that tanh() works nicely in Generator output
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

trainset  = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
loader    = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

# =========================
# 3) Model definitions
#    DCGAN-style architectures that work well on 28x28
# =========================

class Generator(nn.Module):
    """
    z (B, z_dim) -> (B, 1, 28, 28)
    Use transposed convolutions to upsample from 1x1 to 7x7 to 14x14 to 28x28
    """
    def __init__(self, z_dim=100, g_feat=128, out_ch=1):
        super().__init__()
        self.net = nn.Sequential(
            # input: (B, z_dim, 1, 1)
            nn.ConvTranspose2d(z_dim, g_feat*4, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(g_feat*4),
            nn.ReLU(True),

            # (B, g_feat*4, 7, 7) -> (14x14)
            nn.ConvTranspose2d(g_feat*4, g_feat*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(g_feat*2),
            nn.ReLU(True),

            # (B, g_feat*2, 14, 14) -> (28x28)
            nn.ConvTranspose2d(g_feat*2, g_feat, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(g_feat),
            nn.ReLU(True),

            # to 1 channel, tanh to [-1, 1]
            nn.Conv2d(g_feat, out_ch, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        # reshape z: (B, z_dim) -> (B, z_dim, 1, 1)
        z = z.view(z.size(0), z.size(1), 1, 1)
        return self.net(z)

class Discriminator(nn.Module):
    """
    x (B, 1, 28, 28) -> prob real/fake
    """
    def __init__(self, in_ch=1, d_feat=128):
        super().__init__()
        self.net = nn.Sequential(
            # (28x28) -> (14x14)
            nn.Conv2d(in_ch, d_feat, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # (14x14) -> (7x7)
            nn.Conv2d(d_feat, d_feat*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d_feat*2),
            nn.LeakyReLU(0.2, inplace=True),

            # (7x7) -> (7x7) conv + flatten
            nn.Conv2d(d_feat*2, d_feat*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(d_feat*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(d_feat*4*7*7, 1),
            nn.Sigmoid(),  # output in [0, 1] -> prob real
        )

    def forward(self, x):
        return self.net(x)

# =========================
# 4) Weights init (DCGAN recommendation)
# =========================
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d,)):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)

G = Generator(z_dim=z_dim, g_feat=g_feat, out_ch=channels).to(device)
D = Discriminator(in_ch=channels, d_feat=d_feat).to(device)
G.apply(weights_init)
D.apply(weights_init)

# =========================
# 5) Optimizers & loss
# =========================
opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=betas)
opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=betas)
criterion = nn.BCELoss()

# fixed noise for monitoring G's progress
fixed_z = torch.randn(64, z_dim, device=device)

# =========================
# 6) Training loop (two steps per batch)
# =========================
step = 0
for epoch in range(num_epochs):
    for real, _ in loader:
        real = real.to(device)
        b = real.size(0)

        # 6.1) -------- Train Discriminator --------
        # Goal: maximize log D(real) + log(1 - D(fake))
        opt_D.zero_grad()

        # Real batch -> label 1
        real_labels = torch.ones(b, 1, device=device)
        pred_real   = D(real)
        loss_real   = criterion(pred_real, real_labels)

        # Fake batch -> label 0
        z = torch.randn(b, z_dim, device=device)
        fake = G(z).detach()            # detach so G is not updated here
        fake_labels = torch.zeros(b, 1, device=device)
        pred_fake   = D(fake)
        loss_fake   = criterion(pred_fake, fake_labels)

        loss_D = loss_real + loss_fake
        loss_D.backward()
        opt_D.step()

        # 6.2) -------- Train Generator --------
        # Goal: minimize log(1 - D(G(z)))  <=> maximize log D(G(z))
        # Common trick: use real labels (1) to push D(G(z)) -> 1
        opt_G.zero_grad()
        z = torch.randn(b, z_dim, device=device)
        gen_imgs   = G(z)
        pred_fake2 = D(gen_imgs)
        target     = torch.ones(b, 1, device=device)  # want D to think fakes are real
        loss_G     = criterion(pred_fake2, target)
        loss_G.backward()
        opt_G.step()

        # 6.3) Logging & sample saving
        if step % 200 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Step [{step}] "
                  f"| D: {loss_D.item():.4f} | G: {loss_G.item():.4f} "
                  f"| D(x): {pred_real.mean().item():.3f} | D(G(z)): {pred_fake.mean().item():.3f}")

            with torch.no_grad():
                samples = G(fixed_z).cpu()
                # unnormalize from [-1,1] -> [0,1] for grid saving
                samples = (samples + 1) / 2
                grid = make_grid(samples, nrow=8, padding=2)
                save_image(grid, f"gan_samples/epoch{epoch:03d}_step{step:06d}.png")

        step += 1

print("Training complete! Sample images saved to ./gan_samples")
```

---

## 🔍 কী হচ্ছে—সংক্ষেপে ধাপভিত্তিক

1. **ডেটা**: MNIST ইমেজগুলোকে `[-1, 1]`–এ normalize করি (Generator-এর `tanh()` আউটপুটের সাথে ম্যাচ করার জন্য)।
2. **Generator (G)**: Noise z → `ConvTranspose2d` দিয়ে ধাপে ধাপে আপস্যাম্পল করে 28×28 ইমেজ বানায়; শেষে `tanh()`।
3. **Discriminator (D)**: কনভ লেয়ারে ডাউনস্যাম্পল → `sigmoid()` দিয়ে real probability।
4. **Loss**: Binary cross-entropy।

   * **D step**: Real→1, Fake→0; `loss_D = CE(D(real),1) + CE(D(fake),0)`
   * **G step**: `CE(D(G(z)), 1)` (Generator চায় D যেন fake-কে real ভাবে)।
5. **Optimizers**: Adam (`betas=(0.5,0.999)`) DCGAN-স্ট্যান্ডার্ড।
6. **Monitoring**: fixed noise দিয়ে প্রতি কিছু স্টেপে grid ইমেজ সেভ—উন্নতি দেখা যায়।

---

## 🧪 টিপস

* যদি D খুব শক্তিশালী হয়ে যায় → G শেখে না; G/D loss balance দেখো।
* লার্নিং রেট, betas, batch\_size টিউন করলে training স্থিতিশীল হয়।
* CIFAR-10/celeba নিলে arch/channels সামান্য বড় করো।

---

