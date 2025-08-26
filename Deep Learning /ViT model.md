

## ViT কী করে?

ছবিটাকে ছোট ছোট **প্যাচ** (যেমন 16×16 বা 32×32) বানায়, প্রতিটি প্যাচকে **টোকেন** ধরে। তারপর NLP-র ট্রান্সফরমারের মতো **Self-Attention** দিয়ে সব প্যাচ একে-অপরের সাথে তুলনা করে বোঝে—কার সাথে কার সম্পর্ক বেশি। শেষে পুরো ছবির একটা **ফিচার সারাংশ** বের করে ক্লাসিফাই করে।

## ধাপে ধাপে কাজের ফ্লো

1. **Patch Split**: 224×224 ছবি → 16×16 প্যাচে ভাঙি → মোট 14×14 = **196 প্যাচ**।
2. **Patch Embedding**: প্রতিটি প্যাচকে ফ্ল্যাট করে **লিনিয়ার প্রজেকশনে D-ডাইম** (যেমন 768) ভেক্টর বানাই।
3. **Positional Embedding**: কোন প্যাচ ছবির কোন জায়গায় ছিল—এই **পজিশন তথ্য যোগ** করি (লার্নেবল)।
4. **Transformer Encoder (L বার)**:

   * **Multi-Head Self-Attention (MSA)**: প্রতিটি প্যাচ অন্যসব প্যাচকে “খেয়াল” করে।
   * **FFN (MLP)**: ফিচার আরও রিচ হয়।
   * **Residual + LayerNorm**: ট্রেনিং স্টেবল থাকে।
5. **CLS Token + MLP Head**: শুরুতে একটা **\[CLS]** টোকেন যোগ করা হয়। সব লেয়ারের শেষে **CLS ভেক্টর** নিয়ে ছোট **MLP** দিয়ে **লেবেল প্রেডিক্ট** করি (softmax)।

## CNN vs ViT (এক লাইনে)

* **CNN**: ছোট উইন্ডোতে **লোকাল** প্যাটার্ন (edge/texture) ধরতে ভালো।
* **ViT**: Self-Attention দিয়ে **গ্লোবাল কনটেক্সট** ধরতে দারুণ—পুরো ছবির অংশগুলোর সম্পর্ক একসাথে শেখে।

## শক্তি (কেন ভালো)

* **গ্লোবাল বোঝাপড়া**: দূরের প্যাচের সম্পর্কও শিখে।
* **স্কেলেবিলিটি**: বড় ডেটা/মডেলে অসাধারণ ফল।
* **ফ্লেক্সিবল**: ক্লাসিফিকেশন ছাড়াও ডিটেকশন, সেগমেন্টেশন, ভিডিও—সবখানেই মানিয়ে নেয়।

## চ্যালেঞ্জ

* **ডেটা-হাংরি**: স্ক্র্যাচ থেকে ট্রেন করতে অনেক ডেটা লাগে।
* **কম্পিউট কস্ট**: Self-Attention-এর কস্ট **O(N²)** (প্যাচ বেশি হলে কস্ট বাড়ে)।
* সমাধান: সাধারণত **pretrained weights** নিয়ে **fine-tune** করা হয় (ImageNet/CLIP/MAE ইত্যাদি)।

## তোমার মডেলে ViT বসালে কী হয়?

* ViT **backbone** হিসেবে **ছবি → ফিচার** বানায় (যেমন 768-D)।
* এরপর তুমি **DomainAlignment → FeatureBottleneck → Classifier** চালাও।
* পাশেই **GRL + DomainDiscriminator** শাখা থাকে—এটা ফিচারকে **ডোমেইন-ইনভারিয়্যান্ট** হতে বাধ্য করে।
* **Pretrained ViT** নিলে ছোট ডেটাতেও দ্রুত/ভালো ফল মেলে; শেষে কিছু ব্লক আনফ্রিজ করে ফাইন-টিউন করলেই হয়।

## এক লাইনের সারসংক্ষেপ

**ViT = ছবি → প্যাচ → টোকেন → (পজিশন যোগ) → ট্রান্সফরমার এনকোডার → CLS-হেড → প্রেডিকশন।**
গ্লোবাল সম্পর্ক ধরতে পারার কারণে অনেক টাস্কে ViT এখন টপ-পারফর্মার।




```python
# =========================
# Minimal ViT in PyTorch
# =========================
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Patch Embedding: ইমেজকে P×P প্যাচে ভেঙে D-ডাইম টোকেন বানায় ----
class PatchEmbed(nn.Module):
    """
    3×H×W ইমেজ -> N×D টোকেন
    N = (H/P) * (W/P);  D = embed_dim
    Conv2d(kernel=stride=patch) দিয়ে patchify + linear projection একসাথে করা হয়
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        assert img_size % patch_size == 0, "img_size অবশ্যই patch_size দিয়ে বিভাজ্য হতে হবে"
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)

        # Conv: 3×H×W -> D×(H/P)×(W/P)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):  # x: [B, 3, H, W]
        x = self.proj(x)              # [B, D, H/P, W/P]
        x = x.flatten(2)              # [B, D, N]
        x = x.transpose(1, 2)         # [B, N, D]
        return x


# ---- MLP/FFN ব্লক: টোকেন ফিচার প্রসেসিং ----
class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):  # [B, N, D]
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ---- Multi-Head Self-Attention: প্রতিটি টোকেন অন্যসব টোকেনকে "খেয়াল" করে ----
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=12, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, "embed dim অবশ্যই num_heads দিয়ে বিভাজ্য হতে হবে"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5  # 1/sqrt(d_k)

        # একবারে Q,K,V বের করা: [B, N, 3*D]
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):  # [B, N, D]
        B, N, D = x.shape
        qkv = self.qkv(x)                         # [B, N, 3D]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)               # প্রতিটির শেপ: [B, N, H, Hd]

        # (B, H, N, Hd) এ ট্রান্সপোজ: attention ম্যাটমাল সহজ করতে
        q = q.transpose(1, 2)                     # [B, H, N, Hd]
        k = k.transpose(1, 2)                     # [B, H, N, Hd]
        v = v.transpose(1, 2)                     # [B, H, N, Hd]

        # স্কেলড ডট-প্রোডাক্ট অ্যাটেনশন
        attn = (q @ k.transpose(-2, -1)) * self.scale   # [B, H, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v                               # [B, H, N, Hd]
        out = out.transpose(1, 2).reshape(B, N, D)   # [B, N, D]
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# ---- Transformer Encoder Block: LN -> MSA -> Residual -> LN -> MLP -> Residual ----
class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads=12, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop)

    def forward(self, x):  # [B, N, D]
        # Attention সাব-লেয়ার + Residual
        x = x + self.attn(self.norm1(x))
        # MLP সাব-লেয়ার + Residual
        x = x + self.mlp(self.norm2(x))
        return x


# ---- ViT: PatchEmbed -> [CLS]+PosEmbed -> EncoderBlocks × L -> Norm -> Head ----
class ViT(nn.Module):
    def __init__(
        self,
        img_size=224, patch_size=16, in_chans=3,
        num_classes=1000,
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
        drop_rate=0.0, attn_drop_rate=0.0
    ):
        super().__init__()

        # 1) প্যাচ টোকেন বানানো
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # 2) [CLS] টোকেন + Positional Embedding (লার্নেবল)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))         # [1,1,D]
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim))  # [1,N+1,D]
        self.pos_drop = nn.Dropout(drop_rate)

        # 3) Transformer Encoder ব্লকসমূহ
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate
            )
            for _ in range(depth)
        ])

        # 4) নর্ম + হেড
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # ওজন ইনিশিয়ালাইজেশন (সিম্পল)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_vit_weights)

    @staticmethod
    def _init_vit_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):  # x: [B, 3, H, W]
        B = x.size(0)

        # (a) ইমেজ -> টোকেন
        x = self.patch_embed(x)                 # [B, N, D]

        # (b) [CLS] টোকেন prepend
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat((cls, x), dim=1)          # [B, 1+N, D]

        # (c) পজিশনাল এমবেডিং যোগ + ড্রপআউট
        x = x + self.pos_embed                  # [B, 1+N, D]
        x = self.pos_drop(x)

        # (d) Encoder ব্লকগুলো
        for blk in self.blocks:
            x = blk(x)                          # [B, 1+N, D]

        # (e) ফাইনাল নর্ম + CLS টোকেন নেওয়া
        x = self.norm(x)                        # [B, 1+N, D]
        cls_feat = x[:, 0]                      # [B, D]  -> পুরো ছবির সারাংশ

        # (f) ক্লাসিফিকেশন হেড
        logits = self.head(cls_feat)            # [B, num_classes]
        return logits


# ------------- Quick test -------------
if __name__ == "__main__":
    model = ViT(
        img_size=224, patch_size=16, in_chans=3,
        num_classes=10, embed_dim=384, depth=8, num_heads=6, mlp_ratio=4.0,
        drop_rate=0.1, attn_drop_rate=0.0
    )
    x = torch.randn(4, 3, 224, 224)
    y = model(x)              # [4, 10]
    print("logits shape:", y.shape)
```

---

## কী হচ্ছে—সহজ ভাষায়

### 1) PatchEmbed

* 224×224 ইমেজকে 16×16 প্যাচে ভাগ করে (মোট 196 প্যাচ)।
* প্রতিটি প্যাচকে `Conv2d(kernel=stride=16)` দিয়ে **D-ডাইম টোকেন** বানায়।
* আউটপুট: `[B, N, D]` (যেখানে N=196, D=embed\_dim)

### 2) \[CLS] টোকেন + Positional Embedding

* \[CLS] = পুরো ছবির সারাংশ রাখার “বিশেষ” টোকেন (শুরুর দিকে যোগ হয়)।
* পজিশনাল এমবেডিং যোগ করে দেওয়া হয়, যাতে কোন টোকেন ছবির কোন জায়গা থেকে এসেছে সেটা বোঝা যায়।

### 3) Transformer Encoder ব্লক (বারবার)

প্রতিটি ব্লকে:

* **LayerNorm → Multi-Head Self-Attention → Residual Add**
* **LayerNorm → MLP(FFN) → Residual Add**
* Self-Attention: প্রতিটি টোকেন অন্য সব টোকেনকে “খেয়াল” করে গ্লোবাল কনটেক্সট শেখে।
* MLP: ফিচার আরও সমৃদ্ধ করে।

### 4) CLS টোকেন থেকে প্রেডিকশন

* সব ব্লকের পরে LayerNorm।
* `x[:, 0]` = CLS টোকেন (পুরো ছবির সারাংশ)।
* `Linear` হেডে পাঠিয়ে **logits** পাওয়া যায়।

---

## দ্রুত ব্যবহার (ট্রেনিং লুপের খুদে উদাহরণ)

```python
model = ViT(num_classes=2)           # বাইনারি টাস্ক
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

model.train()
imgs = torch.randn(16, 3, 224, 224)
labels = torch.randint(0, 2, (16,))

logits = model(imgs)
loss = criterion(logits, labels)
loss.backward()
opt.step(); opt.zero_grad()
```

---

## Pretrained নিতে চাইলে (timm ব্যবহার)

আপনি নিজের ViT লিখে বুঝলেন—এখন **pretrained** নিতে চাইলে:

```python
from timm import create_model

vit = create_model('vit_base_patch16_224', pretrained=True, num_classes=1000)
# ফিচার বের করতে চাইলে:
vit.reset_classifier(0)   # বা vit.head = nn.Identity()
feats = vit(torch.randn(2,3,224,224))  # -> [2, 768]
```

---

