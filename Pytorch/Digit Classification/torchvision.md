<img src="https://github.com/Mahfuzar148/Machine-Learning-and-Deep-learning/blob/main/Pytorch/Digit%20Classification/torchvision_mindmap_bigfont.png" alt="torchvision mindmap" width="800">

# 0) torchvision আসলে কী সমস্যার সমাধান করে?

কম্পিউটার ভিশনে ৪টা জিনিস বারবার লাগে:

1. **ডাটা লোডিং** (ইমেজ/ভিডিও পড়ে আনা)
2. **ট্রান্সফর্ম/অগমেন্টেশন** (রিসাইজ, ক্রপ, নরমালাইজ, র‍্যান্ডম ফ্লিপ…)
3. **রেডি-মেড মডেল** (ResNet, Faster R-CNN, DeepLab ইত্যাদি, প্রি-ট্রেইন্ড ওয়েটসহ)
4. **ভিশন অপারেশন** (NMS, ROI Align, বক্স ইউটিল, ড্র-ভিজুয়ালাইজ)

`torchvision` এই ৪টা স্তম্ভকে একটি প্যাকেজে দেয়—যাতে আপনি **প্রোটোটাইপ → ট্রেন → ইভ্যাল → ডেপলয়** দ্রুত করতে পারেন।

---

# 1) ডেটা পাইপলাইন—ভিতরটা কীভাবে কাজ করে (থিওরি)

## 1.1 ImageFolder & DataLoader (ফোল্ডারনির্ভর ক্লাসিফিকেশন)

* **ImageFolder** ধরে নেয়: `root/class_x/*.jpg`, `root/class_y/*.png`—ফোল্ডার নামই ক্লাস লেবেল।
* **Transforms** (Compose) ট্রেনিংয়ের সময় **অগমেন্টেশন** করে—এতে মডেল **invariance** শিখে (যেমন লেফট-রাইট ফ্লিপ হলেও ক্লাস বদলায় না) → **ওভারফিটিং কমে**।
* **Normalization** (mean/std) ইমেজ পিক্সেলকে একটি “স্ট্যান্ডার্ড স্কেলে” আনে—অপ্টিমাইজার দ্রুত কনভার্জ করে।
* **DataLoader** ব্যাচ বানায়, শাফল করে; `num_workers>0` হলে **মাল্টি-প্রসেস** ডেটা লোড হয় → GPU অলস বসে থাকে না।

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

IMG_SIZE = 224
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),        # spatial normalization
    transforms.RandomHorizontalFlip(p=0.5),         # invariance to mirroring
    transforms.ToTensor(),                          # [0,1], CHW, float32
    transforms.Normalize([0.485,0.456,0.406],       # ImageNet stats
                         [0.229,0.224,0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

train_ds = datasets.ImageFolder("data/train", transform=train_tf)
val_ds   = datasets.ImageFolder("data/val",   transform=val_tf)
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=4, pin_memory=True)
val_dl   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
```

### গুরুত্বপূর্ণ থিওরি

* **অর্ডার গুরুত্বপূর্ণ**: `ToTensor()`-এর আগে **জিওমেট্রিক** ট্রান্সফর্ম (Resize/Crop/Flip), তারপরে `Normalize()`।
* **PIL vs Tensor transforms**: পুরনো API PIL ইমেজে কাজ করে; `transforms.v2` টেনসর-বেজড (GPU পাইপলাইনও করা যায়) → স্পিড।
* **Interpolation**: `Resize` করলে পিক্সেল রিস্যাম্পলিং হয় (bilinear/bicubic)। বেশি ডাউনস্কেল করলে **ডিটেইল হারায়**—সাইজ বাছাই সাবধানে।
* **Determinism**: র‍্যান্ডম অগমেন্টেশন রিপ্রোডিউস করতে সিড ফিক্স করুন।

```python
import torch, random, numpy as np
torch.manual_seed(42); random.seed(42); np.random.seed(42)
torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
```

---

# 2) ট্রান্সফার লার্নিং—কেন কাজ করে? (থিওরি)

* বড় ডেটাসেটে (ImageNet-1k) ট্রেইনড মডেল **জেনেরিক ফিচার** (edge, texture, shape) শিখে।
* আপনার ছোট ডেটায় **ফাইন-টিউন** করলে দ্রুত কনভার্জ করে, কম ডেটাতেই ভালো অ্যাকিউরেসি।
* দুইটা মোড:

  * **Feature extractor**: বডি ফ্রিজ, শুধু শেষ লেয়ার ট্রেইন → দ্রুত/কম রিসোর্স।
  * **Full fine-tune**: সব আনফ্রিজ → বেশি অ্যাকিউরেসি সম্ভাব্য, কিন্তু সময় লাগে।

```python
import torch
from torchvision import models

NUM_CLASSES = 5
device = "cuda" if torch.cuda.is_available() else "cpu"

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
for p in model.parameters(): 
    p.requires_grad = False              # feature extractor mode
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)  # new head
model = model.to(device)
```

---

# 3) ট্রেনিং লুপ—লস/অপ্টিমাইজার/স্কেজিউলার (থিওরি)

* **Loss**: ক্লাসিফিকেশনে সাধারণত `CrossEntropyLoss` (softmax + NLL একসাথে)।
* **Optimizer**: `SGD(momentum)` বা `AdamW`—ছোট ডেটায় AdamW সুবিধাজনক।
* **LR Scheduler**: `StepLR`, `CosineAnnealingLR`, `OneCycleLR`—লার্নিং রেট ধীরে কমানো মডেলকে ভালো মিনিমায় আনতে সহায়তা করে।
* **AMP** (Automatic Mixed Precision): fp16 + fp32 মিক্স—ফাস্টার ও মেমরি সেভ।

```python
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    for x,y in train_dl:
        x,y = x.to(device), y.to(device)
        opt.zero_grad()
        with torch.cuda.amp.autocast(enabled=(device=="cuda")):
            logits = model(x); loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
    sched.step()
```

### ইভ্যালুয়েশন থিওরি

* **Accuracy** একা যথেষ্ট না (class imbalance হলে)।
* **Precision/Recall/F1** বুঝুন:

  * Precision = predicted positives এর মধ্যে কত সঠিক
  * Recall = সত্যিকারের positives এর কতটা ধরতে পেরেছি
* **Confusion Matrix** → কোথায় ভুল বেশি হচ্ছে বোঝা যায়।

---

# 4) অবজেক্ট ডিটেকশন—ধারণা ঠিকমতো

* **লক্ষ্য**: ছবিতে **কোথায়** (বক্স) এবং **কি** (লেবেল) দুটোই বের করা।
* **Anchors**: Faster R-CNN-এ বিভিন্ন স্কেল/অ্যাসপেক্ট রেশিওর প্রি-ডিফাইন্ড বক্স—এসব থেকে প্রপোজাল তৈরি।
* **IoU (Intersection over Union)**: বক্স মিল পরিমাপ।
* **NMS (Non-Max Suppression)**: ওভারল্যাপিং হাই-স্কোর বক্স থেকে একটিকে রাখা → ডুপ্লিকেট কমে।
* **COCO mAP**: IoU=.5 থেকে .95 পর্যন্ত গড় এভারেজ প্রিসিশন; ডিটেকশন ইভ্যালুয়েশনের স্ট্যান্ডার্ড।

### দ্রুত ইনফারেন্স (প্রি-ট্রেইন্ড)

```python
import torch, torchvision
from torchvision.io import read_image
from torchvision.transforms.functional import convert_image_dtype

device = "cuda" if torch.cuda.is_available() else "cpu"
det = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT").to(device).eval()

img = read_image("test.jpg")             # uint8 CHW
img = convert_image_dtype(img, torch.float32)  # to [0,1]
with torch.no_grad():
    out = det([img.to(device)])[0]       # dict: boxes[N,4], labels[N], scores[N]
```

### কাস্টম ট্রেনিং: টার্গেট ফরম্যাট (অত্যন্ত গুরুত্বপূর্ণ)

* আপনার `Dataset`-এর `__getitem__` → `image, target` রিটার্ন করবে
* `target` হলো dict:

  * `"boxes"`: FloatTensor \[N,4] (xmin,ymin,xmax,ymax)
  * `"labels"`: LongTensor \[N] (১..num\_classes)
  * (ঐচ্ছিক) `"masks"`: \[N,H,W] for instance seg
  * `"image_id"`, `"area"`, `"iscrowd"` (COCO স্টাইল—কখনো প্রয়োজন পড়ে)

```python
from torch.utils.data import Dataset
import torch
from PIL import Image

class DetectionDS(Dataset):
    def __init__(self, items, transforms=None):
        self.items = items   # [(path, annotations), ...]
        self.transforms = transforms
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        path, ann = self.items[idx]
        img = Image.open(path).convert("RGB")
        boxes = torch.tensor(ann["boxes"], dtype=torch.float32)   # [[x1,y1,x2,y2], ...]
        labels = torch.tensor(ann["labels"], dtype=torch.int64)   # [1..K]
        target = {"boxes": boxes, "labels": labels}
        if self.transforms: img = self.transforms(img)
        return img, target
```

**Collate function** লাগে, কারণ টার্গেট ডিক্ট ভ্যারিয়েবল-সাইজ:

```python
def collate_fn(batch):
    imgs, targets = zip(*batch)
    return list(imgs), list(targets)

from torch.utils.data import DataLoader
dl = DataLoader(det_ds, batch_size=4, shuffle=True, collate_fn=collate_fn)
```

---

# 5) সেগমেন্টেশন—পিক্সেল লেভেলে বোঝা

* **Semantic Segmentation**: প্রতিটি পিক্সেল কোন ক্লাস (রোড/স্কাই/পার্সন)।
* **Instance Segmentation**: আলাদা আলাদা অবজেক্ট আলাদা মাস্ক (Mask R-CNN)।
* **লস**: ক্লাসিক `CrossEntropy2d` (semantic), `BCEWithLogits`/ডাইস (বাইনারি/ইম্ব্যালান্সড)।
* **রিসাইজ/প্যাডিং**: আউটপুট সাইজ ডাউনস্যাম্পল/আপস্যাম্পলে বদলায়; ইন্টারপোলেশনের প্রভাব আছে।

```python
import torchvision, torch
from torchvision import transforms
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
seg = torchvision.models.segmentation.deeplabv3_resnet50(weights="DEFAULT").to(device).eval()

tf = transforms.Compose([
    transforms.Resize((520,520)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

img = Image.open("test.jpg").convert("RGB")
x = tf(img).unsqueeze(0).to(device)
with torch.no_grad():
    logits = seg(x)["out"]             # [B, C, H, W]
mask = logits.argmax(1)[0].cpu()       # ক্লাস-ইনডেক্স ম্যাপ
```

---

# 6) ভিশন অপস—NMS/IoU/ROI Align (কেন লাগে)

* **IoU**: ট্রেনিং/ইভ্যালুয়েশনে বক্স ম্যাচিংয়ের মূল মেট্রিক।
* **NMS**: একই অবজেক্টের উপর কয়েকটা বক্স এলে, স্কোর-সোর্ট করে ওভারল্যাপিংগুলো ড্রপ।
* **ROI Align**: ভ্যারিয়েবল সাইজের প্রপোজাল থেকে ফিক্সড ফিচার ম্যাপ বের করা (Faster/Mask R-CNN-এ দরকার)।

```python
import torch
from torchvision.ops import nms, box_iou

boxes = torch.tensor([[0,0,100,100],[10,10,110,110],[200,200,260,260]], dtype=torch.float32)
scores = torch.tensor([0.9, 0.8, 0.7])
keep = nms(boxes, scores, iou_threshold=0.5)  # indices to keep
iou_mat = box_iou(boxes, boxes)               # pairwise IoU
```

---

# 7) ইনপুট/আউটপুট—ডেটাটাইপ, রেঞ্জ, চ্যানেল (খুঁটিনাটি থিওরি)

* **read\_image** → `uint8`, shape `[C,H,W]`, রেঞ্জ `[0,255]`।
* **ToTensor()** বা `convert_image_dtype(..., torch.float32)` → রেঞ্জ `[0,1]`।
* **চ্যানেল অর্ডার**: PyTorch-এ **CHW**; অনেক ওপেনসিভি কোড **HWC** এবং **BGR**—কনভার্সন ভুল হলে রঙ অদ্ভুত দেখাবে।
* **Normalization** সবসময় float টেন্সরে করুন।

```python
from torchvision.io import read_image
from torchvision.transforms.functional import convert_image_dtype

img = read_image("x.jpg")                  # uint8 [C,H,W], 0..255
img = convert_image_dtype(img, torch.float32)  # 0..1
```

---

# 8) ভিজুয়ালাইজেশন—ডিবাগে লাইফসেভার

* **উদ্দেশ্য**: অগমেন্টেশন ঠিক চলছে? ডিটেকশনের বক্স শিফট হচ্ছে না তো?
* **Tools**: `draw_bounding_boxes`, `draw_segmentation_masks`, `make_grid`, `save_image`।

```python
import torch
from torchvision.utils import draw_bounding_boxes, save_image
imgs, _ = next(iter(train_dl))
img0 = (imgs[0]*255).to(torch.uint8)
drawn = draw_bounding_boxes(img0, boxes=torch.tensor([[20,20,120,100]]), labels=["demo"])
save_image(drawn.float()/255, "debug.png")
```

---

# 9) ক্লাস ইম্ব্যালান্স/রেগুলারাইজেশন—প্র্যাক্টিক্যাল থিওরি

* **ইম্ব্যালান্স**: অনেক ক্লাসে স্যাম্পল কম →

  * `WeightedRandomSampler`, `class weights` সহ `CrossEntropyLoss(weight=...)`, অথবা **focal loss**।
* **রেগুলারাইজেশন**: `weight_decay`, **data augmentation**, **dropout** (কিছু আর্কিতে), **early stopping**।
* **Early overfitting symptom**: ট্রেন লস ↓, ভ্যাল লস ↑ → অগমেন্টেশন বাড়ান/মডেল ছোট করুন/লআর কমান।

---

# 10) মডেল সিলেকশন—কবে কোনটা?

* **Mobile/Edge**: MobileNetV3, EfficientNet-Lite (ছোট/দ্রুত)।
* **General classification**: ResNet50/ConvNeXt-T/S, EfficientNet-V2-S।
* **Detection**: Faster R-CNN (অ্যাকিউরেট), RetinaNet (ওয়ান-স্টেজ, দ্রুত), SSDlite (লাইটওয়েট)।
* **Segmentation**: DeepLabV3 (সাধারণ), FCN (সিম্পল), LR-ASPP + MobileNet (লাইট)।

---

# 11) “স্মল কিন্তু ফুলি-ওয়ার্কিং” ক্লাসিফিকেশন স্ক্রিপ্ট

> **ফিচার-এক্সট্রাক্টর** মোডে দ্রুত চলবে। চাইলে `unfreeze_backbone=True` করলে ফুল ফাইন-টিউন হবে।

```python
# train_cls.py
import torch, argparse
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

def main(data_root="data", num_classes=2, batch_size=32, epochs=5, lr=1e-3, unfreeze_backbone=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tf_train = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    tf_val = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    ds_tr = datasets.ImageFolder(f"{data_root}/train", transform=tf_train)
    ds_va = datasets.ImageFolder(f"{data_root}/val",   transform=tf_val)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = models.efficientnet_v2_s(weights=models.Efficientnet_V2_S_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, num_classes)

    if not unfreeze_backbone:
        for n,p in model.features.named_parameters():
            p.requires_grad = False

    model = model.to(device)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))

    best = 1e9
    for ep in range(epochs):
        model.train()
        run_loss = 0.0
        for x,y in dl_tr:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            run_loss += loss.item()*x.size(0)
        tr_loss = run_loss/len(ds_tr)

        model.eval()
        va_loss, correct = 0.0, 0
        with torch.no_grad():
            for x,y in dl_va:
                x,y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                va_loss += loss.item()*x.size(0)
                pred = out.argmax(1)
                correct += (pred==y).sum().item()
        va_loss /= len(ds_va); acc = correct/len(ds_va)
        sched.step()
        print(f"epoch {ep}: train_loss={tr_loss:.4f} val_loss={va_loss:.4f} val_acc={acc:.3f}")

        if va_loss < best:
            best = va_loss
            torch.save(model.state_dict(), "best.pt")
    print("saved best model → best.pt")

if __name__ == "__main__":
    # python train_cls.py --data_root data --num_classes 2 --epochs 10
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--unfreeze_backbone", action="store_true")
    args = parser.parse_args()
    main(**vars(args))
```

---

# 12) “স্মল” ডিটেকশন ট্রেনিং স্কেচ (টার্গেট ফরম্যাটসহ)

```python
import torch, torchvision
from torch.utils.data import DataLoader

def collate_fn(b): 
    return list(zip(*b))  # ([imgs], [targets])

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None,     # স্ক্র্যাচ অথবা
                                                             weights_backbone="DEFAULT").to(device)
# ডেটাসেট আপনার DetectionDS—যেখানে target={"boxes":[[x1,y1,x2,y2],...], "labels":[...]}
train_ds = DetectionDS(items, transforms=None)  # এখানে আপনার অগমেন্টেশন বসান
train_dl = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=collate_fn)

opt = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-4)

for epoch in range(10):
    model.train()
    for imgs, targets in train_dl:
        imgs = [im.to(device) for im in imgs]
        targets = [{k: v.to(device) for k,v in t.items()} for t in targets]
        loss_dict = model(imgs, targets)     # dict of losses (cls, box, obj, rpn)
        loss = sum(loss_dict.values())
        opt.zero_grad(); loss.backward(); opt.step()
```

**থিওরি**: Detection মডেল ট্রেনিং কলে **লস রিটার্ন** করে; ইনফারেন্সে **প্রেডিকশন** রিটার্ন করে—এটাই ক্লাসিফিকেশন থেকে বড় পার্থক্য।

---

# 13) সাধারণ ভুল/ট্র্যাপ (কেন হয়, কী করবেন)

* **লেবেল 0 vs 1..K**: torchvision ডিটেকশনে background আলাদা—আপনার `labels` সাধারণত ১ থেকে শুরু করুন।
* **Resize/Flip করলে বক্সও বদলাতে হবে**: অগমেন্টেশনে ইমেজ বদলালে বক্স/মাস্কও একই ট্রান্সফর্মে আপডেট করুন (Albumentations/kornia সাহায্য করে)।
* **Normalization ভুল**: `Normalize`-এর আগে অবশ্যই টেনসর `[0,1]` করতে হবে।
* **num\_workers বেশি**: Windows/macOS-এ `if __name__ == "__main__":` গার্ড দিন; OOM হলে কমান।
* **Class imbalance**: macro-F1 দেখুন; `WeightedRandomSampler` বা class-weight নিন।

---

---

## 1) **ডেটাসেট ও ডেটা লোডিং**

`torchvision.datasets`
এখান থেকে অনেক **ready-made dataset class** পাবেন। এগুলো PyTorch-এর `Dataset` ইন্টারফেস ফলো করে, তাই `DataLoader`-এ প্লাগ-ইন করা যায়।

* **উদাহরণ:**

  * `ImageFolder` → ফোল্ডার-ভিত্তিক কাস্টম ইমেজ ক্লাসিফিকেশন ডেটা
  * `CIFAR10`, `CIFAR100`
  * `MNIST`, `FashionMNIST`
  * `COCO`, `VOCDetection`, `VOCSegmentation`
  * `Cityscapes`, `CelebA`, `Kinetics` (ভিডিও)

**কাজ**:

* সহজে জনপ্রিয় ডেটাসেট ডাউনলোড ও লোড
* আপনার কাস্টম ডেটার জন্যও কিছু ক্লাস (`ImageFolder`, `DatasetFolder`) কাজে লাগবে

**Example:**

```python
from torchvision import datasets
train_ds = datasets.CIFAR10(root='data', train=True, download=True)
```

---

## 2) **ইমেজ ও ভিডিও ট্রান্সফর্ম**

`torchvision.transforms`
ডেটা অগমেন্টেশন ও প্রিপ্রসেসিং-এর জন্য

* **Geometric transforms**: `Resize`, `CenterCrop`, `RandomCrop`, `RandomRotation`, `RandomHorizontalFlip`, `RandomResizedCrop`
* **Color transforms**: `ColorJitter`, `Grayscale`, `RandomAutocontrast`, `RandomEqualize`
* **Tensor transforms**: `ToTensor`, `Normalize`
* **Compose**: একাধিক ট্রান্সফর্ম চেইনে লাগানো
* **v2 API**: GPU-তে টেনসর-বেসড ফাস্ট ট্রান্সফর্ম (যেমন `transforms.v2.Resize`)

**Example:**

```python
from torchvision import transforms
tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
```

---

## 3) **প্রি-ট্রেইন্ড মডেল**

`torchvision.models`
এখানে কম্পিউটার ভিশনের জন্য প্রচুর আর্কিটেকচার আছে।

* **Classification**: `resnet18`, `resnet50`, `efficientnet_v2_s`, `mobilenet_v3_large`, `convnext_tiny`, ইত্যাদি।
* **Detection**: `fasterrcnn_resnet50_fpn`, `retinanet_resnet50_fpn`, `ssdlite320_mobilenet_v3_large`
* **Segmentation**: `fcn_resnet50`, `deeplabv3_resnet50`, `lraspp_mobilenet_v3_large`
* **Keypoint detection**: `keypointrcnn_resnet50_fpn`

সবগুলোর জন্য **weights** অপশন আছে—ImageNet বা COCO প্রি-ট্রেইন্ড নিতে পারবেন।

**Example:**

```python
from torchvision import models
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
```

---

## 4) **ভিশন অপারেশনস (ops)**

`torchvision.ops`
Detection ও Segmentation-এর জন্য দরকারি লো-লেভেল ফাংশন

* `nms` → Non-Max Suppression
* `box_iou`, `box_convert`, `clip_boxes_to_image`
* `roi_align`, `roi_pool`
* `masks_to_boxes`

**Example:**

```python
from torchvision.ops import nms
```

---

## 5) **ইউটিলিটি ও ভিজুয়ালাইজেশন**

`torchvision.utils`

* `make_grid` → একাধিক ইমেজকে গ্রিডে সাজানো
* `save_image` → ইমেজ সেভ
* `draw_bounding_boxes` → ইমেজে বক্স আঁকা
* `draw_segmentation_masks` → সেগমেন্টেশন মাস্ক আঁকা

**Example:**

```python
from torchvision.utils import make_grid, save_image
```

---

## 6) **ইমেজ ও ভিডিও IO**

`torchvision.io`

* `read_image`, `write_png`
* `read_video`, `write_video`
* `decode_video`, `encode_video`

**Example:**

```python
from torchvision.io import read_image
img = read_image("photo.jpg")  # Tensor[C,H,W]
```

---

## 7) **নতুন ট্রান্সফর্ম API**

`torchvision.transforms.v2`

* GPU-অপ্টিমাইজড ট্রান্সফর্ম
* টেনসর ও PIL উভয়ের সাথে কম্প্যাটিবল
* অগমেন্টেশন ট্রেনিং পাইপলাইনে আরও দ্রুত

---

## 8) **ডেটা ফাংশনাল API**

`torchvision.transforms.functional`
লো-লেভেল ফাংশনাল ট্রান্সফর্ম—Compose ছাড়াই সরাসরি ফাংশন কল
**Example:**

```python
from torchvision.transforms import functional as F
img = F.resize(img, [224,224])
```

---

## সারাংশ টেবিল

| Module                  | কী থাকে                            | প্রধান কাজ                |
| ----------------------- | ---------------------------------- | ------------------------- |
| `datasets`              | CIFAR, MNIST, COCO, ImageFolder    | ডেটা লোড                  |
| `transforms`            | Resize, Flip, Normalize            | ডেটা প্রিপ্রসেস ও অগমেন্ট |
| `models`                | ResNet, EfficientNet, Faster R-CNN | প্রি-ট্রেইন্ড মডেল        |
| `ops`                   | nms, box\_iou, roi\_align          | ডিটেকশন/সেগমেন্টেশন ইউটিল |
| `utils`                 | make\_grid, draw\_bounding\_boxes  | ভিজুয়ালাইজেশন            |
| `io`                    | read\_image, read\_video           | মিডিয়া IO                 |
| `transforms.v2`         | টেনসর-বেসড ফাস্ট ট্রান্সফর্ম       | GPU-অপ্টিমাইজড অগমেন্টেশন |
| `transforms.functional` | লো-লেভেল ট্রান্সফর্ম               | কাস্টম প্রসেসিং           |

---





