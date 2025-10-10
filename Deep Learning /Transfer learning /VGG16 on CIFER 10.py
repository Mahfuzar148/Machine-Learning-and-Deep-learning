import os, math, time, copy, random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# -------------------
# Repro & Device
# -------------------
def set_seed(seed=42):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(42)

# -------------------
# Hyperparameters
# -------------------
BATCH_SIZE = 128
HEAD_EPOCHS = 10          # train classifier head only
FINETUNE_EPOCHS = 20      # then unfreeze last conv block
INIT_LR = 3e-4
WEIGHT_DECAY = 5e-4
PATIENCE = 7              # early stopping patience (epochs)
NUM_CLASSES = 10
NUM_WORKERS = 4
SAVE_PATH = "vgg16_cifar10_best.pt"

# -------------------
# Data & Augmentation
# -------------------
# VGG16 expects 224x224 and ImageNet normalization
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

train_tfms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_tfms = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tfms)
val_set   = datasets.CIFAR10(root="./data", train=False, download=True, transform=val_tfms)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# -------------------
# Model: VGG16 pretrained
# -------------------
# Use torchvision's official weights
vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
# Replace classifier for CIFAR-10
in_features = vgg.classifier[6].in_features
vgg.classifier[6] = nn.Linear(in_features, NUM_CLASSES)
vgg = vgg.to(device)

# Loss
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

# -------------------
# Helpers
# -------------------
def accuracy(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, total_acc, total_n = 0.0, 0.0, 0
    for images, targets in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            logits = model(images)
            loss = criterion(logits, targets)
        bs = images.size(0)
        total_loss += loss.item() * bs
        total_acc  += accuracy(logits, targets) * bs
        total_n    += bs
    return total_loss / total_n, total_acc / total_n

def train_one_epoch(model, loader, optimizer, scaler, criterion):
    model.train()
    total_loss, total_acc, total_n = 0.0, 0.0, 0
    for images, targets in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        bs = images.size(0)
        total_loss += loss.item() * bs
        total_acc  += accuracy(logits, targets) * bs
        total_n    += bs
    return total_loss / total_n, total_acc / total_n

def freeze_all_but_classifier(model):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True

def unfreeze_block5_and_head(model):
    # Unfreeze features from Block 5 (layers after features[23]) and the classifier
    # VGG16 features layout: conv layers through indices; block5 typically last 6 conv/relu + maxpool
    for i, m in enumerate(model.features):
        # Block5 starts at conv after index 23 (exact indices: 24..30 conv/relu + 31 pool)
        if i >= 24:
            for p in m.parameters():
                p.requires_grad = True
        else:
            for p in m.parameters():
                p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True

def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# -------------------
# Phase 1: Train head only
# -------------------
freeze_all_but_classifier(vgg)
print(f"[Phase 1] Trainable params: {count_trainable(vgg):,}")

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, vgg.parameters()),
                        lr=INIT_LR, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=HEAD_EPOCHS)
scaler = torch.cuda.amp.GradScaler()

best_val_acc = 0.0
best_wts = copy.deepcopy(vgg.state_dict())
epochs_no_improve = 0

for epoch in range(1, HEAD_EPOCHS+1):
    t0 = time.time()
    train_loss, train_acc = train_one_epoch(vgg, train_loader, optimizer, scaler, criterion)
    val_loss, val_acc = evaluate(vgg, val_loader, criterion)
    scheduler.step()

    improved = val_acc > best_val_acc
    if improved:
        best_val_acc = val_acc
        best_wts = copy.deepcopy(vgg.state_dict())
        torch.save(best_wts, SAVE_PATH)
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    print(f"[Head {epoch:02d}/{HEAD_EPOCHS}] "
          f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
          f"val_loss={val_loss:.4f} acc={val_acc:.4f} | "
          f"{'**BEST**' if improved else ''} "
          f"time={(time.time()-t0):.1f}s")

    if epochs_no_improve >= PATIENCE:
        print("Early stopping (head).")
        break

# Load best head weights before finetune
vgg.load_state_dict(torch.load(SAVE_PATH, map_location=device))

# -------------------
# Phase 2: Fine-tune Block 5 + classifier
# -------------------
unfreeze_block5_and_head(vgg)
print(f"[Phase 2] Trainable params: {count_trainable(vgg):,}")

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, vgg.parameters()),
                        lr=INIT_LR/2, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=FINETUNE_EPOCHS)
scaler = torch.cuda.amp.GradScaler()

epochs_no_improve = 0

for epoch in range(1, FINETUNE_EPOCHS+1):
    t0 = time.time()
    train_loss, train_acc = train_one_epoch(vgg, train_loader, optimizer, scaler, criterion)
    val_loss, val_acc = evaluate(vgg, val_loader, criterion)
    scheduler.step()

    improved = val_acc > best_val_acc
    if improved:
        best_val_acc = val_acc
        best_wts = copy.deepcopy(vgg.state_dict())
        torch.save(best_wts, SAVE_PATH)
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    print(f"[FT  {epoch:02d}/{FINETUNE_EPOCHS}] "
          f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
          f"val_loss={val_loss:.4f} acc={val_acc:.4f} | "
          f"{'**BEST**' if improved else ''} "
          f"time={(time.time()-t0):.1f}s")

    if epochs_no_improve >= PATIENCE:
        print("Early stopping (finetune).")
        break

# -------------------
# Final evaluation
# -------------------
vgg.load_state_dict(torch.load(SAVE_PATH, map_location=device))
val_loss, val_acc = evaluate(vgg, val_loader, criterion)
print(f"Best checkpoint -> val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
print(f"Saved: {SAVE_PATH}")
