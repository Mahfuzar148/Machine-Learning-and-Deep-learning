!pip install timm --quiet

import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import timm
from sklearn.metrics import accuracy_score

# ----------------- Config -----------------
DATA_DIR = "/kaggle/input/uadfv-dataset/UADFV"  # your dataset path here
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- Dataset -----------------
class UADFVDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir should contain 'fake/frames' and 'real/frames'
        """
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Load fake images (label 1)
        fake_dirs = glob.glob(os.path.join(root_dir, "fake", "frames", "*"))
        for folder in fake_dirs:
            imgs = glob.glob(os.path.join(folder, "*.png"))
            for img_path in imgs:
                self.image_paths.append(img_path)
                self.labels.append(1)  # fake label

        # Load real images (label 0)
        real_dirs = glob.glob(os.path.join(root_dir, "real", "frames", "*"))
        for folder in real_dirs:
            imgs = glob.glob(os.path.join(folder, "*.png"))
            for img_path in imgs:
                self.image_paths.append(img_path)
                self.labels.append(0)  # real label

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# ----------------- Transforms -----------------
train_transforms = transforms.Compose([
    transforms.Resize((299, 299)),  # Xception input size
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ----------------- Split (80/20) -----------------
# শুধু length বের করতে neutral dataset (কোনো transform ছাড়া)
full_for_split = UADFVDataset(DATA_DIR, transform=None)
dataset_size = len(full_for_split)

indices = torch.randperm(dataset_size).tolist()
split = int(0.8 * dataset_size)
train_indices, val_indices = indices[:split], indices[split:]

# ✅ FIX: train/val এর জন্য আলাদা dataset instance
train_base = UADFVDataset(DATA_DIR, transform=train_transforms)
val_base   = UADFVDataset(DATA_DIR, transform=val_transforms)

train_dataset = Subset(train_base, train_indices)
val_dataset   = Subset(val_base,   val_indices)

pin_mem = (DEVICE.type == "cuda")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=pin_mem)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=pin_mem)

print(f"Total samples: {dataset_size}, Train: {len(train_dataset)}, Val: {len(val_dataset)}")

# ----------------- Model -----------------
class XceptionFeatureExtractor(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.xception = timm.create_model('legacy_xception', pretrained=pretrained)
        self.xception.reset_classifier(0)  # Remove FC -> returns features (≈2048-d)

    def forward(self, x):
        return self.xception(x)

class DeepFakeDetector(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.feature_extractor = XceptionFeatureExtractor(pretrained=pretrained)
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()   # keeping BCELoss below
        )

    def forward(self, x):
        features = self.feature_extractor(x)   # [B, 2048]
        out = self.classifier(features)        # [B, 1] in (0,1)
        return out

model_scratch = DeepFakeDetector(pretrained=False).to(DEVICE)

# ----------------- Loss/Optim -----------------
criterion = nn.BCELoss()  # with sigmoid above
optimizer_scratch = torch.optim.Adam(model_scratch.parameters(), lr=LEARNING_RATE)

# ----------------- Train / Validate -----------------
from sklearn.metrics import accuracy_score

def train_one_epoch(model, dataloader, optimizer):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for imgs, labels in dataloader:
        imgs = imgs.to(DEVICE)
        labels = labels.float().unsqueeze(1).to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)                  # probs ∈ (0,1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = (outputs > 0.5).long().cpu()
        all_preds.extend(preds.squeeze().tolist())
        all_labels.extend(labels.cpu().squeeze().tolist())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc  = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

@torch.no_grad()
def validate(model, dataloader):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for imgs, labels in dataloader:
        imgs = imgs.to(DEVICE)
        labels = labels.float().unsqueeze(1).to(DEVICE)

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * imgs.size(0)
        preds = (outputs > 0.5).long().cpu()
        all_preds.extend(preds.squeeze().tolist())
        all_labels.extend(labels.cpu().squeeze().tolist())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc  = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

# ----------------- Loop -----------------
print("\nTraining model from scratch (pretrained=False)")
best_val_acc = 0.0
for epoch in range(NUM_EPOCHS):
    train_loss, train_acc = train_one_epoch(model_scratch, train_loader, optimizer_scratch)
    val_loss, val_acc     = validate(model_scratch, val_loader)

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model_scratch.state_dict(), "xception_scratch.pth")
        print("Model saved.")

print("Training complete.")
