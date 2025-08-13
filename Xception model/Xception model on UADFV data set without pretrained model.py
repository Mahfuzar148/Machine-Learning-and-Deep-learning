!pip install timm --quiet

import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
from sklearn.metrics import accuracy_score

# Configuration
DATA_DIR = "/kaggle/input/uadfv-dataset/UADFV"  # your dataset path here
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom Dataset to handle nested folders inside fake/frames and real/frames
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

# Data transforms
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

# Split dataset into train/val (e.g., 80/20 split)
full_dataset = UADFVDataset(DATA_DIR, transform=train_transforms)

# Shuffle and split indices
dataset_size = len(full_dataset)
indices = torch.randperm(dataset_size).tolist()
split = int(0.8 * dataset_size)
train_indices, val_indices = indices[:split], indices[split:]

from torch.utils.data import Subset

train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(full_dataset, val_indices)

# Override transforms for val dataset
val_dataset.dataset.transform = val_transforms

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

print(f"Total samples: {dataset_size}, Train: {len(train_dataset)}, Val: {len(val_dataset)}")

# Model classes same as before:

class XceptionFeatureExtractor(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.xception = timm.create_model('legacy_xception', pretrained=pretrained)
        self.xception.reset_classifier(0)  # Remove FC

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
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        out = self.classifier(features)
        return out

#model_pretrained = DeepFakeDetector(pretrained=True).to(DEVICE)
model_scratch = DeepFakeDetector(pretrained=False).to(DEVICE)

criterion = nn.BCELoss()
#optimizer_pretrained = torch.optim.Adam(model_pretrained.parameters(), lr=LEARNING_RATE)
optimizer_scratch = torch.optim.Adam(model_scratch.parameters(), lr=LEARNING_RATE)

def train_one_epoch(model, dataloader, optimizer):
    model.train()
    running_loss = 0
    all_preds, all_labels = [], []
    for imgs, labels in dataloader:
        imgs = imgs.to(DEVICE)
        labels = labels.float().unsqueeze(1).to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

        preds = (outputs > 0.5).long().cpu()
        all_preds.extend(preds.squeeze().tolist())
        all_labels.extend(labels.cpu().squeeze().tolist())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

def validate(model, dataloader):
    model.eval()
    running_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
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
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

# for pretrained_flag in [True, False]:
#     print(f"\nTraining model with pretrained={pretrained_flag}")
#     model = model_pretrained if pretrained_flag else model_scratch
#     optimizer = optimizer_pretrained if pretrained_flag else optimizer_scratch

print("\nTraining model from scratch (pretrained=False)")
best_val_acc = 0
for epoch in range(NUM_EPOCHS):
    train_loss, train_acc = train_one_epoch(model_scratch, train_loader, optimizer_scratch)
    val_loss, val_acc = validate(model_scratch, val_loader)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model_scratch.state_dict(), "xception_scratch.pth")
        print("Model saved.")

print("Training complete.")
