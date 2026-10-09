# Small CPU-feasibility 2D U-Net baseline for MBH-Seg25

from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLIT_CSV = PROJECT_ROOT / "preprocessing" / "segmentation" / "mbh_seg25_split.csv"
NUM_CLASSES = 6
IMAGE_SIZE = 256
BATCH_SIZE = 4
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "segmentation"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

class SliceDataset(Dataset):
    def __init__(self, split):
        df = pd.read_csv(SPLIT_CSV)
        df = df[df["split"] == split].reset_index(drop=True)
        self.samples = []
        for _, row in df.iterrows():
            image = nib.load(str(row["image_path"])).get_fdata(dtype=np.float32)
            mask = nib.load(str(row["annot_3_path"])).get_fdata(dtype=np.float32)
            for z in range(image.shape[2]):
                self.samples.append((image[:, :, z], mask[:, :, z].astype(np.uint8)))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        image, mask = self.samples[index]
        image = torch.from_numpy(image.copy()).unsqueeze(0)
        mask = torch.from_numpy(mask.copy()).long().unsqueeze(0)
        image = F.interpolate(image.unsqueeze(0), size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False).squeeze(0)
        mask = F.interpolate(mask.float().unsqueeze(0), size=(IMAGE_SIZE, IMAGE_SIZE), mode="nearest").squeeze(0).long()
        image = torch.clamp(image, -1000, 1000)
        image = (image + 1000.0) / 2000.0
        return image, mask.squeeze(0)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
    def forward(self, x):
        return self.block(x)

class SmallUNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.enc1 = DoubleConv(1, 16)
        self.enc2 = DoubleConv(16, 32)
        self.enc3 = DoubleConv(32, 64)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(64, 96)
        self.up3 = nn.ConvTranspose2d(96, 64, 2, stride=2)
        self.dec3 = DoubleConv(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = DoubleConv(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = DoubleConv(32, 16)
        self.out = nn.Conv2d(16, num_classes, 1)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)

def dice_loss(logits, target):
    probabilities = torch.softmax(logits, dim=1)
    target_one_hot = F.one_hot(target, NUM_CLASSES).permute(0, 3, 1, 2).float()
    intersection = (probabilities * target_one_hot).sum(dim=(0, 2, 3))
    denominator = probabilities.sum(dim=(0, 2, 3)) + target_one_hot.sum(dim=(0, 2, 3))
    dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
    return 1.0 - dice.mean()

def main():
    torch.set_num_threads(12)
    device = torch.device("cpu")
    print("Device:", device)
    train_dataset = SliceDataset("train")
    val_dataset = SliceDataset("validation")
    print("Training slices:", len(train_dataset))
    print("Validation slices:", len(val_dataset))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model = SmallUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ce_loss = nn.CrossEntropyLoss()
    epochs = 2
    best_val_loss = float("inf")
    history = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for step, (images, masks) in enumerate(train_loader, start=1):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = ce_loss(logits, masks) + dice_loss(logits, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if step == 1 or step % 25 == 0:
                print(f"Epoch {epoch + 1}/{epochs} Step {step}/{len(train_loader)} Loss {loss.item():.4f}")
        train_loss = running_loss / len(train_loader)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                val_loss += (ce_loss(logits, masks) + dice_loss(logits, masks)).item()
        val_loss /= len(val_loader)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch + 1}/{epochs} TrainLoss {train_loss:.4f} ValLoss {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = CHECKPOINT_DIR / "small_unet_mbh_seg25_best.pt"
            torch.save({"epoch": epoch + 1, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "train_loss": train_loss, "val_loss": val_loss, "history": history}, checkpoint_path)
            print("Saved checkpoint:", checkpoint_path)

if __name__ == "__main__":
    main()
