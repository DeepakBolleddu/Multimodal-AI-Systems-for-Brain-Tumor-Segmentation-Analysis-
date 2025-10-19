import os
import gc
import json
import random
import warnings
from tqdm import tqdm

import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split
from scipy.ndimage import zoom

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

# Ignore common warnings for cleaner output
warnings.filterwarnings('ignore')

# Set seeds for reproducibility, which is crucial for research
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

print(">à Brain Tumor Segmentation - Final Training Framework >à")
print("=" * 60)

# ============================================================================
# 1. 3D SSU-NET MODEL ARCHITECTURE
# A standard and robust choice for volumetric medical image segmentation.
# ============================================================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_z = x2.size()[2] - x1.size()[2]
        diff_y = x2.size()[3] - x1.size()[3]
        diff_x = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2,
                        diff_z // 2, diff_z - diff_z // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class SSUNet3D(nn.Module):
    def __init__(self, n_channels=4, n_classes=4):
        super(SSUNet3D, self).__init__()
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.up4 = Up(64, 32)
        self.outc = nn.Conv3d(32, n_classes, kernel_size=1)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

# ============================================================================
# 2. DATASET AND DATA LOADING
# Handles data splitting, loading all 4 modalities, and correct mask processing.
# ============================================================================

class BraTSDataset(Dataset):
    def __init__(self, data_dir, patient_ids, target_size=(128, 128, 128)):
        self.data_dir = data_dir
        self.patient_ids = patient_ids
        self.target_size = target_size
        self.modalities = ['flair', 't1', 't1ce', 't2']

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        patient_path = os.path.join(self.data_dir, patient_id)
        
        image_channels = []
        for modality in self.modalities:
            file_path = os.path.join(patient_path, f"{patient_id}_{modality}.nii.gz")
            img_data = nib.load(file_path).get_fdata(dtype=np.float32)
            
            if np.sum(img_data) > 0:
                img_data = (img_data - np.mean(img_data)) / (np.std(img_data) + 1e-8)
            
            img_resized = zoom(img_data, [d / s for d, s in zip(self.target_size, img_data.shape)], order=1)
            image_channels.append(img_resized)

        full_image = np.stack(image_channels, axis=0)

        mask_path = os.path.join(patient_path, f"{patient_id}_seg.nii.gz")


        mask_data = nib.load(mask_path).get_fdata()  # Load as default float type
        mask_data = mask_data.astype(np.int64)      # THEN convert to integer

        
        mask_data[mask_data == 4] = 3
        
        mask_resized = zoom(mask_data, [d / s for d, s in zip(self.target_size, mask_data.shape)], order=0)
        
        return torch.from_numpy(full_image), torch.from_numpy(mask_resized).long()

# ============================================================================
# 3. LOSS FUNCTION AND METRICS
# Combined Dice and Cross-Entropy loss for robust training.
# ============================================================================

def dice_coefficient(pred, target, epsilon=1e-6):
    pred_softmax = torch.softmax(pred, dim=1)
    target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 4, 1, 2, 3).float()
    
    pred_softmax = pred_softmax[:, 1:]
    target_one_hot = target_one_hot[:, 1:]

    intersection = torch.sum(pred_softmax * target_one_hot, dim=(2, 3, 4))
    union = torch.sum(pred_softmax, dim=(2, 3, 4)) + torch.sum(target_one_hot, dim=(2, 3, 4))
    
    dice = (2. * intersection + epsilon) / (union + epsilon)
    return torch.mean(dice)

def combined_loss(pred, target, alpha=0.5):
    ce_loss = F.cross_entropy(pred, target)
    dice_loss = 1 - dice_coefficient(pred, target)
    return alpha * ce_loss + (1 - alpha) * dice_loss

# ============================================================================
# 4. TRAINING AND VALIDATION LOOPS
# ============================================================================

def train_epoch(model, loader, optimizer, scaler, device):
    model.train()
    total_loss, total_dice = 0, 0
    pbar = tqdm(loader, desc="   Training", leave=False)
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = combined_loss(outputs, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        dice = dice_coefficient(outputs.detach(), masks)
        total_loss += loss.item()
        total_dice += dice.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{dice.item():.4f}")
    return total_loss / len(loader), total_dice / len(loader)

def validate_epoch(model, loader, device):
    model.eval()
    total_loss, total_dice = 0, 0
    with torch.no_grad():
        pbar = tqdm(loader, desc=" Validating", leave=False)
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = combined_loss(outputs, masks)
            dice = dice_coefficient(outputs, masks)
            total_loss += loss.item()
            total_dice += dice.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{dice.item():.4f}")
    return total_loss / len(loader), total_dice / len(loader)

# ============================================================================
# 5. MAIN TRAINING PIPELINE
# ============================================================================

def main():
    # --- W CRITICAL CONFIGURATION W ---
    # UPDATE THIS PATH to point to your dataset directory on the Gadi HPC.
    # This directory should contain the individual patient folders (e.g., BraTS2021_00000, etc.)
    DATASET_PATH = "/g/data/ii16/Image/BrainTumorSeg/Data/BRATS2021_standardized/"
    
    EPOCHS = 60
    BATCH_SIZE = 1
    LEARNING_RATE = 1e-4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # --- Data Splitting (integrated from your script) ---
    print("\n=Ê Creating Train/Validation Splits...")
    try:
        patient_ids = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
        if not patient_ids:
            raise FileNotFoundError # Trigger error if no subdirectories found
    except FileNotFoundError:
        print(f"L ERROR: No patient sub-folders found in '{DATASET_PATH}'.")
        print("Please ensure DATASET_PATH is correct and points to the directory containing folders like 'BraTS2021_00000'.")
        return
        
    train_ids, val_ids = train_test_split(patient_ids, test_size=0.15, random_state=42)
    print(f"   Found {len(patient_ids)} patients. Using {len(train_ids)} for training and {len(val_ids)} for validation.")
    
    # --- Datasets and DataLoaders ---
    train_dataset = BraTSDataset(DATASET_PATH, train_ids)
    val_dataset = BraTSDataset(DATASET_PATH, val_ids)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # --- Model, Optimizer, and Scheduler ---
    model = SSUNet3D(n_channels=4, n_classes=4).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler()
    
    # --- Training Loop ---
    best_val_dice = 0.0
    print("\n= Starting Training...")
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        train_loss, train_dice = train_epoch(model, train_loader, optimizer, scaler, device)
        val_loss, val_dice = validate_epoch(model, val_loader, device)
        scheduler.step()
        
        print(f"   Epoch Summary | Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f} | Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
        
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"   -> < New best model saved with Val Dice: {best_val_dice:.4f}")
        
        gc.collect()

    print(f"\n< Training complete! The best model achieved a validation Dice score of {best_val_dice:.4f}.")
    print("The weights have been saved to 'best_model.pth'. This is the file you need for your web application.")

if __name__ == '__main__':
    main()
