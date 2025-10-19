import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# ============================================================================
# 1. MODEL ARCHITECTURE
# The SSUNet3D architecture from the training script is now included directly
# to ensure the loader knows the exact structure of the model.
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
    """ The high-performance model architecture from your successful training run. """
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
# 2. UPDATED LOADER FUNCTIONS
# These functions now build and load the SSUNet3D model correctly.
# ============================================================================

def _device_from_env():
    dev = os.getenv("DEVICE", "auto").lower()
    return torch.device("cuda" if dev == "auto" and torch.cuda.is_available() else "cpu")

def build_model(in_channels=4, n_classes=4) -> nn.Module:
    """ Builds the SSUNet3D model. """
    print("Building high-performance SSUNet3D model.")
    return SSUNet3D(n_channels=in_channels, n_classes=n_classes)

def load_model(model_path: Optional[str]) -> nn.Module:
    """
    Loads the SSUNet3D model and its trained weights from best_model.pth.
    """
    device = _device_from_env()
    # Your model has 4 input channels (modalities) and 4 output classes
    model = build_model(in_channels=4, n_classes=4).to(device)

    if not model_path or not os.path.exists(model_path):
        print("WARNING: Model path not found. The application will not produce valid segmentations.")
        return model.eval()

    try:
        # Load the state dictionary from your .pth file
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        print(f"Successfully loaded model weights from {os.path.basename(model_path)}")
    except Exception as e:
        print(f"ERROR loading model weights: {e}")
        print("   The application will not produce valid segmentations.")
    
    return model.eval()

@torch.inference_mode()
def run_inference(model: nn.Module, vol_4d: torch.Tensor) -> torch.Tensor:
    """
    Runs inference on the input volume.
    vol_4d: (1, 4, D, H, W) float32 tensor
    returns: (1, 4, D, H, W) raw logits from the model
    """
    device = next(model.parameters()).device
    return model(vol_4d.to(device))
