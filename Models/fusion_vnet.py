# Models/fusion_vnet.py

import torch
import torch.nn as nn
from model import conv_block, UpTransition, DownTransition # Reuse blocks from our old model.py
from msff_module import ChannelSELayer # Import our new MSFF module

class FusionVNet(nn.Module):
    """
    The final V-Net architecture incorporating modality-specific pathways
    and an MSFF attention module for feature-level fusion.
    """
    def __init__(self, in_channels=4, out_channels=1, n_filters=8):
        super(FusionVNet, self).__init__()
        
        # --- Modality-Specific Encoder Branches ---
        # Create a separate initial conv block for each of the 4 modalities
        self.in_conv_flair = conv_block(1, n_filters, num_convs=1)
        self.in_conv_t1 = conv_block(1, n_filters, num_convs=1)
        self.in_conv_t1ce = conv_block(1, n_filters, num_convs=1)
        self.in_conv_t2 = conv_block(1, n_filters, num_convs=1)
        
        # --- MSFF Module ---
        # This will fuse the features from the 4 parallel branches.
        # Input channels = 4 branches * n_filters per branch
        self.msff = ChannelSELayer(num_channels=n_filters * 4)
        
        # --- Common V-Net Pathway (post-fusion) ---
        # The rest of the V-Net architecture is standard.
        # It takes the fused features as input.
        self.down1 = DownTransition(n_filters * 4, n_filters * 8, num_convs=2)
        self.down2 = DownTransition(n_filters * 8, n_filters * 16, num_convs=3)
        self.down3 = DownTransition(n_filters * 16, n_filters * 32, num_convs=3)
        
        self.up3 = UpTransition(n_filters * 32, n_filters * 16, num_convs=3)
        self.up2 = UpTransition(n_filters * 16, n_filters * 8, num_convs=3)
        self.up1 = UpTransition(n_filters * 8, n_filters * 4, num_convs=2)
        
        self.final_conv = nn.Conv3d(n_filters * 4, out_channels, kernel_size=1)

    def forward(self, x):
        # x is the 4-channel input (B, 4, H, W, D)
        
        # Split the input into 4 separate modalities
        x_flair = x[:, 0:1, :, :, :]
        x_t1 = x[:, 1:2, :, :, :]
        x_t1ce = x[:, 2:3, :, :, :]
        x_t2 = x[:, 3:4, :, :, :]
        
        # Pass each modality through its dedicated encoder branch
        f_flair = self.in_conv_flair(x_flair)
        f_t1 = self.in_conv_t1(x_t1)
        f_t1ce = self.in_conv_t1ce(x_t1ce)
        f_t2 = self.in_conv_t2(x_t2)
        
        # Concatenate the modality-specific features
        x1_concat = torch.cat([f_flair, f_t1, f_t1ce, f_t2], dim=1)
        
        # Apply the MSFF attention module to fuse the features
        x1_fused = self.msff(x1_concat)
        
        # Pass the fused features through the rest of the V-Net
        x2 = self.down1(x1_fused)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        up3 = self.up3(x4, x3)
        up2 = self.up2(up3, x2)
        up1 = self.up1(up2, x1_fused) # Use the fused features for the final skip connection
        
        logits = self.final_conv(up1)
        return logits
