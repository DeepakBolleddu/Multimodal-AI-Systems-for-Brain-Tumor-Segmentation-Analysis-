import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_block(in_channels, out_channels, kernel_size=3, padding=1, num_convs=2):
    """A block of 2 or 3 3D convolutions, each followed by Batch Norm and PReLU."""
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding))
        layers.append(nn.BatchNorm3d(out_channels))
        layers.append(nn.PReLU(out_channels))
        in_channels = out_channels
    return nn.Sequential(*layers)

class DownTransition(nn.Module):
    """Downsampling path: 2x2x2 convolution, dropout, and a convolutional block."""
    def __init__(self, in_channels, out_channels, num_convs, dropout_p=0.2):
        super(DownTransition, self).__init__()
        self.down_conv = nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = conv_block(out_channels, out_channels, num_convs=num_convs)
        # Dropout layer to help prevent overfitting
        self.dropout = nn.Dropout3d(p=dropout_p)

    def forward(self, x):
        down = self.down_conv(x)
        out = self.conv(down)
        # Apply dropout before the residual connection
        out = self.dropout(out) 
        out = F.prelu(out + down, self.conv[-1].weight) # Residual connection
        return out

class UpTransition(nn.Module):
    """Upsampling path: transposed convolution, cropping, concatenation, and a convolutional block."""
    def __init__(self, in_channels, out_channels, num_convs):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = conv_block(out_channels * 2, out_channels, num_convs=num_convs)

    def forward(self, x, skip_x):
        up = self.up_conv(x)
        
        # Handle spatial dimension mismatch by center-cropping the skip connection tensor
        if skip_x.shape != up.shape:
            skip_shape = skip_x.shape[2:]
            up_shape = up.shape[2:]
            
            padding = [(skip_shape[i] - up_shape[i]) for i in range(len(skip_shape))]
            
            crop_start = [p // 2 for p in padding]
            crop_end = [p - s for p, s in zip(padding, crop_start)]
            
            skip_x = skip_x[:, :, 
                          crop_start[0] : skip_shape[0] - crop_end[0], 
                          crop_start[1] : skip_shape[1] - crop_end[1], 
                          crop_start[2] : skip_shape[2] - crop_end[2]]

        out = torch.cat([up, skip_x], dim=1) # Concatenate with the (now cropped) skip connection
        out = self.conv(out)
        return out

class VNet(nn.Module):
    """
    A standard V-Net implementation for 3D volumetric segmentation with Dropout for regularization.
    Adapted from: https://arxiv.org/abs/1606.04797
    """
    def __init__(self, in_channels=1, out_channels=1, n_filters=16):
        super(VNet, self).__init__()
        
        # --- Encoder Path ---
        self.in_conv = conv_block(in_channels, n_filters, num_convs=1)
        self.down1 = DownTransition(n_filters, n_filters*2, num_convs=2)
        self.down2 = DownTransition(n_filters*2, n_filters*4, num_convs=3)
        self.down3 = DownTransition(n_filters*4, n_filters*8, num_convs=3)
        
        # --- Bottleneck ---
        self.bottleneck = DownTransition(n_filters*8, n_filters*16, num_convs=3)

        # --- Decoder Path ---
        self.up3 = UpTransition(n_filters*16, n_filters*8, num_convs=3)
        self.up2 = UpTransition(n_filters*8, n_filters*4, num_convs=3)
        self.up1 = UpTransition(n_filters*4, n_filters*2, num_convs=2)
        self.out_conv = UpTransition(n_filters*2, n_filters, num_convs=1)

        # --- Final Output Layer ---
        self.final_conv = nn.Conv3d(n_filters, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Bottleneck
        x5 = self.bottleneck(x4)
        
        # Decoder
        up3 = self.up3(x5, x4)
        up2 = self.up2(up3, x3)
        up1 = self.up1(up2, x2)
        out = self.out_conv(up1, x1)
        
        # Final output
        logits = self.final_conv(out)
        return logits

