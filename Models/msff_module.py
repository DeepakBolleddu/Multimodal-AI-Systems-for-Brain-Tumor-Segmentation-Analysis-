# Models/msff_module.py

import torch
import torch.nn as nn

class ChannelSELayer(nn.Module):
    """
    The Squeeze-and-Excitation block for channel-wise attention, adapted for 3D.
    This will serve as our Modality Selection Feature Fusion (MSFF) module.
    """
    def __init__(self, num_channels, reduction_ratio=2):
        """
        Args:
            num_channels (int): Number of input channels.
            reduction_ratio (int): Factor by which to reduce the number of channels
                                   in the bottleneck of the SE block.
        """
        super(ChannelSELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        
        # The "bottleneck" MLP for learning channel importance
        self.fc = nn.Sequential(
            nn.Conv3d(num_channels, num_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_channels // reduction_ratio, num_channels, kernel_size=1),
            nn.Sigmoid() # Sigmoid to output attention weights between 0 and 1
        )

    def forward(self, x):
        # Squeeze: Global information embedding
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x)
        
        # Excitation: Learns channel-specific weights
        y = self.fc(y)
        
        # Scale: Apply the learned attention weights to the input feature map
        return x * y.expand_as(x)
