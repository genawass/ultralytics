import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv

class CoordinateAttention(nn.Module):
    """Coordinate Attention module as described in the PC-YOLO paper.
    Extracts location-sensitive features by pooling and combining coordinate information.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Ensure reduction doesn't create zero channels
        if channels < reduction:
            reduction = max(1, channels // 2)
            
        # Calculate reduced channels
        reduced_channels = max(1, channels // reduction)
        
        # Coordinate information embedding
        self.x_embed = nn.Conv2d(channels, reduced_channels, 1)
        self.y_embed = nn.Conv2d(channels, reduced_channels, 1) 
        
        # Attention weights
        self.attention = nn.Sequential(
            nn.Conv2d(reduced_channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )
        
    def forward(self, x):
        b, c, h, w = x.size()
        
        # Generate coordinate information
        x_avg = torch.mean(x, dim=2, keepdim=True)  # [b,c,1,w]
        y_avg = torch.mean(x, dim=3, keepdim=True)  # [b,c,h,1]
        
        # Embed coordinate information
        x_embed = self.x_embed(x_avg)  # [b,c/r,1,w]
        y_embed = self.y_embed(y_avg)  # [b,c/r,h,1]
        
        # Combine coordinate information
        x_embed = x_embed.expand(-1, -1, h, -1)
        y_embed = y_embed.expand(-1, -1, -1, w)
        coord_embed = x_embed + y_embed
        
        # Generate attention weights
        attention = self.attention(coord_embed)
        
        return x * attention

class CSA_Bottleneck(nn.Module):
    """Coordinate-Sensitive Attention Bottleneck for PC-YOLO.
    Replaces standard Bottleneck but adds coordinate attention.
    """
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        # Standard Bottleneck channel calculation
        c_ = int(c1 * e)
        c_ = max(c_, 1)  # Ensure at least 1 hidden channel

        # Standard Bottleneck layers
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
        # Add coordinate attention
        self.ca = CoordinateAttention(c2)
        
    def forward(self, x):
        # Standard Bottleneck forward with coordinate attention
        return x + self.ca(self.cv2(self.cv1(x))) if self.add else self.ca(self.cv2(self.cv1(x)))

class CSA_C3k2(nn.Module):
    """Coordinate-Sensitive Attention C3k2 block for PC-YOLO.
    Modified C3k2 with coordinate attention.
    """
    def __init__(self, c1, c2=None, n=1, c3k=False, e=0.5):
        super().__init__()
        # Ensure c1 is a valid number
        c1 = max(c1, 1)
        
        # Convert c2 if it's None or a float to an integer
        if c2 is None:
            c2 = c1
        elif isinstance(c2, float):
            c2 = int(c1 * c2)
        else:
            c2 = int(c2)
        
        # Standard C3 channel calculations
        c_ = int(c1 * e)
        c_ = max(c_, 1)  # Ensure at least 1 hidden channel
        
        # Create layers, matching C3k2 structure but with CSA_Bottleneck
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(CSA_Bottleneck(c_, c_, True, 1, 1.0) for _ in range(n)))
        # Add coordinate attention at the end
        self.ca = CoordinateAttention(c2)
        
    def forward(self, x):
        # Standard C3k2 forward with added coordinate attention
        return self.ca(self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1)))

class CSA_SPPF(nn.Module):
    """Coordinate-Sensitive Attention SPPF module for PC-YOLO."""
    def __init__(self, c1, c2=None, k=5):
        super().__init__()
        # Validate inputs
        c1 = max(c1, 1)  # Ensure at least 1 input channel
        
        # Handle c2 parameter based on type
        if isinstance(c2, float):
            c2 = int(c1 * c2)  # Interpret float as a multiplier of c1
        else:
            c2 = c1 if c2 is None else max(int(c2), 1)  # Default c2 to c1 if None
            
        c_ = max(c1 // 2, 1)  # Ensure at least 1 hidden channel
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.ca = CoordinateAttention(c2)
        
    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        cat = torch.cat((x, y1, y2, self.m(y2)), 1)
        cv2_out = self.cv2(cat)
        ca_out = self.ca(cv2_out)
        return ca_out

class CSA_C2PSA(nn.Module):
    """Coordinate-Sensitive Attention C2PSA module for PC-YOLO."""
    def __init__(self, c1, c2=None, n=1, e=0.5):
        super().__init__()
        # Validate inputs
        c1 = max(c1, 1)  # Ensure at least 1 input channel
        
        # Handle c2 parameter
        if isinstance(c2, float):
            c2 = int(c1 * c2)  # Interpret float as a multiplier of c1
        else:
            c2 = c1 if c2 is None else max(int(c2), 1)  # Default c2 to c1 if None
        
        c_ = max(int(c1 * e), 1)  # Use c1 for expansion as per original C2PSA
        self.cv1 = Conv(c1, 2 * c_, 1, 1)
        self.cv2 = Conv(2 * c_, c2, 1)  # Output uses c2 (either same as c1 or specified value)
        self.m = nn.Sequential(*(CSA_Bottleneck(c_, c_, True, 1, 1.0) for _ in range(n)))
        self.ca = CoordinateAttention(c2)
        
    def forward(self, x):
        a, b = self.cv1(x).chunk(2, 1)
        m_out = self.m(a)
        cat = torch.cat((m_out, b), 1)
        cv2_out = self.cv2(cat)
        ca_out = self.ca(cv2_out)
        return ca_out

class CSA_C3k2_compat(nn.Module):
    """CSA_C3k2 module completely compatible with C3k2 API and implementation.
    This module is a direct copy of C3k2 but adds coordinate attention at the end.
    """
    def __init__(self, c1, c2=None, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """
        Initialize CSA_C3k2_compat module.
        
        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of blocks.
            c3k (bool): Whether to use C3k blocks.
            e (float): Expansion ratio.
            g (int): Groups for convolutions.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__()
        print(f"CSA_C3k2_compat: c1={c1}, c2={c2}, n={n}, e={e}")
        
        # C2f implementation (parent of C3k2)
        c2 = c1 if c2 is None else c2  # if no c2 specified, set to c1
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        
        # Use Bottleneck same as in C3k2
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )
        
        # Add coordinate attention at the end
        self.ca = CoordinateAttention(c2)
        
    def forward(self, x):
        """Forward pass through CSA_C3k2_compat layer."""
        # Same forward pass as C2f (parent of C3k2)
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        # Add coordinate attention to the output
        return self.ca(self.cv2(torch.cat(y, 1))) 