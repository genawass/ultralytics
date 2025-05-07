import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings

# Ensure these imports match your Ultralytics version's structure.
# These are common locations for these modules.
from ultralytics.nn.modules.conv import Conv, autopad
from ultralytics.nn.modules.block import Bottleneck # Base Bottleneck
from ultralytics.nn.modules import SPPF # Base SPPF

# Spatial Attention Module (SAM) - Based on Figure 4 of the PC-YOLO paper
class SAM(nn.Module):
    """
    Spatial Attention Module
    Applies spatial attention to the input feature map.
    As described in CBAM and used in PC-YOLO's CSA module.
    """
    def __init__(self, kernel_size=7):
        super(SAM, self).__init__()
        assert kernel_size in (3, 7), 'SAM kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        # Convolution layer to process concatenated average and max pooled features
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid() # Sigmoid activation to get attention weights

    def forward(self, x):
        # Perform average pooling and max pooling along the channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # Concatenate the pooled features
        x_cat = torch.cat([avg_out, max_out], dim=1)
        # Pass through convolution and sigmoid to get spatial attention map
        attention_map = self.sigmoid(self.conv(x_cat))
        # Multiply the input feature map by the attention map
        return attention_map * x

# Coordinate Spatial Attention (CSA) Module - Based on Figure 4 of the PC-YOLO paper
class CSA(nn.Module):
    """
    Coordinate Spatial Attention (CSA) Module
    Combines Spatial Attention (SAM) with Coordinate Attention.
    """
    def __init__(self, c1, reduction=32, sam_kernel_size=7): # c1: input channels
        super(CSA, self).__init__()
        print(f"[CSA.__init__] Instantiating with c1={c1}, type={type(c1)}, reduction={reduction}") # DEBUG
        if not isinstance(c1, int) or c1 <= 0:
            raise ValueError(f"CSA input channels c1 must be a positive integer, got {c1}")

        # First, apply Spatial Attention Module (SAM)
        self.sam = SAM(kernel_size=sam_kernel_size)

        # Coordinate Attention part
        # Adaptive pooling for height and width dimensions
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1)) # Average pool along width
        self.pool_w = nn.AdaptiveAvgPool2d((1, None)) # Average pool along height

        # Calculate intermediate channels for dimensionality reduction
        mip = max(8, c1 // reduction)
        print(f"[CSA.__init__] Calculated mip={mip}, type={type(mip)}") # DEBUG
        if not isinstance(mip, int) or mip <= 0:
            raise ValueError(f"CSA intermediate channels mip must be a positive integer, got {mip}")

        # Convolution layers for processing pooled features
        print(f"[CSA.__init__] Creating conv1: Conv2d({c1}, {mip}, 1)") # DEBUG
        self.conv1 = nn.Conv2d(c1, mip, 1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU() # SiLU activation function (common in YOLO)

        # Convolution layers to generate attention weights for height and width
        print(f"[CSA.__init__] Creating conv_h: Conv2d({mip}, {c1}, 1)") # DEBUG
        self.conv_h = nn.Conv2d(mip, c1, 1, stride=1, padding=0, bias=False)
        print(f"[CSA.__init__] Creating conv_w: Conv2d({mip}, {c1}, 1)") # DEBUG
        self.conv_w = nn.Conv2d(mip, c1, 1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply SAM to get spatially refined features
        x_sam = self.sam(x)

        # Coordinate Attention mechanism
        _, _, h, w = x_sam.shape # Get height and width of SAM output

        # Pool features along height and width
        x_h_pooled = self.pool_h(x_sam)
        x_w_pooled = self.pool_w(x_sam).permute(0, 1, 3, 2) # Permute to make width the last dim for cat

        # Concatenate pooled features
        xy_cat = torch.cat([x_h_pooled, x_w_pooled], dim=2)
        
        # Process concatenated features
        xy_processed = self.act(self.bn1(self.conv1(xy_cat)))

        # Split back into height and width components
        x_h_processed, x_w_processed = torch.split(xy_processed, [h, w], dim=2)
        x_w_processed = x_w_processed.permute(0, 1, 3, 2) # Permute back

        # Generate attention weights for height and width
        attention_h = self.sigmoid(self.conv_h(x_h_processed))
        attention_w = self.sigmoid(self.conv_w(x_w_processed))

        # Apply coordinate attention weights to the SAM-processed features
        out = x_sam * attention_w * attention_h
        return out

# CSA_Bottleneck - Based on Figure 5 (integrating CSA into Bottleneck)
class CSA_Bottleneck(Bottleneck):
    """
    Bottleneck block with CSA attention.
    The CSA module is applied after the residual connection.
    This Bottleneck maintains an internal expansion factor 'e' of 1.0 by default,
    meaning its internal channels (c_) are equal to its output channels (c2)
    if not overridden by 'e' during instantiation for other purposes.
    For C3-like blocks, 'e=1.0' is typically used for the Bottlenecks.
    """
    def __init__(self, c1, c2, shortcut=True, g=1, e=1.0, csa_reduction=32):
        # Call the parent Bottleneck's __init__
        # 'e=1.0' ensures that the internal channels of this bottleneck (self.c_)
        # are equal to c2 if it's a standard bottleneck structure (c1->c_ ->c2 where c_=c2*e)
        # However, Ultralytics Bottleneck uses e for c_ = int(c2 * e).
        # For Bottleneck(c_, c_, e=1.0) as in C2f, c_internal becomes c_ * 1.0 = c_.
        super().__init__(c1, c2, shortcut, g, e=e)
        # Initialize CSA module, applied on the output channels 'c2' of the bottleneck
        self.csa = CSA(c2, reduction=csa_reduction)

    def forward(self, x):
        # Standard Bottleneck forward pass (cv1 -> cv2 -> add residual)
        # The 'self.add' flag is set in Bottleneck's __init__
        out = self.cv2(self.cv1(x))
        if self.add: # self.add is True if shortcut=True and c1==c2
            out = x + out
        # Apply CSA after the residual connection (or final convs)
        return self.csa(out)

# CSA_SPPF Module - Based on Figure 5
class CSA_SPPF(SPPF):
    """
    SPPF (Spatial Pyramid Pooling - Fast) block with CSA attention.
    CSA is applied after the initial convolution (self.cv1).
    """
    def __init__(self, c1, c2, k=5, csa_reduction=32): # k is the SPPF MaxPool kernel size
        super().__init__(c1, c2, k)
        # The output channels of self.cv1 is self.c_ (which is c2 // 2 in SPPF)
        # CSA is applied on these channels.
        self.csa = CSA(self.cv1.conv.out_channels, reduction=csa_reduction)

    def forward(self, x):
        x = self.cv1(x) # Initial convolution
        x = self.csa(x) # Apply CSA here
        
        # Standard SPPF MaxPool operations
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # Suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            # Concatenate and pass through final convolution
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

# CSA_C3k2_F Module
# This module is intended to replace C3k2 blocks where 'shortcut=False' in the YAML.
# It's structurally similar to Ultralytics' C2f block, but uses CSA_Bottleneck.
class CSA_C3k2_F(nn.Module):
    """
    A C3-like module for 'shortcut=False' cases (similar to C2f), using CSA_Bottlenecks.
    Args:
        c1 (int): input channels.
        c2 (int): output channels.
        n (int, optional): number of CSA_Bottleneck repeats. Defaults to 1.
        e (float, optional): expansion factor for hidden channels within this C3 block. Defaults to 0.5.
                             This 'e' determines the channel size for the bottlenecks.
        csa_reduction (int, optional): reduction factor for CSA modules. Defaults to 32.
        g (int, optional): groups for convolutions within Bottleneck. Defaults to 1.
    """
    def __init__(self, c1, c2, n=1, e=0.5, csa_reduction=32, g=1):
        super().__init__()
        self.n = n # Store n as an instance attribute
        self.c_ = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c_, 1, 1, act=nn.SiLU())
        self.cv2 = Conv((2 + n) * self.c_, c2, 1, act=nn.SiLU())
        self.m = nn.ModuleList(
            CSA_Bottleneck(self.c_, self.c_, shortcut=True, g=g, e=1.0, csa_reduction=csa_reduction) for _ in range(n)
        )

    def forward(self, x):
        # Forward pass similar to C2f
        y = list(self.cv1(x).split((self.c_, self.c_), 1)) # Split into two parts of size self.c_
        # The first n parts from y (if y was split into n+1 for m and 1 for direct) go to m.
        # In C2f, y is a list of tensors resulting from splitting cv1's output.
        # y.extend(m(y[-1]) for m in self.m) # This is if cv1 output 1 part for m, and m takes that one.
        
        # C2f structure:
        # x_cv1 = self.cv1(x) # Output shape (batch, 2 * self.c, H, W)
        # y_splits = list(x_cv1.split((self.c_, self.c_), dim=1)) # Two initial splits
        # y_main_path = y_splits[0] # This goes through the bottlenecks
        # y_direct_concat = y_splits[1] # This is one of the direct concatenations

        # Let's use the simpler split logic from C2f's forward if possible:
        # x_cv1 = self.cv1(x)
        # y = list(x_cv1.chunk(2 + self.n, 1)) # This is not how C2f does it.
        # C2f:
        # x0 = x.transpose(0, 1)
        # x1 = list(self.cv1(x0).chunk(2, 1))
        # x2 = [x1[0], x1[1]]
        # x0 = x0.transpose(0, 1)
        # x1[0] = self.m[0](x1[0])
        # ... this is for DDP.

        # Simpler C2f forward:
        x_cv1_out = self.cv1(x) # shape: (batch, 2 * self.c_, H, W)
        
        # The first self.c_ channels go to the first bottleneck,
        # the next self.c_ channels are for the direct path in the first concatenation.
        # This isn't quite right for multiple bottlenecks in sequence.
        # C2f's `self.m` takes one of the splits and processes it through all bottlenecks.
        
        # Correct C2f logic for 'y':
        splits = self.cv1(x).split((self.c_, self.c_), 1) # Two initial splits from cv1's output
        
        # The first split (splits[0]) is processed by the ModuleList of bottlenecks
        current_split_for_bottlenecks = splits[0]
        bottleneck_outputs = []
        for i in range(self.n):
            current_split_for_bottlenecks = self.m[i](current_split_for_bottlenecks)
            bottleneck_outputs.append(current_split_for_bottlenecks)
            
        # Concatenate the initial direct split (splits[1]) and all bottleneck outputs
        # The order in C2f is [splits[0], splits[1], m[0](splits[0]), m[1](m[0](splits[0])) ... ]
        # No, C2f forward:
        # y = list(self.cv1(x).split((self.c, self.c), 1))
        # y.extend(m(y[-1]) for m in self.m) # This is the key part for C2f
        # This means the *second* initial split (y[1]) is passed through the bottlenecks sequentially.

        y_for_cat = list(splits) # Contains [split_for_bottlenecks, direct_split_2]
        
        # Pass the first split through the chain of bottlenecks
        processed_by_bottlenecks = y_for_cat[0] # This is splits[0]
        for i in range(self.n):
            processed_by_bottlenecks = self.m[i](processed_by_bottlenecks)
            y_for_cat.append(processed_by_bottlenecks) # Append output of each bottleneck

        # y_for_cat will now contain:
        # [initial_split_0 (that went into m), initial_split_1 (direct), m0_out, m1_out, ..., mn-1_out]
        # This is not quite C2f's y.extend(m(y[-1])...
        # Let's re-do CSA_C3k2_F's forward to match C2f's y.extend() logic:

        # cv1 output is (batch, 2 * self.c_, H, W)
        # It's split into two tensors, each (batch, self.c_, H, W)
        # Let these be s0 and s1.
        # y initially contains [s0, s1].
        # Then, m[0] processes s1 (y[-1]). Output is appended. y = [s0, s1, m[0](s1)]
        # Then, m[1] processes m[0](s1) (y[-1]). Output is appended. y = [s0, s1, m[0](s1), m[1](m[0](s1))]
        
        # Corrected forward for CSA_C3k2_F, mimicking C2f:
        y = list(self.cv1(x).split((self.c_, self.c_), 1)) # y = [split0, split1]
        for i in range(self.n):
            y.append(self.m[i](y[-1])) # y becomes [split0, split1, m0(split1), m1(m0(split1)), ...]
                                       # The input to m[i] is the output of m[i-1] (or split1 for m[0])
        
        return self.cv2(torch.cat(y, 1))