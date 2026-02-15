import torch
import torch.nn as nn
import torch.nn.functional as F
from ptflops import get_model_complexity_info
import einops
from einops import rearrange, repeat
from torch import Tensor
from torchvision.ops.misc import MLP, Permute
from torchvision.ops.stochastic_depth import StochasticDepth
from torchvision.transforms._presets import ImageClassification, InterpolationMode
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import WeightsEnum, Weights
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _ovewrite_named_param
from typing import Optional, Callable, List, Any
from torchinfo import summary


class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2):
        super(GhostModule, self).__init__()
        self.out_channels = out_channels
        self.primary_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.cheap_operation = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels, bias=False)
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.act = nn.PReLU()

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        out = self.bn(out)
        out = self.act(out)
        return out[:, :self.out_channels, :, :]

class eca_layer(nn.Module):
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)
    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)
        return output
class DSConv(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 9,
        extend_scope: float = 1.0,
        morph: int = 0,
        if_offset: bool = True,
    ):
        """
        A Dynamic Snake Convolution Implementation
        
        Args:
            in_channels: number of input channels
            out_channels: number of output channels
            kernel_size: the size of kernel
            extend_scope: the range to expand
            morph: 0 for x-axis, 1 for y-axis
            if_offset: whether deformation is required
        """
        super().__init__()

        if morph not in (0, 1):
            raise ValueError("morph should be 0 or 1.")

        self.extend_scope = extend_scope
        self.morph = morph
        self.if_offset = if_offset
        
        self.gn_offset = nn.GroupNorm(kernel_size, 2 * kernel_size)
        self.gn = nn.GroupNorm(out_channels // 4 if out_channels >= 4 else 1, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

        # Ensure input and output channel handling
        self.input_adjust = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.offset_conv = nn.Conv2d(out_channels, 2 * kernel_size, 3, padding=1)

        self.dsc_conv_x = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            stride=(kernel_size, 1),
            padding=0,
        )
        self.dsc_conv_y = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            stride=(1, kernel_size),
            padding=0,
        )

    def forward(self, input: torch.Tensor):
        # Adjust input channels
        x = self.input_adjust(input)
        
        if not self.if_offset:
            # Standard convolution if offset is disabled
            output = self.dsc_conv_y(x) if self.morph else self.dsc_conv_x(x)
            output = self.gn(output)
            return self.relu(output)
            
        # Predict offset map
        offset = self.offset_conv(x)
        offset = self.gn_offset(offset)
        offset = self.tanh(offset)  # Range [-1, 1]
        
        batch_size, _, width, height = offset.shape
        kernel_size = offset.shape[1] // 2
        center = kernel_size // 2
        device = input.device
        
        y_offset, x_offset = torch.split(offset, kernel_size, dim=1)
        
        y_center = torch.arange(0, width, dtype=torch.float32, device=device)
        y_center = einops.repeat(y_center, "w -> k w h", k=kernel_size, h=height)
        
        x_center = torch.arange(0, height, dtype=torch.float32, device=device)
        x_center = einops.repeat(x_center, "h -> k w h", k=kernel_size, w=width)
        
        if self.morph == 0:  # x-axis
            y_spread = torch.zeros([kernel_size], device=device)
            x_spread = torch.linspace(-center, center, kernel_size, device=device)
            
            y_grid = einops.repeat(y_spread, "k -> k w h", w=width, h=height)
            x_grid = einops.repeat(x_spread, "k -> k w h", w=width, h=height)
            
            y_new = y_center + y_grid
            x_new = x_center + x_grid
            
            y_new = einops.repeat(y_new, "k w h -> b k w h", b=batch_size)
            x_new = einops.repeat(x_new, "k w h -> b k w h", b=batch_size)
            
            y_offset = einops.rearrange(y_offset, "b k w h -> k b w h")
            y_offset_new = y_offset.detach().clone()
            
            y_offset_new[center] = 0
            
            for index in range(1, center + 1):
                y_offset_new[center + index] = (
                    y_offset_new[center + index - 1] + y_offset[center + index]
                )
                y_offset_new[center - index] = (
                    y_offset_new[center - index + 1] + y_offset[center - index]
                )
                
            y_offset_new = einops.rearrange(y_offset_new, "k b w h -> b k w h")
            
            y_new = y_new.add(y_offset_new.mul(self.extend_scope))
            
            y_coordinate_map = einops.rearrange(y_new, "b k w h -> b (w k) h")
            x_coordinate_map = einops.rearrange(x_new, "b k w h -> b (w k) h")
            
        else:  # y-axis
            y_spread = torch.linspace(-center, center, kernel_size, device=device)
            x_spread = torch.zeros([kernel_size], device=device)
            
            y_grid = einops.repeat(y_spread, "k -> k w h", w=width, h=height)
            x_grid = einops.repeat(x_spread, "k -> k w h", w=width, h=height)
            
            y_new = y_center + y_grid
            x_new = x_center + x_grid
            
            y_new = einops.repeat(y_new, "k w h -> b k w h", b=batch_size)
            x_new = einops.repeat(x_new, "k w h -> b k w h", b=batch_size)
            
            x_offset = einops.rearrange(x_offset, "b k w h -> k b w h")
            x_offset_new = x_offset.detach().clone()
            
            x_offset_new[center] = 0
            
            for index in range(1, center + 1):
                x_offset_new[center + index] = (
                    x_offset_new[center + index - 1] + x_offset[center + index]
                )
                x_offset_new[center - index] = (
                    x_offset_new[center - index + 1] + x_offset[center - index]
                )
                
            x_offset_new = einops.rearrange(x_offset_new, "k b w h -> b k w h")
            
            x_new = x_new.add(x_offset_new.mul(self.extend_scope))
            
            y_coordinate_map = einops.rearrange(y_new, "b k w h -> b w (h k)")
            x_coordinate_map = einops.rearrange(x_new, "b k w h -> b w (h k)")
        
        # Prepare for grid_sample
        y_max = input.shape[-2] - 1
        x_max = input.shape[-1] - 1
        
        y_coordinate_map = torch.clamp(y_coordinate_map / width * 2 - 1, -1, 1)
        x_coordinate_map = torch.clamp(x_coordinate_map / height * 2 - 1, -1, 1)
        
        y_coordinate_map = torch.unsqueeze(y_coordinate_map, dim=-1)
        x_coordinate_map = torch.unsqueeze(x_coordinate_map, dim=-1)
        
        grid = torch.cat([x_coordinate_map, y_coordinate_map], dim=-1)
        
        deformed_feature = F.grid_sample(
            input=x,
            grid=grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        
        output = self.dsc_conv_y(deformed_feature) if self.morph else self.dsc_conv_x(deformed_feature)
        
        # Apply normalization and activation
        output = self.gn(output)
        output = self.relu(output)
        
        return output

class FeaturePyramidEnhance(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.downsample_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, None)),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.PReLU(channels)
        )
        self.upsample_branch = nn.Sequential(
            nn.Upsample(scale_factor=1, mode='bilinear', align_corners=False),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.PReLU(channels)
        )

    def forward(self, x):
        down_feat = self.downsample_branch(x)
        up_feat = self.upsample_branch(x)
        return torch.cat([x,down_feat, up_feat], dim=1)

class FeatureInteractionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.reduce_dim = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels)
        )
        self.cross_interaction = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, groups=2),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels)
        )
        self.residual_enhance = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=2),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels)
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels)
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        interaction = self.reduce_dim(x)
        cross_feat = self.cross_interaction(interaction)
        enhanced_feat = cross_feat + self.residual_enhance(cross_feat)
        out = self.fusion(enhanced_feat)
        return out

class HybridDirectionalFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = in_channels // 4
        
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            BNPReLU(mid_channels)
        )
        
        self.ghost_initial = GhostModule(mid_channels, mid_channels)
        

        self.dsconv_x = DSConv(mid_channels, mid_channels, kernel_size=7, morph=0, if_offset=True)
        self.dsconv_y = DSConv(mid_channels, mid_channels, kernel_size=7, morph=1, if_offset=True)
        

        self.dilated_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, kernel_size=3, dilation=d, padding=d, groups=mid_channels),
                BNPReLU(mid_channels)
            ) for d in [2, 4, 6]
        ])
        self.zip_convs = nn.Sequential(
            nn.Conv2d(mid_channels*3, mid_channels, kernel_size=3, padding=1, groups=mid_channels),
            BNPReLU(mid_channels)
        )
    
        self.feature_enhancer = FeaturePyramidEnhance(mid_channels)
        
     
        self.feature_interaction = FeatureInteractionModule(
            in_channels=mid_channels * 2, 
            out_channels=mid_channels * 3
        )
        
       
        self.dynamic_fusion = nn.Sequential(
            nn.Conv2d(mid_channels * 6, out_channels, kernel_size=1),
            BNPReLU(out_channels)
        )
        
      
        self.attention_guide = eca_layer(out_channels)
        
      
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        
        
        x = self.initial_conv(x)
        
      
        ghost_feat = self.ghost_initial(x)
        
        ds_x_feat = self.dsconv_x(ghost_feat)
        ds_y_feat = self.dsconv_y(ghost_feat)

       
        interact_feat = self.feature_interaction(ds_x_feat, ds_y_feat)
        
        dilated_feats = [branch(ghost_feat) for branch in self.dilated_convs]
        dilated_feat = torch.cat(dilated_feats, dim=1)
        dilated_feat = self.zip_convs(dilated_feat)
       
        enhanced_feat = self.feature_enhancer(dilated_feat)
       
        #print(f"enhanced_feat shape: {enhanced_feat.shape}")
        
        #print(f"interact_feat shape: {interact_feat.shape}")
        fused_feat = torch.cat([interact_feat, enhanced_feat], dim=1)
        fused_feat = self.dynamic_fusion(fused_feat)
        
       
        fused_feat = self.attention_guide(fused_feat)
        
        
        out = fused_feat + residual

        return out
