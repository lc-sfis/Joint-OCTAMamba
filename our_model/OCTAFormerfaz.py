# filename: OCTASwin_Medium_Channel.py (v4 - Final Alignment with LSA)
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import math
from einops import rearrange, repeat
import time
from functools import partial
from typing import Optional, Callable, List, Tuple
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import Tensor
import sys
import os

# --- 从 OCTAMamba 继承的辅助模块 ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from HDFE import HybridDirectionalFeatureExtractor
from AdaptiveFeatureFusion import SimplifiedAttentionalFeatureFusion
# [!!! 新增 !!!] 导入 LSA
try:
    from .lsa import LSA
except ImportError:
    from lsa import LSA
from torchvision.ops.misc import MLP
from torchvision.ops.stochastic_depth import StochasticDepth

class ShiftedWindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: List[int], shift_size: List[int], num_heads: int, qkv_bias: bool = True, proj_bias: bool = True, attention_dropout: float = 0.0, dropout: float = 0.0):
        super().__init__()
        if len(window_size) != 2 or len(shift_size) != 2:
            raise ValueError("window_size and shift_size must be of length 2")
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1).view(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        with torch.no_grad():
            nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
    def forward(self, x: Tensor) -> Tensor:
        B, H, W, C = x.shape
        pad_r = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_b = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        x_padded = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, pad_H, pad_W, _ = x_padded.shape
        shift_size = self.shift_size
        if self.window_size[0] >= pad_H: shift_size = [0, shift_size[1]]
        if self.window_size[1] >= pad_W: shift_size = [shift_size[0], 0]
        if sum(shift_size) > 0:
            x_shifted = torch.roll(x_padded, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
        else:
            x_shifted = x_padded
        num_windows = (pad_H // self.window_size[0]) * (pad_W // self.window_size[1])
        x_windows = x_shifted.view(B, pad_H // self.window_size[0], self.window_size[0], pad_W // self.window_size[1], self.window_size[1], C)
        x_windows = x_windows.permute(0, 1, 3, 2, 4, 5).reshape(B * num_windows, self.window_size[0] * self.window_size[1], C)
        qkv = self.qkv(x_windows).reshape(x_windows.size(0), x_windows.size(1), 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * ((C // self.num_heads) ** -0.5)
        attn = q @ k.transpose(-2, -1)
        N = self.window_size[0] * self.window_size[1]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index].view(N, N, -1).permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if sum(shift_size) > 0:
            attn_mask = x.new_zeros((pad_H, pad_W))
            h_slices = (slice(0, -self.window_size[0]), slice(-self.window_size[0], -shift_size[0]), slice(-shift_size[0], None))
            w_slices = (slice(0, -self.window_size[1]), slice(-self.window_size[1], -shift_size[1]), slice(-shift_size[1], None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    attn_mask[h, w] = cnt
                    cnt += 1
            mask_windows = attn_mask.view(pad_H // self.window_size[0], self.window_size[0], pad_W // self.window_size[1], self.window_size[1])
            mask_windows = mask_windows.permute(0, 2, 1, 3).reshape(num_windows, self.window_size[0] * self.window_size[1])
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
            attn = attn.view(x_windows.size(0) // num_windows, num_windows, self.num_heads, x_windows.size(1), x_windows.size(1)) + attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, x_windows.size(1), x_windows.size(1))
        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.attention_dropout)
        x = (attn @ v).transpose(1, 2).reshape(x_windows.size(0), x_windows.size(1), C)
        x = self.proj(x)
        x = F.dropout(x, p=self.dropout)
        x = x.view(B, pad_H // self.window_size[0], pad_W // self.window_size[1], self.window_size[0], self.window_size[1], C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, pad_H, pad_W, C)
        if sum(shift_size) > 0:
            x = torch.roll(x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
        x = x[:, :H, :W, :].contiguous()
        return x
class SwinTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: List[int], shift_size: List[int], mlp_ratio: float = 4.0, dropout: float = 0.0, attention_dropout: float = 0.0, stochastic_depth_prob: float = 0.0, norm_layer: Callable[..., nn.Module] = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = ShiftedWindowAttention(dim, window_size, shift_size, num_heads, attention_dropout=attention_dropout, dropout=dropout)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, [mlp_hidden_dim, dim], activation_layer=nn.GELU, dropout=dropout)
    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.stochastic_depth(x)
        x = identity + x
        identity = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.stochastic_depth(x)
        x = identity + x
        return x

# --- [!!! 核心修改 !!!] Swin VSSBlock, 对齐了 Mamba 的 LSA ---
class SwinTransformerVSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            window_size: int = 7,
            is_shift: bool = True,
            expand: int = 2,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.inner_dim = int(hidden_dim * expand)
        self.in_proj = nn.Linear(hidden_dim, self.inner_dim * 2, bias=False)
       # self.dual_att = VesselAwareDAM(self.inner_dim, reduction=max(4, self.inner_dim // 16))
        self.swin_block = SwinTransformerBlock(
            dim=self.inner_dim,
            num_heads=max(1, self.inner_dim // 32),
            window_size=[window_size, window_size],
            shift_size=[window_size // 2 if is_shift else 0, window_size // 2 if is_shift else 0],
            stochastic_depth_prob=drop_path
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.out_proj = nn.Linear(self.inner_dim, hidden_dim, bias=False)
        
        reduction = max(hidden_dim // 8, 4)
        self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(hidden_dim, reduction, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduction, hidden_dim, 1),
                nn.Sigmoid()
            )
        
        # [!!! 对齐Mamba !!!] 新增 LSA 正则化
        self.lsa = LSA(p=0.85) if hidden_dim >= 32 else None

    def forward(self, input: torch.Tensor):
        x = self.ln_1(input)
        x_for_swin, z_for_dam = self.in_proj(x).chunk(2, dim=-1)
        swin_out = self.swin_block(x_for_swin)
        # z_for_dam_perm = z_for_dam.permute(0, 3, 1, 2).contiguous()
        # dam_out = self.dual_att(z_for_dam_perm)
        # dam_out_perm = dam_out.permute(0, 2, 3, 1).contiguous()
        fused_out = swin_out * F.silu(z_for_dam)
        final_out = self.out_proj(fused_out)
        x = input + self.drop_path(final_out)
        
        x_perm = x.permute(0, 3, 1, 2)
        att_weight = self.channel_attention(x_perm)
        x_perm = x_perm * att_weight
        x = x_perm.permute(0, 2, 3, 1)
        
        # [!!! 对齐Mamba !!!] 应用 LSA 正则化
        if self.lsa is not None:
            B, H, W, C = x.shape
            x_reshape = x.permute(0, 3, 1, 2).contiguous().view(B, C, -1)  # B C L
            x_reshape = self.lsa(x_reshape)
            x = x_reshape.view(B, C, H, W).permute(0, 2, 3, 1).contiguous()
            
        return x

class CompactFAZEnhancement(nn.Module):
    """紧凑型FAZ增强模块，突出中心无血管区域和边界感知"""
    def __init__(self, in_channels, use_mask=True):
        super().__init__()
        # 环形/中心感知卷积
        self.annular_conv = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, dilation=2, groups=in_channels)
        # SEBlock可加强对“区域”的全局感知
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels//8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//8, in_channels, 1),
            nn.Sigmoid()
        )
        self.use_mask = use_mask
    def forward(self, x):
        x0 = x
        annular = self.annular_conv(x)
        se = self.se(annular) * annular

        if self.use_mask:
            b, c, h, w = se.size()
            center_h = h // 2
            center_w = w // 2
            radius = min(h, w) // 4
            y, x_coord = torch.meshgrid(
                torch.arange(h, device=x.device),
                torch.arange(w, device=x.device),
                indexing='ij'
            )
            mask = (((x_coord - center_w) ** 2 + (y - center_h) ** 2) < (radius ** 2)).float()
            mask = mask.unsqueeze(0).unsqueeze(0)
            se = se * (1 + mask)

        # --- 对齐se和x0的空间尺寸 ---
        if se.shape[2:] != x0.shape[2:]:
            se = F.interpolate(se, size=x0.shape[2:], mode='bilinear', align_corners=False)

        return x0 + se * 0.3
class MediumChannelOCTASwinBlock(nn.Module):
    def __init__(self, in_c, out_c, is_shift=True):
        super().__init__()
        self.conv = HybridDirectionalFeatureExtractor(in_c, out_c)
        self.faz_enhancement = CompactFAZEnhancement(out_c)
        self.ln = nn.LayerNorm(out_c)
        self.act = nn.GELU()
        self.block = SwinTransformerVSSBlock(hidden_dim=out_c, drop_path=0.1, is_shift=is_shift)
        self.residual_conv = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        skip = self.residual_conv(x)
        x = self.conv(x)
        x = self.faz_enhancement(x)
        x_perm = x.permute(0, 2, 3, 1)
        x_perm = self.block(x_perm)
        x = x_perm.permute(0, 3, 1, 2)
        x_perm = x.permute(0, 2, 3, 1)
        x_perm = self.act(self.ln(x_perm))
        x = x_perm.permute(0, 3, 1, 2)
        return x + skip * self.scale
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, max(channel // reduction, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(channel // reduction, 1), channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
class CompactQSEME(nn.Module):
    def __init__(self, out_c=16):
        super().__init__()
        init_channels = 32
        self.init_conv = nn.Sequential(nn.Conv2d(1, init_channels, kernel_size=3, padding=1), nn.BatchNorm2d(init_channels), nn.ReLU())
        branch_channels = init_channels // 4
        self.branch1 = nn.Sequential(nn.MaxPool2d(2), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.branch2 = nn.Sequential(nn.AvgPool2d(2), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.branch3 = nn.Sequential(nn.Conv2d(branch_channels, branch_channels, 3, padding=1, groups=branch_channels), nn.Conv2d(branch_channels, branch_channels, 1), nn.BatchNorm2d(branch_channels), nn.ReLU())
        self.branch4 = SEBlock(branch_channels, reduction=4)
        self.fusion = nn.Sequential(nn.Conv2d(init_channels, out_c, 1), nn.BatchNorm2d(out_c), nn.ReLU())
    def forward(self, x):
        x = self.init_conv(x)
        x1, x2, x3, x4 = x.chunk(4, dim=1)
        fused = torch.cat([self.branch1(x1), self.branch2(x2), self.branch3(x3), self.branch4(x4)], dim=1)
        return self.fusion(fused)
class MediumChannelEncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, is_shift=True):
        super().__init__()
        self.octaswin = MediumChannelOCTASwinBlock(in_c, out_c, is_shift=is_shift)
        self.se = SEBlock(out_c, reduction=8)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.GELU()
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        x = self.octaswin(x)
        x = self.se(x)
        skip = self.act(self.bn(x))
        x = self.down(skip)
        return x, skip
class MediumChannelDecoderBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c, use_advanced_fusion=False, is_shift=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.feature_fusion = SimplifiedAttentionalFeatureFusion(skip_channels=skip_c, up_channels=in_c, out_channels=in_c+skip_c)
        self.bn2 = nn.BatchNorm2d(in_c + skip_c)
        self.octaswin = MediumChannelOCTASwinBlock(in_c + skip_c, out_c, is_shift=is_shift)
        self.act = nn.ReLU()
    def forward(self, x, skip):
        x = self.up(x)
        x = self.feature_fusion(skip, x)
        x = self.act(self.bn2(x))
        x = self.octaswin(x)
        return x
class OCTASwin_Medium_Channel(nn.Module):
    def __init__(self, qseme_type='conservative'):
        super().__init__()
        print("✅ Initializing OCTA-Swin Medium Channel Model (v4 - Final Alignment with Mamba's internal flow + LSA)")
        self.qseme = CompactQSEME(out_c=16)
        self.e1 = MediumChannelEncoderBlock(16, 32, is_shift=False)
        self.e2 = MediumChannelEncoderBlock(32, 64, is_shift=True)
        self.e3 = MediumChannelEncoderBlock(64, 128, is_shift=False)
        self.bottleneck = nn.Sequential(
            MediumChannelOCTASwinBlock(128, 256, is_shift=True),
            nn.Dropout2d(0.1)
        )
        self.d3 = MediumChannelDecoderBlock(256, 128, 128, is_shift=False)
        self.d2 = MediumChannelDecoderBlock(128, 64, 64, is_shift=True)
        self.d1 = MediumChannelDecoderBlock(64, 32, 32, is_shift=False)
        self.deep_supervision = nn.ModuleList([
            nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, 1, 1)),
            nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Conv2d(32, 1, 1))
        ])
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, return_deep_supervision=False):
        input_size = x.shape[-2:]
        x = self.qseme(x)
        x, skip1 = self.e1(x)
        x, skip2 = self.e2(x)
        x, skip3 = self.e3(x)
        x = self.bottleneck(x)
        x = self.d3(x, skip3)
        if return_deep_supervision:
            deep_out2 = self.deep_supervision[0](x)
        x = self.d2(x, skip2)
        if return_deep_supervision:
            deep_out1 = self.deep_supervision[1](x)
        x = self.d1(x, skip1)
        final_out = self.final_conv(x)
        final_out = self.sigmoid(final_out)
        if return_deep_supervision:
            deep_out1 = F.interpolate(deep_out1, size=input_size, mode='bilinear', align_corners=True)
            deep_out2 = F.interpolate(deep_out2, size=input_size, mode='bilinear', align_corners=True)
            return final_out, [self.sigmoid(deep_out1), self.sigmoid(deep_out2)]
        else:
            return final_out

# import torch
# import torch.nn as nn
# import torch.utils.checkpoint as checkpoint
# import math
# from einops import rearrange, repeat
# import time
# from functools import partial
# from typing import Optional, Callable, List, Tuple
# import torch.nn.functional as F
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from torch import Tensor
# import sys
# import os

# # --- 从 OCTAMamba 继承的辅助模块 ---
# # 为了独立运行，我们将尝试直接导入，如果失败则说明需要正确的包结构
# from MDR import MultiScaleConvModule
# from VesselAwareDAM import VesselAwareDAM
# from AdaptiveFeatureFusion import SimplifiedAdaptiveFusion
# from lsa import LSA



# # --- 从 SwinSnake 中引入的 Swin Transformer 核心模块 ---
# from torchvision.ops.misc import MLP
# from torchvision.ops.stochastic_depth import StochasticDepth

# class ShiftedWindowAttention(nn.Module):
#     def __init__(self, dim: int, window_size: List[int], shift_size: List[int], num_heads: int, qkv_bias: bool = True, proj_bias: bool = True, attention_dropout: float = 0.0, dropout: float = 0.0):
#         super().__init__()
#         if len(window_size) != 2 or len(shift_size) != 2:
#             raise ValueError("window_size and shift_size must be of length 2")
#         self.window_size = window_size
#         self.shift_size = shift_size
#         self.num_heads = num_heads
#         self.attention_dropout = attention_dropout
#         self.dropout = dropout
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.proj = nn.Linear(dim, dim, bias=proj_bias)
#         self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
#         coords_h = torch.arange(self.window_size[0])
#         coords_w = torch.arange(self.window_size[1])
#         coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
#         coords_flatten = torch.flatten(coords, 1)
#         relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
#         relative_coords = relative_coords.permute(1, 2, 0).contiguous()
#         relative_coords[:, :, 0] += self.window_size[0] - 1
#         relative_coords[:, :, 1] += self.window_size[1] - 1
#         relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
#         relative_position_index = relative_coords.sum(-1).view(-1)
#         self.register_buffer("relative_position_index", relative_position_index)
#         with torch.no_grad():
#             nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
#     def forward(self, x: Tensor) -> Tensor:
#         B, H, W, C = x.shape
#         pad_r = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
#         pad_b = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
#         x_padded = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
#         _, pad_H, pad_W, _ = x_padded.shape
#         shift_size = self.shift_size
#         if self.window_size[0] >= pad_H: shift_size = [0, shift_size[1]]
#         if self.window_size[1] >= pad_W: shift_size = [shift_size[0], 0]
#         if sum(shift_size) > 0:
#             x_shifted = torch.roll(x_padded, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
#         else:
#             x_shifted = x_padded
#         num_windows = (pad_H // self.window_size[0]) * (pad_W // self.window_size[1])
#         x_windows = x_shifted.view(B, pad_H // self.window_size[0], self.window_size[0], pad_W // self.window_size[1], self.window_size[1], C)
#         x_windows = x_windows.permute(0, 1, 3, 2, 4, 5).reshape(B * num_windows, self.window_size[0] * self.window_size[1], C)
#         qkv = self.qkv(x_windows).reshape(x_windows.size(0), x_windows.size(1), 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]
#         q = q * ((C // self.num_heads) ** -0.5)
#         attn = q @ k.transpose(-2, -1)
#         N = self.window_size[0] * self.window_size[1]
#         relative_position_bias = self.relative_position_bias_table[self.relative_position_index].view(N, N, -1).permute(2, 0, 1).contiguous()
#         attn = attn + relative_position_bias.unsqueeze(0)
#         if sum(shift_size) > 0:
#             attn_mask = x.new_zeros((pad_H, pad_W))
#             h_slices = (slice(0, -self.window_size[0]), slice(-self.window_size[0], -shift_size[0]), slice(-shift_size[0], None))
#             w_slices = (slice(0, -self.window_size[1]), slice(-self.window_size[1], -shift_size[1]), slice(-shift_size[1], None))
#             cnt = 0
#             for h in h_slices:
#                 for w in w_slices:
#                     attn_mask[h, w] = cnt
#                     cnt += 1
#             mask_windows = attn_mask.view(pad_H // self.window_size[0], self.window_size[0], pad_W // self.window_size[1], self.window_size[1])
#             mask_windows = mask_windows.permute(0, 2, 1, 3).reshape(num_windows, self.window_size[0] * self.window_size[1])
#             attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
#             attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
#             attn = attn.view(x_windows.size(0) // num_windows, num_windows, self.num_heads, x_windows.size(1), x_windows.size(1)) + attn_mask.unsqueeze(1).unsqueeze(0)
#             attn = attn.view(-1, self.num_heads, x_windows.size(1), x_windows.size(1))
#         attn = F.softmax(attn, dim=-1)
#         attn = F.dropout(attn, p=self.attention_dropout)
#         x = (attn @ v).transpose(1, 2).reshape(x_windows.size(0), x_windows.size(1), C)
#         x = self.proj(x)
#         x = F.dropout(x, p=self.dropout)
#         x = x.view(B, pad_H // self.window_size[0], pad_W // self.window_size[1], self.window_size[0], self.window_size[1], C)
#         x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, pad_H, pad_W, C)
#         if sum(shift_size) > 0:
#             x = torch.roll(x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
#         x = x[:, :H, :W, :].contiguous()
#         return x

# class SwinTransformerBlock(nn.Module):
#     def __init__(self, dim: int, num_heads: int, window_size: List[int], shift_size: List[int], mlp_ratio: float = 4.0, dropout: float = 0.0, attention_dropout: float = 0.0, stochastic_depth_prob: float = 0.0, norm_layer: Callable[..., nn.Module] = nn.LayerNorm):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = ShiftedWindowAttention(dim, window_size, shift_size, num_heads, attention_dropout=attention_dropout, dropout=dropout)
#         self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = MLP(dim, [mlp_hidden_dim, dim], activation_layer=nn.GELU, dropout=dropout)
#     def forward(self, x: Tensor) -> Tensor:
#         identity = x
#         x = self.norm1(x)
#         x = self.attn(x)
#         x = self.stochastic_depth(x)
#         x = identity + x
#         identity = x
#         x = self.norm2(x)
#         x = self.mlp(x)
#         x = self.stochastic_depth(x)
#         x = identity + x
#         return x

# # --- [!!! 核心修改 !!!] Swin VSSBlock, 对齐了 Mamba 的 LSA ---
# class SwinTransformerVSSBlock(nn.Module):
#     def __init__(
#             self,
#             hidden_dim: int = 0,
#             drop_path: float = 0,
#             norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
#             window_size: int = 7,
#             is_shift: bool = True,
#             expand: int = 2,
#             **kwargs,
#     ):
#         super().__init__()
#         self.ln_1 = norm_layer(hidden_dim)
#         self.inner_dim = int(hidden_dim * expand)
#         self.in_proj = nn.Linear(hidden_dim, self.inner_dim * 2, bias=False)
#         self.swin_block = SwinTransformerBlock(
#             dim=self.inner_dim,
#             num_heads=max(1, self.inner_dim // 32),
#             window_size=[window_size, window_size],
#             shift_size=[window_size // 2 if is_shift else 0, window_size // 2 if is_shift else 0],
#             stochastic_depth_prob=drop_path
#         )
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.out_proj = nn.Linear(self.inner_dim, hidden_dim, bias=False)
        
#         reduction = max(hidden_dim // 8, 4)
#         self.channel_attention = nn.Sequential(
#                 nn.AdaptiveAvgPool2d(1),
#                 nn.Conv2d(hidden_dim, reduction, 1),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(reduction, hidden_dim, 1),
#                 nn.Sigmoid()
#             )
        
#         self.lsa = LSA(p=0.85) if hidden_dim >= 32 else nn.Identity()

#     def forward(self, input: torch.Tensor):
#         x = self.ln_1(input)
#         x_for_swin, z_for_dam = self.in_proj(x).chunk(2, dim=-1)
#         swin_out = self.swin_block(x_for_swin)
#         fused_out = swin_out * F.silu(z_for_dam)
#         final_out = self.out_proj(fused_out)
#         x = input + self.drop_path(final_out)
        
#         x_perm = x.permute(0, 3, 1, 2)
#         att_weight = self.channel_attention(x_perm)
#         x_perm = x_perm * att_weight
#         x = x_perm.permute(0, 2, 3, 1)
        
#         if self.lsa is not nn.Identity:
#             B, H, W, C = x.shape
#             x_reshape = x.permute(0, 3, 1, 2).contiguous().view(B, C, -1)  # B C L
#             x_reshape = self.lsa(x_reshape)
#             x = x_reshape.view(B, C, H, W).permute(0, 2, 3, 1).contiguous()
            
#         return x

# class CompactFAZEnhancement(nn.Module):
#     def __init__(self, in_channels, use_mask=True):
#         super().__init__()
#         self.annular_conv = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, dilation=2, groups=in_channels)
#         self.se = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(in_channels, in_channels//8, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels//8, in_channels, 1),
#             nn.Sigmoid()
#         )
#         self.use_mask = use_mask
#     def forward(self, x):
#         x0 = x
#         annular = self.annular_conv(x)
#         se = self.se(annular) * annular

#         if self.use_mask:
#             b, c, h, w = se.size()
#             center_h = h // 2
#             center_w = w // 2
#             radius = min(h, w) // 4
#             y, x_coord = torch.meshgrid(
#                 torch.arange(h, device=x.device),
#                 torch.arange(w, device=x.device),
#                 indexing='ij'
#             )
#             mask = (((x_coord - center_w) ** 2 + (y - center_h) ** 2) < (radius ** 2)).float()
#             mask = mask.unsqueeze(0).unsqueeze(0)
#             se = se * (1 + mask)
        
#         if se.shape[2:] != x0.shape[2:]:
#             se = F.interpolate(se, size=x0.shape[2:], mode='bilinear', align_corners=False)

#         return x0 + se * 0.3

# class MediumChannelOCTASwinBlock(nn.Module):
#     def __init__(self, in_c, out_c, is_shift=True):
#         super().__init__()
#         # 使用 DummyModule 替换 MultiScaleConvModule
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_c),
#             nn.ReLU(inplace=True)
#         )
#         self.faz_enhancement = CompactFAZEnhancement(out_c)
#         self.ln = nn.LayerNorm(out_c)
#         self.act = nn.GELU()
#         self.block = SwinTransformerVSSBlock(hidden_dim=out_c, drop_path=0.1, is_shift=is_shift)
#         self.residual_conv = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1)
#         self.scale = nn.Parameter(torch.ones(1))

#     def forward(self, x):
#         skip = self.residual_conv(x)
#         x = self.conv(x)
#         x = self.faz_enhancement(x)
#         x_perm = x.permute(0, 2, 3, 1)
#         x_perm = self.block(x_perm)
#         x = x_perm.permute(0, 3, 1, 2)
#         x_perm = x.permute(0, 2, 3, 1)
#         x_perm = self.act(self.ln(x_perm))
#         x = x_perm.permute(0, 3, 1, 2)
#         return x + skip * self.scale

# class SEBlock(nn.Module):
#     def __init__(self, channel, reduction=8):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, max(channel // reduction, 1), bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(max(channel // reduction, 1), channel, bias=False),
#             nn.Sigmoid()
#         )
#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y.expand_as(x)

# class CompactQSEME(nn.Module):
#     def __init__(self, out_c=16):
#         super().__init__()
#         init_channels = 32
#         self.init_conv = nn.Sequential(nn.Conv2d(1, init_channels, kernel_size=3, padding=1), nn.BatchNorm2d(init_channels), nn.ReLU())
#         branch_channels = init_channels // 4
#         self.branch1 = nn.Sequential(nn.MaxPool2d(2), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
#         self.branch2 = nn.Sequential(nn.AvgPool2d(2), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
#         self.branch3 = nn.Sequential(nn.Conv2d(branch_channels, branch_channels, 3, padding=1, groups=branch_channels), nn.Conv2d(branch_channels, branch_channels, 1), nn.BatchNorm2d(branch_channels), nn.ReLU())
#         self.branch4 = SEBlock(branch_channels, reduction=4)
#         self.fusion = nn.Sequential(nn.Conv2d(init_channels, out_c, 1), nn.BatchNorm2d(out_c), nn.ReLU())
#     def forward(self, x):
#         x = self.init_conv(x)
#         x1, x2, x3, x4 = x.chunk(4, dim=1)
#         fused = torch.cat([self.branch1(x1), self.branch2(x2), self.branch3(x3), self.branch4(x4)], dim=1)
#         return self.fusion(fused)

# class MediumChannelEncoderBlock(nn.Module):
#     def __init__(self, in_c, out_c, is_shift=True):
#         super().__init__()
#         self.octaswin = MediumChannelOCTASwinBlock(in_c, out_c, is_shift=is_shift)
#         self.se = SEBlock(out_c, reduction=8)
#         self.bn = nn.BatchNorm2d(out_c)
#         self.act = nn.GELU()
#         self.down = nn.MaxPool2d(kernel_size=2, stride=2)
#     def forward(self, x):
#         x = self.octaswin(x)
#         x = self.se(x)
#         skip = self.act(self.bn(x))
#         x = self.down(skip)
#         return x, skip

# class MediumChannelDecoderBlock(nn.Module):
#     def __init__(self, in_c, skip_c, out_c, use_advanced_fusion=False, is_shift=True):
#         super().__init__()
#         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         # 使用 DummyFusion 替换
#         #self.feature_fusion = DummyFusion(skip_channels=skip_c, up_channels=in_c, out_channels=in_c+skip_c)
#         self.bn2 = nn.BatchNorm2d(in_c + skip_c)
#         self.octaswin = MediumChannelOCTASwinBlock(in_c + skip_c, out_c, is_shift=is_shift)
#         self.act = nn.ReLU()
#     def forward(self, x, skip):
#         x = self.up(x)
#         x = self.feature_fusion(skip, x)
#         x = self.act(self.bn2(x))
#         x = self.octaswin(x)
#         return x

# class OCTASwin_Medium_Channel(nn.Module):
#     def __init__(self, qseme_type='conservative'):
#         super().__init__()
#         print("✅ Initializing OCTA-Swin Medium Channel Model (v4 - Final Alignment with Mamba's internal flow + LSA)")
#         self.qseme = CompactQSEME(out_c=16)
#         self.e1 = MediumChannelEncoderBlock(16, 32, is_shift=False)
#         self.e2 = MediumChannelEncoderBlock(32, 64, is_shift=True)
#         self.e3 = MediumChannelEncoderBlock(64, 128, is_shift=False)
#         self.bottleneck = nn.Sequential(
#             MediumChannelOCTASwinBlock(128, 256, is_shift=True),
#             nn.Dropout2d(0.1),
#         )
#         self.d3 = MediumChannelDecoderBlock(256, 128, 128, is_shift=False)
#         self.d2 = MediumChannelDecoderBlock(128, 64, 64, is_shift=True)
#         self.d1 = MediumChannelDecoderBlock(64, 32, 32, is_shift=False)
#         self.deep_supervision = nn.ModuleList([
#             nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, 1, 1)),
#             nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Conv2d(32, 1, 1))
#         ])
#         self.final_conv = nn.Sequential(
#             nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
#             nn.Conv2d(32, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
#             nn.Conv2d(16, 1, 1)
#         )
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x, return_deep_supervision=False):
#         input_size = x.shape[-2:]
#         x = self.qseme(x)
#         x, skip1 = self.e1(x)
#         x, skip2 = self.e2(x)
#         x, skip3 = self.e3(x)
#         x = self.bottleneck(x)
#         x = self.d3(x, skip3)
        
#         deep_outputs = []
#         if return_deep_supervision:
#             deep_out2 = self.deep_supervision[0](x)
#             deep_outputs.append(deep_out2)

#         x = self.d2(x, skip2)
#         if return_deep_supervision:
#             deep_out1 = self.deep_supervision[1](x)
#             deep_outputs.append(deep_out1)

#         x = self.d1(x, skip1)
#         final_out = self.final_conv(x)
        
#         # Sigmoid is applied to all outputs at the end
#         final_out = self.sigmoid(final_out)
        
#         if return_deep_supervision:
#             # Reverse the list to have them from shallow to deep and apply sigmoid/interpolation
#             processed_deep_outputs = [
#                 self.sigmoid(F.interpolate(out, size=input_size, mode='bilinear', align_corners=True))
#                 for out in reversed(deep_outputs)
#             ]
#             return final_out, processed_deep_outputs
#         else:
#             return final_out

def count_parameters(model):
    """计算模型的总参数量和可训练参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

# [!!! 新增 !!!] Main function for testing
if __name__ == "__main__":
    # 实例化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OCTASwin_Medium_Channel().to(device)
    model.eval() # or model.train()

    # 计算并打印参数量
    total_params, trainable_params = count_parameters(model)
    print("\n" + "="*50)
    print(f"Model: {model.__class__.__name__}")
    print(f"  - Total Parameters:     {total_params/1e6:.2f} M")
    print(f"  - Trainable Parameters: {trainable_params/1e6:.2f} M")
    print("="*50 + "\n")

    # 创建一个模拟输入张量
    # B, C, H, W -> 1, 1, 256, 256
    dummy_input = torch.randn(1, 1, 256, 256).to(device)

    # 测试前向传播
    try:
        print("Testing forward pass without deep supervision...")
        start_time = time.time()
        output = model(dummy_input, return_deep_supervision=False)
        end_time = time.time()
        print(f"  - Output shape: {output.shape}")
        print(f"  - Forward pass successful in {end_time - start_time:.4f} seconds.")

        print("\nTesting forward pass with deep supervision...")
        start_time = time.time()
        final_output, deep_outputs = model(dummy_input, return_deep_supervision=True)
        end_time = time.time()
        print(f"  - Final Output shape: {final_output.shape}")
        for i, deep_out in enumerate(deep_outputs):
            print(f"  - Deep Supervision Output {i+1} shape: {deep_out.shape}")
        print(f"  - Forward pass successful in {end_time - start_time:.4f} seconds.")

    except Exception as e:
        print(f"\n[ERROR] An error occurred during the forward pass: {e}")
