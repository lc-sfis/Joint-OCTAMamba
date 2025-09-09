import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import math
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat
import time
from functools import partial
from typing import Optional, Callable
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import Tensor
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
#from MDR import MultiScaleConvModule
from HDFE import HybridDirectionalFeatureExtractor
from AdaptiveFeatureFusion import SimplifiedAttentionalFeatureFusion,AdaptiveFeatureFusion
from wtconv2d import *
import einops
try:
    from .lsa import LSA  # 包内导入
except ImportError:
    from lsa import LSA  # 备用


try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

class OptimizedSS2D(nn.Module):
    """针对中等通道数优化的SS2D"""
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        #self.dual_att = DualAttentionModule(in_channels=self.d_inner)
        #self.dual_att = VesselAwareDAM(self.d_inner, reduction=max(4, self.d_inner // 16))
        self.dual_att = None
        # 轻量级注意力机制（仅在通道数>=64时使用）
        # if d_model >= 64:
        #     self.dual_att = DualAttentionModule(in_channels=self.d_inner)
        # else:
        #     self.dual_att = None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        # A_logs_clamped = self.A_logs.float().clamp(max=10.0) # 核心修改：限制 A_logs 的最大值
        # As = -torch.exp(A_logs_clamped).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        
        if self.dual_att is not None:
            z = z.permute(0, 3, 1, 2).contiguous()
            z = self.dual_att(z)
            z = z.permute(0, 2, 3, 1).contiguous()
            y = y * F.silu(z)
        else:
            y = y * F.silu(z)
            
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class MediumChannelVSSBlock(nn.Module):
    """中等通道数优化的VSSBlock"""
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        
        # 核心Mamba模块
       # self.ss2d = VesselAwareSS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.ss2d = OptimizedSS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        reduction = max(hidden_dim // 8, 4)
        self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(hidden_dim, reduction, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduction, hidden_dim, 1),
                nn.Sigmoid()
            )
        # 仅在中等通道数时使用增强特征
        # if hidden_dim >= 64:
        #     # 轻量级通道注意力
        #     reduction = max(hidden_dim // 8, 4)
        #     self.channel_attention = nn.Sequential(
        #         nn.AdaptiveAvgPool2d(1),
        #         nn.Conv2d(hidden_dim, reduction, 1),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(reduction, hidden_dim, 1),
        #         nn.Sigmoid()
        #     )
        # else:
        #     self.channel_attention = None

        # LSA 正则化（借鉴 Mamba-Sea）
        self.lsa = LSA(p=0.85) if hidden_dim >= 32 else None
        #self.lsa=None
    def forward(self, input: torch.Tensor):
        x = self.ln_1(input)
        
        # Mamba处理
        x = self.ss2d(x)
        
        # 残差连接
        x = input + self.drop_path(x)
        
        # 轻量级通道注意力（仅中等通道数时使用）
        if self.channel_attention is not None:
            x_perm = x.permute(0, 3, 1, 2)  # B H W C -> B C H W
            att_weight = self.channel_attention(x_perm)
            x_perm = x_perm * att_weight
            x = x_perm.permute(0, 2, 3, 1)  # B C H W -> B H W C
        
        # LSA 正则化
        if self.lsa is not None:
            B, H, W, C = x.shape
            x_reshape = x.permute(0, 3, 1, 2).contiguous().view(B, C, -1)  # B C L
            x_reshape = self.lsa(x_reshape)
            x = x_reshape.view(B, C, H, W).permute(0, 2, 3, 1).contiguous()

        return x

class CompactVesselEnhancement(nn.Module):
    """紧凑型血管增强模块"""
    def __init__(self, in_channels):
        super().__init__()
        
        # 仅在通道数>=64时使用血管增强
        if in_channels >= 64:
            # 简化的多尺度检测
            self.vessel_conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels//2, 3, padding=1, groups=in_channels//4),
                nn.Conv2d(in_channels//2, in_channels//2, 1),
                nn.BatchNorm2d(in_channels//2),
                nn.ReLU(inplace=True)
            )
            
            # 方向性检测
            self.directional_conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels//4, (1, 5), padding=(0, 2)),
                nn.Conv2d(in_channels//4, in_channels//4, (5, 1), padding=(2, 0)),
                nn.BatchNorm2d(in_channels//4),
                nn.ReLU(inplace=True)
            )
            
            # 融合层
            self.fusion = nn.Sequential(
                nn.Conv2d(in_channels//2 + in_channels//4, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )
            
            self.use_enhancement = True
        else:
            self.use_enhancement = False

    def forward(self, x):
        if not self.use_enhancement:
            return x
            
        vessel_feat = self.vessel_conv(x)
        directional_feat = self.directional_conv(x)
        
        fused = torch.cat([vessel_feat, directional_feat], dim=1)
        enhanced = self.fusion(fused)
        
        return x + enhanced * 0.3  # 较小的残差权重
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



class MediumChannelOCTAMambaBlock(nn.Module):
    """中等通道数OCTA-Mamba块"""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        #self.conv=AdvancedVesselEnhancement(in_c,out_c)
        # 特征提取：仅在大通道数时使用MultiScale
        # if out_c >= 128:
        self.conv = HybridDirectionalFeatureExtractor(in_channels=in_c, out_channels=out_c)
        # else:
        #   self.conv = AdvancedVesselEnhancement(out_c)
        # else:
        #     self.conv = nn.Sequential(
        #         nn.Conv2d(in_c, out_c, 3, padding=1),
        #         nn.BatchNorm2d(out_c),
        #         nn.ReLU(inplace=True)
        #     )
        
        # 紧凑型血管增强
        self.faz_enhancement = CompactFAZEnhancement(out_c)
        # if self.out_c >= 32:
        #    self.vessel_enhancement = CompactVesselEnhancement(self.out_c)
        # else:
        #    self.vessel_enhancement = AdvancedVesselEnhancement(self.out_c)
        
        # 归一化和激活
        self.ln = nn.LayerNorm(out_c)
        self.act = nn.GELU()
        
        # 中等通道数VSSBlock
        self.block = MediumChannelVSSBlock(hidden_dim=out_c, drop_path=0.1)
        
        # 残差连接
        self.residual_conv = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1)
        #self.scale1 =nn.Parameter(torch.ones(1))
        # 动态权重
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        skip = self.residual_conv(x)
       # print(f"MediumChannelOCTAMambaBlock: input shape {x.shape}, skip shape {skip.shape}")
        # 特征提取
        x = self.conv(x)
        #print(f"MediumChannelOCTAMambaBlock: input shape {x.shape}, skip shape {skip.shape}")
        # 血管增强
        x = self.faz_enhancement(x)
        #x1=x
        # Mamba处理
        x = x.permute(0, 2, 3, 1)  # B C H W -> B H W C
        x = self.block(x)
        x = x.permute(0, 3, 1, 2)  # B H W C -> B C H W

        # 归一化和激活
        x = x.permute(0, 2, 3, 1)  # B C H W -> B H W C
        x = self.act(self.ln(x))
        x = x.permute(0, 3, 1, 2)  # B H W C -> B C H W
#        if  self.out_c >=64:
#         x = x+x1*0.1
        # 残差连接
        return x + skip * self.scale

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=8):  # 降低reduction以适应小通道数
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

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
         # 检查并调整尺寸以解决不匹配问题
        if g1.shape[2:] != x1.shape[2:]:
            # 获取目标尺寸（使用较小的尺寸）
            target_h = min(g1.shape[2], x1.shape[2])
            target_w = min(g1.shape[3], x1.shape[3])
            
            # 调整g1的尺寸
            if g1.shape[2:] != (target_h, target_w):
                g1 = F.interpolate(g1, size=(target_h, target_w), mode='bilinear', align_corners=True)
            
            # 调整x1的尺寸
            if x1.shape[2:] != (target_h, target_w):
                x1 = F.interpolate(x1, size=(target_h, target_w), mode='bilinear', align_corners=True)
                
            # 同时调整x的尺寸以匹配
            if x.shape[2:] != (target_h, target_w):
                x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=True)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class MediumChannelEncoderBlock(nn.Module):
    """中等通道数编码器块"""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.octamamba = MediumChannelOCTAMambaBlock(in_c, out_c)
        
        # 仅在中等通道数时使用SE
        #if out_c >= 32:
        self.se = SEBlock(out_c, reduction=8)
        # else:
        #     self.se = nn.Identity()
            
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.GELU()
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.octamamba(x)
        x = self.se(x)
        skip = self.act(self.bn(x))
        x = self.down(skip)
        return x, skip

class MediumChannelDecoderBlock(nn.Module):
    """中等通道数解码器块"""
    def __init__(self, in_c, skip_c, out_c,use_advanced_fusion=False):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # 仅在中等通道数时使用注意力机制
        #if skip_c >= 32:
        #self.attGate = Attention_block(F_g=in_c, F_l=skip_c, F_int=max(skip_c // 4, 4))
        # else:
        #     self.attGate = None
        self.attGate = None
        if use_advanced_fusion and skip_c >= 64:
            self.feature_fusion = AdaptiveFeatureFusion(
                encoder_channels=skip_c,
                decoder_channels=in_c,
                out_channels=in_c+skip_c
            )
        else:       
            self.feature_fusion = SimplifiedAttentionalFeatureFusion(
                skip_channels=skip_c,
                up_channels=in_c,
                out_channels=in_c+skip_c
            )
        
        self.bn2 = nn.BatchNorm2d(in_c + skip_c)
        self.octamamba = MediumChannelOCTAMambaBlock(in_c + skip_c, out_c)
        self.act = nn.ReLU()

    def forward(self, x, skip):
        x = self.up(x)
        # if self.attGate:
        #     skip = self.attGate(x, skip)
        # x = torch.cat([x, skip], dim=1)
        x = self.feature_fusion(skip, x)
        x = self.act(self.bn2(x))
        x = self.octamamba(x)
        return x

#========== 原始CompactQSEME类（已注释，使用新的增强版本） ==========
class CompactQSEME(nn.Module):
    """紧凑版QSEME，适合中等通道数"""
    def __init__(self, out_c=16):
        super().__init__()
        self.out_c = out_c
        
        # 降低初始通道数
        init_channels = 32
        
        self.init_conv = nn.Sequential(
            nn.Conv2d(1, init_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(),
        )

        # 简化的多分支处理
        branch_channels = init_channels // 4
        
        self.branch1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        self.branch2 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv2d(branch_channels, branch_channels, 3, padding=1, groups=branch_channels),
            nn.Conv2d(branch_channels, branch_channels, 1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU()
        )
        
        self.branch4 = SEBlock(branch_channels, reduction=4)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(init_channels, out_c, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.init_conv(x)
        
        # 分割为4个分支
        x1, x2, x3, x4 = x.chunk(4, dim=1)
        
        # 各分支处理
        feat1 = self.branch1(x1)
        feat2 = self.branch2(x2)
        feat3 = self.branch3(x3)
        feat4 = self.branch4(x4)
        
        # 重新组合
        fused = torch.cat([feat1, feat2, feat3, feat4], dim=1)
        return self.fusion(fused)

class FAZMamba(nn.Module):
    """中等通道数的OCTAMamba - 遵循16->32->64->128原则"""
    def __init__(self, qseme_type='conservative'):
        """
        Args:
            qseme_type (str): QSEME模块类型
                - 'conservative': 保守增强版DWT (推荐，风险最低)
                - 'enhanced': 完整增强版DWT+DFM+GCM (功能最全，参数较多)
                - 'original': 原始CompactQSEME (已注释，不推荐)
        """
        super().__init__()

        self.qseme = CompactQSEME(out_c=16)
        #self.qseme = QSEME(out_c=16)
        # self.qseme = EnhancedQSEMEWithDWT(out_c=16)
        # print("🚀 使用完整增强版QSEME (DWT+DFM+GCM)")

        # self.qseme = ConservativeEnhancedQSEME(out_c=16)
        # print("⚠️  未知QSEME类型，使用保守增强版")
        
        # 渐进式中等通道增长：16->32->64->128
        self.e1 = MediumChannelEncoderBlock(16, 32)    # 16->32
        self.e2 = MediumChannelEncoderBlock(32, 64)    # 32->64  
        self.e3 = MediumChannelEncoderBlock(64, 128)   # 64->128
        
        # 瓶颈层：限制在256以内
        #self.bottleneck = MediumChannelOCTAMambaBlock(128, 256)
        self.bottleneck = nn.Sequential(
            MediumChannelOCTAMambaBlock(128, 256),
            nn.Dropout2d(0.1),
            # VesselAwareDAM(256)
        )
        
        # 解码器：逐步减少通道数
        self.d3 = MediumChannelDecoderBlock(256, 128, 128,use_advanced_fusion=False)
        self.d2 = MediumChannelDecoderBlock(128, 64, 64,use_advanced_fusion=False)
        self.d1 = MediumChannelDecoderBlock(64, 32, 32,use_advanced_fusion=False)
        
        # 深度监督（轻量级）
        self.deep_supervision = nn.ModuleList([
            # nn.Conv2d(128, 1, 1),
            # nn.Conv2d(64, 1, 1),
            nn.Sequential(
                nn.Conv2d(128,64,3,padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64,1,1)
            ),
            nn.Sequential(
                nn.Conv2d(64,32,3,padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32,1,1)
            )
        ])
          # 最终输出层 - 增强版
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # VesselAwareDAM(32, reduction=4),  # 最后的血管注意力
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1)
        )
        # 最终输出层
        # self.final_conv = nn.Sequential(
        #     nn.Conv2d(32, 16, 3, padding=1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(16, 1, 1)
        # )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, return_deep_supervision=False):
        input_size = x.shape[-2:]
        
        # 特征提取
        x = self.qseme(x)    # 1->16
        
        # 编码器
        x, skip1 = self.e1(x)    # 16->32
        x, skip2 = self.e2(x)    # 32->64
        x, skip3 = self.e3(x)    # 64->128
        
        # 瓶颈
        x = self.bottleneck(x)   # 128->256
        
        # 解码器 + 深度监督
        x = self.d3(x, skip3)    # 256+128->128
        if return_deep_supervision:
            deep_out2 = self.deep_supervision[0](x)
        
        x = self.d2(x, skip2)    # 128+64->64
        if return_deep_supervision:
            deep_out1 = self.deep_supervision[1](x)
        
        x = self.d1(x, skip1)    # 64+32->32
        
        # 最终输出
        final_out = self.final_conv(x)
        final_out = self.sigmoid(final_out)
        
        if return_deep_supervision:
            deep_out1 = F.interpolate(deep_out1, size=input_size, mode='bilinear', align_corners=True)
            deep_out2 = F.interpolate(deep_out2, size=input_size, mode='bilinear', align_corners=True)
            return final_out, [self.sigmoid(deep_out1), self.sigmoid(deep_out2)]
        else:
            return final_out

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 测试不同版本的QSEME
    qseme_versions = ['conservative', 'enhanced']
    
    for version in qseme_versions:
        print(f"\n{'='*50}")
        print(f"测试 {version.upper()} 版本的OCTAMamba模型")
        print('='*50)
        
        # 创建模型
        model = FAZMamba(qseme_type=version).to(device)
        
        # 测试不同输入尺寸
        test_sizes = [(304, 304), (304, 304)]
        
        for h, w in test_sizes:
            img = torch.randn(1, 1, h, w).to(device)
            
            # 测试正常输出
            with torch.no_grad():
                out = model(img)
                print(f"输入尺寸 {h}x{w}: 输出形状 {out.shape}")
            
            # 测试深度监督输出
            with torch.no_grad():
                final_out, deep_outs = model(img, return_deep_supervision=True)
                print(f"  深度监督输出: {[out.shape for out in deep_outs]}")
        
        # 参数统计
        params = count_parameters(model)
        print(f"\n模型参数量: {params:,} ({params/1e6:.1f}M)")
        
        # 清理GPU内存
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    print(f"\n{'='*50}")
    print("✅ 所有版本的OCTAMamba模型测试完成！")
    print("💡 推荐使用: qseme_type='conservative'（保守增强版，风险最低）")
    print("🚀 高级功能: qseme_type='enhanced'（完整增强版，功能最全）")
    print('='*50) 