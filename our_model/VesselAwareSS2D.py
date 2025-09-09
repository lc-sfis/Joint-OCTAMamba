import torch
import torch.nn as nn
import math
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange
import torch.nn.functional as F
from functools import partial
from timm.models.layers import DropPath


class VesselAwareSS2D(nn.Module):
    """血管感知的SS2D - 专门优化血管分割"""
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

        # 输入投影
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        
        # 使用简单的分组方式，不再尝试优化为3等分，而是使用更简单的2等分或不分组
        # 对于多尺度卷积，使用1/3、1/3和1/3的分布
        div1 = max(1, self.d_inner // 3)
        div2 = max(1, self.d_inner // 3)
        div3 = self.d_inner - div1 - div2  # 剩余部分
        
        # 确保所有通道数为正数
        if div3 <= 0:
            div1 = self.d_inner // 2
            div2 = self.d_inner - div1
            div3 = 0
        
        # 多尺度卷积 - 捕获不同粗细的血管，使用固定分组数
        self.multi_scale_conv = nn.ModuleList([
            nn.Conv2d(self.d_inner, div1, kernel_size=3, padding=1, groups=1, **factory_kwargs),
            nn.Conv2d(self.d_inner, div2, kernel_size=5, padding=2, groups=1, **factory_kwargs)
        ])
        
        # 如果有足够的通道，添加第三个卷积
        if div3 > 0:
            self.multi_scale_conv.append(
                nn.Conv2d(self.d_inner, div3, kernel_size=7, padding=3, groups=1, **factory_kwargs)
            )
        
        # 方向感知卷积 - 使用固定通道数分配
        div_dir = max(1, self.d_inner // 4)  # 每个方向卷积的输出通道数
        
        # 方向感知卷积 - 捕获血管的方向性
        self.directional_conv = nn.ModuleList([
            nn.Conv2d(self.d_inner, div_dir, kernel_size=(1, 7), padding=(0, 3), groups=1),  # 水平
            nn.Conv2d(self.d_inner, div_dir, kernel_size=(7, 1), padding=(3, 0), groups=1),  # 垂直
            nn.Conv2d(self.d_inner, div_dir, kernel_size=5, padding=2, groups=1),  # 45度
            nn.Conv2d(self.d_inner, div_dir, kernel_size=5, padding=2, groups=1)   # -45度
        ])
        
        self.act = nn.SiLU()

        # Mamba参数初始化
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        # 血管细节增强模块
        self.vessel_detail_enhance = nn.Sequential(
            nn.Conv2d(self.d_inner, self.d_inner // 2, 1),
            nn.BatchNorm2d(self.d_inner // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.d_inner // 2, self.d_inner // 2, 3, padding=1, groups=self.d_inner // 16 or 1),  # 使用较小的分组数
            nn.BatchNorm2d(self.d_inner // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.d_inner // 2, self.d_inner, 1),
            nn.Sigmoid()
        )

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
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
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).view(1, -1).repeat(d_inner, 1).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = A_log.unsqueeze(0).repeat(copies, 1, 1)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = D.unsqueeze(0).repeat(copies, 1)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core_enhanced(self, x: torch.Tensor):
        """增强的核心前向传播 - 包含血管特征增强"""
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        # 多尺度特征提取
        multi_scale_feats = []
        for conv in self.multi_scale_conv:
            multi_scale_feats.append(conv(x))
        x_multi = torch.cat(multi_scale_feats, dim=1)
        
        # 方向感知特征
        directional_feats = []
        for i, conv in enumerate(self.directional_conv):
            if i == 2:  # 45度旋转
                x_rot = torch.rot90(x, 1, [2, 3])
                feat = conv(x_rot)
                feat = torch.rot90(feat, -1, [2, 3])
            elif i == 3:  # -45度旋转
                x_rot = torch.rot90(x, -1, [2, 3])
                feat = conv(x_rot)
                feat = torch.rot90(feat, 1, [2, 3])
            else:
                feat = conv(x)
            directional_feats.append(feat)
        x_directional = torch.cat(directional_feats, dim=1)
        
        # 融合多尺度和方向特征
        x = (x_multi + x_directional) / 2
        
        # 血管细节增强
        vessel_attention = self.vessel_detail_enhance(x)
        x = x * vessel_attention

        # 标准Mamba扫描
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
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
        x = self.act(x)
        
        # 使用增强的核心
        y1, y2, y3, y4 = self.forward_core_enhanced(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out 