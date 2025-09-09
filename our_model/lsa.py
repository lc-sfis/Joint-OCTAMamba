import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class LSA(nn.Module):
    """Local Stochastic Affine (LSA) 正则化, 引入局部随机仿射变换以提升泛化。
    代码简化自 Mamba-Sea 项目，仅保留关键逻辑。

    参数:
        p (float): 触发概率.
        eps (float): 数值稳定因子.
    """

    def __init__(self, p: float = 0.9, eps: float = 1e-6):
        super().__init__()
        self.p = p
        self.eps = eps

    def _reparameterize(self, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """重参数化采样."""
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def _sqrtvar(self, x: torch.Tensor) -> torch.Tensor:
        """沿 batch 维度复用方差平方根."""
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
        return t.repeat(x.shape[0], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 仅在 batch>1 且随机概率满足时启用
        if x.shape[0] == 1 or random.random() > self.p:
            return x

        mean = x.mean(dim=2, keepdim=False)  # B C
        std = (x.var(dim=2, keepdim=False) + self.eps).sqrt()  # B C

        sqrtvar_mu = self._sqrtvar(mean)
        sqrtvar_std = self._sqrtvar(std)

        beta = self._reparameterize(mean, sqrtvar_mu)
        gamma = self._reparameterize(std, sqrtvar_std)

        # 标准化并应用随机仿射
        x_hat = (x - mean.unsqueeze(2)) / std.unsqueeze(2)
        x_hat = x_hat * gamma.unsqueeze(2) + beta.unsqueeze(2)

        # 连续随机掩码 (75% 区域) 改变部分时间步
        B, C, L = x_hat.shape
        mask = torch.zeros((B, 1, L), device=x.device)
        for i in range(B):
            length = int(L * 0.75)
            start = torch.randint(0, L - length + 1, (1,), device=x.device).item()
            mask[i, :, start:start + length] = 1.0
        return x_hat * mask + x * (1 - mask) 