import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelWiseAttention(nn.Module):
    """通道级注意力 - 自适应选择重要通道"""
    def __init__(self, num_channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(num_channels * 2, num_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // reduction, num_channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, encoder_feat, decoder_feat):
        # 合并编码器和解码器特征的统计信息
        b, c, h, w = encoder_feat.size()
        _, c_dec, _, _ = decoder_feat.size()
        
        # 获取编码器特征的统计信息
        avg_enc = self.avg_pool(encoder_feat).reshape(b, c)
        max_enc = self.max_pool(encoder_feat).reshape(b, c)
        
        # 处理解码器特征，确保通道匹配
        if c_dec == c:
            # 通道数匹配时直接使用
            avg_dec = self.avg_pool(decoder_feat).reshape(b, c)
            max_dec = self.max_pool(decoder_feat).reshape(b, c)
        else:
            # 通道数不匹配时，先进行池化再调整尺寸
            avg_dec_raw = self.avg_pool(decoder_feat).reshape(b, c_dec)
            max_dec_raw = self.max_pool(decoder_feat).reshape(b, c_dec)
            
            # 使用线性层调整通道数
            # 如果c_dec > c，截断; 如果c_dec < c，填充
            if c_dec > c:
                avg_dec = avg_dec_raw[:, :c]
                max_dec = max_dec_raw[:, :c]
            else:
                avg_dec = torch.cat([avg_dec_raw, torch.zeros(b, c - c_dec, device=encoder_feat.device)], dim=1)
                max_dec = torch.cat([max_dec_raw, torch.zeros(b, c - c_dec, device=encoder_feat.device)], dim=1)
        
        # 拼接统计信息
        combined = torch.cat([avg_enc + avg_dec, max_enc + max_dec], dim=1)
        
        # 生成通道权重
        channel_weights = self.fc(combined).view(b, c, 1, 1)
        
        return channel_weights


class SpatialAttentionGate(nn.Module):
    """空间注意力门控 - 选择性地传递空间信息"""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        
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
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class DynamicConvFusion(nn.Module):
    """动态卷积融合 - 根据内容动态调整卷积核"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # 动态权重生成网络
        self.weight_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels * 2, in_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels * in_channels * kernel_size * kernel_size, 1)
        )
        
        # 标准卷积作为基础
        self.base_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        
    def forward(self, encoder_feat, decoder_feat):
        b, c, h, w = encoder_feat.shape
        b_dec, c_dec, h_dec, w_dec = decoder_feat.shape
        
        # 确保通道数一致
        if c + c_dec != self.in_channels * 2:
            # 如果拼接后的通道数与权重网络期望的不符，调整weight_net的第一层
            first_conv = self.weight_net[1]
            if not hasattr(self, 'adjusted_for_channels'):
                # 创建新的卷积层，输入通道数为c+c_dec
                new_conv = nn.Conv2d(c + c_dec, self.in_channels, 1).to(encoder_feat.device)
                # 复制原始权重的前部分（如果可能）
                with torch.no_grad():
                    if c + c_dec <= first_conv.in_channels:
                        new_conv.weight.data[:, :(c + c_dec)] = first_conv.weight.data[:, :(c + c_dec)].to(encoder_feat.device)
                    else:
                        # 如果新的输入通道数更大，需要初始化额外的权重
                        min_channels = min(c + c_dec, first_conv.in_channels)
                        new_conv.weight.data[:, :min_channels] = first_conv.weight.data[:, :min_channels].to(encoder_feat.device)
                    # 复制偏置
                    if first_conv.bias is not None:
                        new_conv.bias.data.copy_(first_conv.bias.data)
                # 替换
                self.weight_net[1] = new_conv
                self.adjusted_for_channels = True
        
        # 确保decoder_feat的空间尺寸匹配
        if decoder_feat.shape[2:] != encoder_feat.shape[2:]:
            decoder_feat = F.interpolate(
                decoder_feat, 
                size=encoder_feat.shape[2:], 
                mode='bilinear', 
                align_corners=True
            )
        
        # 生成动态权重
        combined = torch.cat([encoder_feat, decoder_feat], dim=1)
        dynamic_weight = self.weight_net(combined)
        dynamic_weight = dynamic_weight.view(b, self.out_channels, self.in_channels, 
                                           self.kernel_size, self.kernel_size)
        
        # 应用动态卷积
        output = []
        for i in range(b):
            out_i = F.conv2d(encoder_feat[i:i+1], dynamic_weight[i], 
                           padding=self.kernel_size//2)
            output.append(out_i)
            
        output = torch.cat(output, dim=0)
        
        # 与基础卷积结合
        base_output = self.base_conv(encoder_feat)
        
        return output + base_output


class CrossScaleFeatureFusion(nn.Module):
    """跨尺度特征融合 - 处理不同分辨率的特征"""
    def __init__(self, high_channels, low_channels, out_channels):
        super().__init__()
        
        # 高分辨率特征处理
        self.high_process = nn.Sequential(
            nn.Conv2d(high_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 低分辨率特征处理
        self.low_process = nn.Sequential(
            nn.Conv2d(low_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 多尺度融合
        self.ms_fusion = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels // 4, 3, padding=1, dilation=1),
            nn.Conv2d(out_channels, out_channels // 4, 3, padding=2, dilation=2),
            nn.Conv2d(out_channels, out_channels // 4, 3, padding=4, dilation=4),
            nn.Conv2d(out_channels, out_channels // 4, 1)
        ])
        
        # 最终融合
        self.final_fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, high_res_feat, low_res_feat):
        # 处理不同分辨率
        high_feat = self.high_process(high_res_feat)
        low_feat = self.low_process(low_res_feat)
        
        # 上采样低分辨率特征
        low_feat_up = F.interpolate(low_feat, size=high_feat.shape[2:], 
                                   mode='bilinear', align_corners=True)
        
        # 多尺度处理高分辨率特征
        ms_feats = []
        for conv in self.ms_fusion:
            ms_feats.append(conv(high_feat))
        high_feat_ms = torch.cat(ms_feats, dim=1)
        
        # 融合
        fused = torch.cat([high_feat_ms, low_feat_up], dim=1)
        output = self.final_fusion(fused)
        
        return output + high_feat  # 残差连接


class AdaptiveFeatureFusion(nn.Module):
    """自适应特征融合模块 - 智能融合跳跃连接"""
    def __init__(self, encoder_channels, decoder_channels, out_channels):
        super().__init__()
        
        # 通道注意力
        self.channel_attention = ChannelWiseAttention(encoder_channels)
        
        # 空间注意力门控
        self.spatial_gate = SpatialAttentionGate(
            F_g=decoder_channels,
            F_l=encoder_channels, 
            F_int=max(encoder_channels // 4, 32)
        )
        
        # 动态卷积融合（可选，计算量较大）
        self.use_dynamic_conv = encoder_channels >= 64
        if self.use_dynamic_conv:
            self.dynamic_fusion = DynamicConvFusion(
                encoder_channels,  # 使用encoder_channels作为输入
                out_channels
            )
        
        # 跨尺度融合
        self.cross_scale_fusion = CrossScaleFeatureFusion(
            encoder_channels, 
            decoder_channels,
            out_channels
        )
        
        # 最终细化
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, encoder_feat, decoder_feat):
        """
        Args:
            encoder_feat: 来自编码器的跳跃连接特征
            decoder_feat: 来自解码器的上采样特征
        """
        # 确保空间尺寸匹配
        if encoder_feat.shape[2:] != decoder_feat.shape[2:]:
            decoder_feat = F.interpolate(decoder_feat, size=encoder_feat.shape[2:], 
                                       mode='bilinear', align_corners=True)
        
        # 1. 通道注意力加权
        channel_weights = self.channel_attention(encoder_feat, decoder_feat)
        encoder_weighted = encoder_feat * channel_weights
        
        # 2. 空间注意力门控
        encoder_gated = self.spatial_gate(decoder_feat, encoder_weighted)
        
        # 3. 跨尺度融合
        fused_features = self.cross_scale_fusion(encoder_gated, decoder_feat)
        
        # 4. 动态卷积融合（可选）
        if self.use_dynamic_conv:
            concat_feat = torch.cat([encoder_gated, decoder_feat], dim=1)
            dynamic_feat = self.dynamic_fusion(encoder_gated, decoder_feat)
            fused_features = fused_features + dynamic_feat * 0.5
        
        # 5. 最终细化
        output = self.refine(fused_features)
        
        return output


class SimplifiedAttentionalFeatureFusion(nn.Module):
    """简化版融合 - 计算效率更高"""
    def __init__(self, skip_channels, up_channels, out_channels):
        super().__init__()
        
        # 特征对齐
        self.skip_conv = nn.Conv2d(skip_channels, out_channels, 1)
        self.up_conv = nn.Conv2d(up_channels, out_channels, 1)
        
        # 注意力权重生成
        self.attention = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels // 2, 1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, 2, 1),
            nn.Softmax(dim=1)
        )
        
        # 细化
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, skip_feat, up_feat):
        # 确保空间尺寸匹配
        if skip_feat.shape[2:] != up_feat.shape[2:]:
            up_feat = F.interpolate(up_feat, size=skip_feat.shape[2:], 
                                   mode='bilinear', align_corners=True)
            
        # 特征对齐
        skip_aligned = self.skip_conv(skip_feat)
        up_aligned = self.up_conv(up_feat)
        
        # 生成注意力权重
        concat = torch.cat([skip_aligned, up_aligned], dim=1)
        weights = self.attention(concat)
        
        # 加权融合
        fused = skip_aligned * weights[:, 0:1] + up_aligned * weights[:, 1:2]
        
        # 细化
        output = self.refine(fused)
        
        return output 