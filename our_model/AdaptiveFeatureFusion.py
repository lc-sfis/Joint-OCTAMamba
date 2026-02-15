import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelWiseAttention(nn.Module):
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
        b, c, h, w = encoder_feat.size()
        _, c_dec, _, _ = decoder_feat.size()
        
        avg_enc = self.avg_pool(encoder_feat).reshape(b, c)
        max_enc = self.max_pool(encoder_feat).reshape(b, c)
        
        if c_dec == c:
            avg_dec = self.avg_pool(decoder_feat).reshape(b, c)
            max_dec = self.max_pool(decoder_feat).reshape(b, c)
        else:
            avg_dec_raw = self.avg_pool(decoder_feat).reshape(b, c_dec)
            max_dec_raw = self.max_pool(decoder_feat).reshape(b, c_dec)
            
            if c_dec > c:
                avg_dec = avg_dec_raw[:, :c]
                max_dec = max_dec_raw[:, :c]
            else:
                avg_dec = torch.cat([avg_dec_raw, torch.zeros(b, c - c_dec, device=encoder_feat.device)], dim=1)
                max_dec = torch.cat([max_dec_raw, torch.zeros(b, c - c_dec, device=encoder_feat.device)], dim=1)
        
        combined = torch.cat([avg_enc + avg_dec, max_enc + max_dec], dim=1)
        
        channel_weights = self.fc(combined).view(b, c, 1, 1)
        
        return channel_weights


class SpatialAttentionGate(nn.Module):
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
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        

        self.weight_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels * 2, in_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels * in_channels * kernel_size * kernel_size, 1)
        )
        
        self.base_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        
    def forward(self, encoder_feat, decoder_feat):
        b, c, h, w = encoder_feat.shape
        b_dec, c_dec, h_dec, w_dec = decoder_feat.shape
        
        if c + c_dec != self.in_channels * 2:
        
            first_conv = self.weight_net[1]
            if not hasattr(self, 'adjusted_for_channels'):
                new_conv = nn.Conv2d(c + c_dec, self.in_channels, 1).to(encoder_feat.device)
                with torch.no_grad():
                    if c + c_dec <= first_conv.in_channels:
                        new_conv.weight.data[:, :(c + c_dec)] = first_conv.weight.data[:, :(c + c_dec)].to(encoder_feat.device)
                    else:
                        min_channels = min(c + c_dec, first_conv.in_channels)
                        new_conv.weight.data[:, :min_channels] = first_conv.weight.data[:, :min_channels].to(encoder_feat.device)
                    if first_conv.bias is not None:
                        new_conv.bias.data.copy_(first_conv.bias.data)
                self.weight_net[1] = new_conv
                self.adjusted_for_channels = True
        
        if decoder_feat.shape[2:] != encoder_feat.shape[2:]:
            decoder_feat = F.interpolate(
                decoder_feat, 
                size=encoder_feat.shape[2:], 
                mode='bilinear', 
                align_corners=True
            )
        
        combined = torch.cat([encoder_feat, decoder_feat], dim=1)
        dynamic_weight = self.weight_net(combined)
        dynamic_weight = dynamic_weight.view(b, self.out_channels, self.in_channels, 
                                           self.kernel_size, self.kernel_size)
        
        output = []
        for i in range(b):
            out_i = F.conv2d(encoder_feat[i:i+1], dynamic_weight[i], 
                           padding=self.kernel_size//2)
            output.append(out_i)
            
        output = torch.cat(output, dim=0)
        
        base_output = self.base_conv(encoder_feat)
        
        return output + base_output


class CrossScaleFeatureFusion(nn.Module):
    def __init__(self, high_channels, low_channels, out_channels):
        super().__init__()
        
        self.high_process = nn.Sequential(
            nn.Conv2d(high_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.low_process = nn.Sequential(
            nn.Conv2d(low_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.ms_fusion = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels // 4, 3, padding=1, dilation=1),
            nn.Conv2d(out_channels, out_channels // 4, 3, padding=2, dilation=2),
            nn.Conv2d(out_channels, out_channels // 4, 3, padding=4, dilation=4),
            nn.Conv2d(out_channels, out_channels // 4, 1)
        ])
        
        self.final_fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, high_res_feat, low_res_feat):
        high_feat = self.high_process(high_res_feat)
        low_feat = self.low_process(low_res_feat)
        low_feat_up = F.interpolate(low_feat, size=high_feat.shape[2:], 
                                   mode='bilinear', align_corners=True)
        
        ms_feats = []
        for conv in self.ms_fusion:
            ms_feats.append(conv(high_feat))
        high_feat_ms = torch.cat(ms_feats, dim=1)
        
        fused = torch.cat([high_feat_ms, low_feat_up], dim=1)
        output = self.final_fusion(fused)
        
        return output + high_feat 


class AdaptiveFeatureFusion(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, out_channels):
        super().__init__()
        
        self.channel_attention = ChannelWiseAttention(encoder_channels)
        
        self.spatial_gate = SpatialAttentionGate(
            F_g=decoder_channels,
            F_l=encoder_channels, 
            F_int=max(encoder_channels // 4, 32)
        )
        
        self.use_dynamic_conv = encoder_channels >= 64
        if self.use_dynamic_conv:
            self.dynamic_fusion = DynamicConvFusion(
                encoder_channels,  # 使用encoder_channels作为输入
                out_channels
            )
        
        self.cross_scale_fusion = CrossScaleFeatureFusion(
            encoder_channels, 
            decoder_channels,
            out_channels
        )
        
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, encoder_feat, decoder_feat):
        if encoder_feat.shape[2:] != decoder_feat.shape[2:]:
            decoder_feat = F.interpolate(decoder_feat, size=encoder_feat.shape[2:], 
                                       mode='bilinear', align_corners=True)
        

        channel_weights = self.channel_attention(encoder_feat, decoder_feat)
        encoder_weighted = encoder_feat * channel_weights
        
        encoder_gated = self.spatial_gate(decoder_feat, encoder_weighted)
        
        fused_features = self.cross_scale_fusion(encoder_gated, decoder_feat)
        
        if self.use_dynamic_conv:
            concat_feat = torch.cat([encoder_gated, decoder_feat], dim=1)
            dynamic_feat = self.dynamic_fusion(encoder_gated, decoder_feat)
            fused_features = fused_features + dynamic_feat * 0.5
        
        output = self.refine(fused_features)
        
        return output


class SimplifiedAttentionalFeatureFusion(nn.Module):
    def __init__(self, skip_channels, up_channels, out_channels):
        super().__init__()
        
        self.skip_conv = nn.Conv2d(skip_channels, out_channels, 1)
        self.up_conv = nn.Conv2d(up_channels, out_channels, 1)
        
        self.attention = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels // 2, 1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, 2, 1),
            nn.Softmax(dim=1)
        )

        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, skip_feat, up_feat):

        if skip_feat.shape[2:] != up_feat.shape[2:]:
            up_feat = F.interpolate(up_feat, size=skip_feat.shape[2:], 
                                   mode='bilinear', align_corners=True)
            
 
        skip_aligned = self.skip_conv(skip_feat)
        up_aligned = self.up_conv(up_feat)

        concat = torch.cat([skip_aligned, up_aligned], dim=1)
        weights = self.attention(concat)

        fused = skip_aligned * weights[:, 0:1] + up_aligned * weights[:, 1:2]
        
        output = self.refine(fused)
        
        return output 
