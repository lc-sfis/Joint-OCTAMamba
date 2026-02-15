import torch
import torch.nn as nn
import torch.nn.functional as F


class VesselChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.local_pool = nn.AvgPool2d(kernel_size=7, stride=1, padding=3)
        
        self.attention_net = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels // reduction, 3, padding=1, groups=in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):

        avg_global = self.avg_pool(x)
        max_global = self.max_pool(x)
        
     
        local_feat = self.local_pool(x)
        local_feat = F.interpolate(local_feat, size=x.shape[2:], mode='bilinear', align_corners=True)
        local_global = self.avg_pool(local_feat)
        
        combined = torch.cat([avg_global, max_global, local_global], dim=1)
        
        attention = self.attention_net(combined)
        
        return attention


class VesselSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        
        self.basic_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(1)
        )
        
        self.horizontal_attention = nn.Conv2d(1, 1, kernel_size=(1, kernel_size), padding=(0, kernel_size//2))
        self.vertical_attention = nn.Conv2d(1, 1, kernel_size=(kernel_size, 1), padding=(kernel_size//2, 0))
        
        self.diagonal_kernel_size = 5
        self.diagonal_attention1 = nn.Conv2d(1, 1, kernel_size=self.diagonal_kernel_size, padding=self.diagonal_kernel_size//2)
        self.diagonal_attention2 = nn.Conv2d(1, 1, kernel_size=self.diagonal_kernel_size, padding=self.diagonal_kernel_size//2)
        
        self.fusion = nn.Sequential(
            nn.Conv2d(5, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        basic_input = torch.cat([avg_out, max_out], dim=1)
        basic_att = self.basic_attention(basic_input)
        
        h_att = self.horizontal_attention(avg_out)
        v_att = self.vertical_attention(avg_out)
        
        avg_rot45 = torch.rot90(avg_out, 1, [2, 3])
        d1_att = self.diagonal_attention1(avg_rot45)
        d1_att = torch.rot90(d1_att, -1, [2, 3])
        
        avg_rot_45 = torch.rot90(avg_out, -1, [2, 3])
        d2_att = self.diagonal_attention2(avg_rot_45)
        d2_att = torch.rot90(d2_att, 1, [2, 3])
        
        all_attention = torch.cat([basic_att, h_att, v_att, d1_att, d2_att], dim=1)
        spatial_attention = self.fusion(all_attention)
        
        return spatial_attention


class VesselStructureAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        self.centerline_detect = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels // 4, 3, padding=1, groups=in_channels // 4),
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Sigmoid()
        )
        

        self.bifurcation_detect = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels // 4, in_channels // 8, (1, 5), padding=(0, 2)),
            nn.Conv2d(in_channels // 8, in_channels // 8, (5, 1), padding=(2, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, 1, 1),
            nn.Sigmoid()
        )
        
       
        self.width_variation = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels // 4, 3, padding=2, dilation=2),
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Sigmoid()
        )
        
       
        self.combine = nn.Sequential(
            nn.Conv2d(3, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        
        centerline = self.centerline_detect(x)
        bifurcation = self.bifurcation_detect(x)
        width_var = self.width_variation(x)
        
       
        combined = torch.cat([centerline, bifurcation, width_var], dim=1)
        structure_attention = self.combine(combined)
        
        return structure_attention


class VesselMultiAttentionFusion(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        
        
        self.channel_attention = VesselChannelAttention(in_channels, reduction)
        self.spatial_attention = VesselSpatialAttention(kernel_size=7)
        self.structure_attention = VesselStructureAttention(in_channels)
        
        
        self.weight_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 3, 1),
            nn.Softmax(dim=1)
        )
        
        
        self.feature_enhance = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels // 4),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels)
        )
        
      
        self.final_conv = nn.Conv2d(in_channels, in_channels, 1)
        
    def forward(self, x):
      
        weights = self.weight_generator(x)  # (B, 3, 1, 1)
        
       
        channel_att = self.channel_attention(x)
        spatial_att = self.spatial_attention(x)
        structure_att = self.structure_attention(x)
        
        
        x_channel = x * channel_att
        x_spatial = x * spatial_att  
        x_structure = x * structure_att
        

        x_weighted = x_channel * weights[:, 0:1] + \
                    x_spatial * weights[:, 1:2] + \
                    x_structure * weights[:, 2:3]
        
       
        x_enhanced = self.feature_enhance(x_weighted)
        
      
        output = self.final_conv(x_enhanced)
        
        
        return x + output 
