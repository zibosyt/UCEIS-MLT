import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out) * x


def insert_attention_to_coatnet(backbone, attention_type='channel'):
    if hasattr(backbone, 'stages'):
        for i, stage in enumerate(backbone.stages):
            if isinstance(stage, nn.Sequential):
                in_channels = stage[-1].conv.out_channels if hasattr(stage[-1], 'conv') else stage[-1].out_channels
                if attention_type == 'channel':
                    stage.add_module(f'channel_attention_{i}', ChannelAttention(in_channels))
                elif attention_type == 'spatial':
                    stage.add_module(f'spatial_attention_{i}', SpatialAttention())
                elif attention_type == 'both':
                    stage.add_module(f'channel_attention_{i}', ChannelAttention(in_channels))
                    stage.add_module(f'spatial_attention_{i}', SpatialAttention())
            else:
                last_module = stage
                if hasattr(last_module, 'out_channels'):
                    in_channels = last_module.out_channels
                elif hasattr(last_module, 'conv') and hasattr(last_module.conv, 'out_channels'):
                    in_channels = last_module.conv.out_channels
                else:
                    in_channels = 512

                if attention_type == 'channel':
                    setattr(stage, f'channel_attention_{i}', ChannelAttention(in_channels))
                elif attention_type == 'spatial':
                    setattr(stage, f'spatial_attention_{i}', SpatialAttention())
                elif attention_type == 'both':
                    setattr(stage, f'channel_attention_{i}', ChannelAttention(in_channels))
                    setattr(stage, f'spatial_attention_{i}', SpatialAttention())