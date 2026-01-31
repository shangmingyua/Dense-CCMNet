import sys

sys.path.extend(['../..', '..'])
import torch
import torch.nn as nn


class FiLMLayer(nn.Module):
    """ Feature-wise Linear Modulation Layer """

    def __init__(self, channels, condition_dim):
        super(FiLMLayer, self).__init__()
        self.fc = nn.Linear(condition_dim, 2 * channels)
        # 初始化 gamma=1, beta=0
        self.fc.weight.data.normal_(0, 0.02)
        self.fc.bias.data.fill_(0)
        self.fc.bias.data[:channels].fill_(1)

    def forward(self, x, condition):
        # x: [B, C, H, W]
        # condition: [B, condition_dim]
        params = self.fc(condition).unsqueeze(2).unsqueeze(3)
        gamma, beta = torch.chunk(params, 2, dim=1)
        return gamma * x + beta


class ConvBlock(nn.Module):
    """ Standard Conv Block """

    def __init__(self, kernel, in_depth, conv_depth, stride=1, padding=1,
                 normalization=False, norm_type='BN', pooling=False,
                 bias_initialization='zeros', activation=True, dilation=1,
                 return_before_pooling=False):
        super(ConvBlock, self).__init__()

        conv = nn.Conv2d(in_depth, conv_depth, kernel, stride=stride,
                         dilation=dilation, padding=padding,
                         padding_mode='replicate')
        nn.init.kaiming_normal_(conv.weight)
        if bias_initialization == 'ones':
            nn.init.ones_(conv.bias)
        else:
            nn.init.zeros_(conv.bias)

        self.activation = nn.LeakyReLU(inplace=False) if activation else None

        self.normalization = None
        if normalization:
            if norm_type == 'BN':
                self.normalization = nn.BatchNorm2d(conv_depth, affine=True)
            elif norm_type == 'IN':
                self.normalization = nn.InstanceNorm2d(conv_depth, affine=False)

        self.conv = conv
        self.pooling = nn.MaxPool2d(2, stride=2) if pooling else None
        self.return_before_pooling = return_before_pooling

    def forward(self, x):
        x = self.conv(x)
        if self.normalization is not None:
            x = self.normalization(x)
        if self.activation is not None:
            x = self.activation(x)

        if self.pooling is not None:
            y = self.pooling(x)
            if self.return_before_pooling:
                return y, x
            return y
        return x


class DoubleConvBlock(nn.Module):
    """ Double Conv Block for U-Net """

    def __init__(self, in_depth, out_depth, mid_depth=None, kernel=3, stride=1,
                 padding=None, dilation=None, normalization=False, norm_type='BN',
                 pooling=True, return_before_pooling=False):
        super().__init__()
        if padding is None: padding = [1, 1]
        if mid_depth is None: mid_depth = out_depth

        # 简化版：仅在第二层后进行 Pooling
        self.double_conv_1 = ConvBlock(kernel=kernel, in_depth=in_depth,
                                       conv_depth=mid_depth, stride=stride,
                                       padding=padding[0], pooling=False,
                                       normalization=normalization, norm_type=norm_type)

        self.double_conv_2 = ConvBlock(kernel=kernel, in_depth=mid_depth,
                                       conv_depth=out_depth, stride=stride,
                                       padding=padding[1], pooling=pooling,
                                       return_before_pooling=return_before_pooling,
                                       normalization=normalization, norm_type=norm_type)

    def forward(self, x):
        x = self.double_conv_1(x)
        return self.double_conv_2(x)


class ChannelAttention(nn.Module):
    """Channel Attention Module (CBAM)"""
    
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: [B, C, H, W]
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial Attention Module (CBAM)"""
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: [B, C, H, W]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM)"""
    
    def __init__(self, channels, reduction=16, spatial_kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)
    
    def forward(self, x):
        # Channel attention first
        out = x * self.channel_attention(x)
        # Then spatial attention
        out = out * self.spatial_attention(out)
        return out