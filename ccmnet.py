import sys

sys.path.append('..')
sys.path.append('../..')
import torch
import torch.nn as nn
import numpy as np
import colour
import torchvision.models as models
from src.diff_hist import DifferentiableHistogram
from src import ops


class network(nn.Module):
    def __init__(self, input_size=64, cfe_feature_num=8,
                 net_depth=4, max_conv_depth=128, device='cuda'):
        super(network, self).__init__()

        self.device = device
        self.input_size = input_size

        # ==========================================================
        # 1. CFE Branch (Camera Fingerprint Extraction)
        # ==========================================================
        # [Fix 1] 调整边界以适应 Log-Chroma 空间 (-3, 3)
        self.hist_layer = DifferentiableHistogram(nbins=64, hist_boundary=(-3.0, 3.0))
        self.cfe_encoder = CFEEncoder(size=input_size, n_layers=4,
                                      feat_ch=cfe_feature_num, max_depth=128)

        # ==========================================================
        # 2. Spatial Encoder (ResNet-18 Backbone)
        # ==========================================================
        self.spatial_encoder = ResNetEncoder(pretrained=True)
        # ResNet-18 channels: Stem(64), Layer1(64), Layer2(128), Layer3(256), Layer4(512)
        enc_channels = [64, 64, 128, 256, 512]

        # ==========================================================
        # 3. Spatial Decoder
        # ==========================================================
        self.spatial_decoder = SpatialDecoderWithFiLM(
            output_channels=3,
            enc_channels=enc_channels,
            condition_dim=cfe_feature_num,
            use_cbam=True,
            use_deep_supervision=True
        )

    def forward(self, img_patch, cm1, cm2):
        # --- A. CFE Branch ---
        cm1 = cm1.view(-1, 3, 3)
        cm2 = cm2.view(-1, 3, 3)

        # [Fix 2] 获取可导的 Log-Chroma 直方图
        cfe_hist = self.get_illum_hist(cm1, cm2)

        if torch.isnan(cfe_hist).any():
            cfe_hist = torch.nan_to_num(cfe_hist, nan=0.0)
        cfe_emb = self.cfe_encoder(cfe_hist)

        # --- B. Spatial Encoder ---
        bottleneck, skips = self.spatial_encoder(img_patch)

        # --- C. Decoder ---
        decoder_output = self.spatial_decoder(bottleneck, skips, condition=cfe_emb)

        # --- D. Output ---
        if isinstance(decoder_output, dict):
            pred_map = torch.nn.functional.normalize(decoder_output['final'], dim=1, eps=1e-8)
            aux_map = torch.nn.functional.normalize(decoder_output['aux'], dim=1, eps=1e-8) if decoder_output[
                                                                                                   'aux'] is not None else None
            return {'final': pred_map, 'aux': aux_map}
        else:
            pred_map = torch.nn.functional.normalize(decoder_output, dim=1, eps=1e-8)
            return {'final': pred_map, 'aux': None}

    def get_illum_hist(self, cm1_3x3, cm2_3x3, colortemp_step=80):
        batch_size = cm1_3x3.shape[0]
        color_temps = torch.arange(2500, 7501, colortemp_step).to(self.device)

        try:
            xy_chromas = np.array([colour.temperature.CCT_to_xy(t.item()) for t in color_temps])
        except:
            xy_chromas = np.random.rand(len(color_temps), 2)

        xy = torch.tensor(xy_chromas).float().to(self.device)
        z = 1 - xy[:, 0] - xy[:, 1]
        xyz = torch.stack([xy[:, 0], xy[:, 1], z], dim=1)
        xyz = xyz / (xyz[:, 1:2] + 1e-8)

        g = torch.clip((1 / color_temps - 1 / 2856) / (1 / 6504 - 1 / 2856), 0, 1)
        g = g.view(1, -1, 1, 1).repeat(batch_size, 1, 3, 3)

        cm1_exp = cm1_3x3.unsqueeze(1).repeat(1, len(color_temps), 1, 1)
        cm2_exp = cm2_3x3.unsqueeze(1).repeat(1, len(color_temps), 1, 1)
        cm_t = g * cm1_exp + (1 - g) * cm2_exp

        xyz_exp = xyz.view(1, len(color_temps), 3, 1).repeat(batch_size, 1, 1, 1)
        cam_rgb = torch.matmul(cm_t, xyz_exp).squeeze(-1)
        cam_rgb = torch.clamp(cam_rgb, min=1e-6, max=10.0)

        # [Fix 3] 转换为 Log-Chromaticity (u, v) 并且移除 detach()
        # Norm: R/G, 1, B/G
        cam_rgb_norm = cam_rgb / (cam_rgb[:, :, 1:2] + 1e-8)
        # Log: ln(R/G), 0, ln(B/G)
        cam_rgb_log = torch.log(cam_rgb_norm + 1e-8)

        # 取 u=log(R/G) 和 v=log(B/G)
        chroma_input = cam_rgb_log[:, :, [0, 2]]

        # 输入 Soft Histogram
        hist = self.hist_layer(chroma_input)

        return hist.unsqueeze(1)


# ==============================================================================
# Helper Modules
# ==============================================================================

class ResNetEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        backbone = models.resnet18(pretrained=pretrained)
        self.initial = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x):
        x0 = self.initial(x)  # 128x128
        x1 = self.maxpool(x0)  # 64x64
        x1 = self.layer1(x1)  # 64x64
        x2 = self.layer2(x1)  # 32x32
        x3 = self.layer3(x2)  # 16x16
        x4 = self.layer4(x3)  # 8x8
        return x4, [x0, x1, x2, x3]


class SpatialDecoderWithFiLM(nn.Module):
    def __init__(self, output_channels, enc_channels, condition_dim,
                 use_cbam=True, use_deep_supervision=True):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.films = nn.ModuleList()
        self.cbams = nn.ModuleList()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.use_cbam = use_cbam
        self.use_deep_supervision = use_deep_supervision

        skip_channels_list = enc_channels[:-1][::-1]
        prev_c = enc_channels[-1]

        # U-Net Decoder Path
        for i, skip_c in enumerate(skip_channels_list):
            in_c = prev_c + skip_c
            out_c = skip_c
            self.blocks.append(DoubleConvBlock(in_depth=in_c, out_depth=out_c, pooling=False, normalization=True))
            self.films.append(FiLMLayer(out_c, condition_dim))
            if use_cbam:
                self.cbams.append(CBAM(out_c))
            else:
                self.cbams.append(nn.Identity())
            prev_c = out_c

        # [Fix 4] Final Upsampling Block (128->256)
        self.final_upsample_block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(prev_c, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.final = nn.Conv2d(32, output_channels, 1)
        nn.init.constant_(self.final.bias, 0.5)
        nn.init.xavier_uniform_(self.final.weight, gain=0.05)

        if use_deep_supervision:
            aux_c = skip_channels_list[-2]
            self.aux_head = nn.Conv2d(aux_c, output_channels, 1)
            nn.init.constant_(self.aux_head.bias, 0.5)
            nn.init.xavier_uniform_(self.aux_head.weight, gain=0.05)
        else:
            self.aux_head = None

    def forward(self, x, skips, condition):
        skips = skips[::-1]
        aux_map = None
        for i, (block, film, cbam) in enumerate(zip(self.blocks, self.films, self.cbams)):
            x = self.upsample(x)
            if x.shape[2:] != skips[i].shape[2:]:
                x = nn.functional.interpolate(x, size=skips[i].shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skips[i]], dim=1)
            x = block(x)
            x = film(x, condition)
            x = nn.functional.leaky_relu(x, 0.2)
            x = cbam(x)
            if self.use_deep_supervision and i == len(self.blocks) - 2:
                aux_map = self.aux_head(x)

        # Final Upsample
        x = self.final_upsample_block(x)
        final_map = self.final(x)

        if self.use_deep_supervision and aux_map is not None:
            return {'final': final_map, 'aux': aux_map}
        else:
            return {'final': final_map, 'aux': None}


class DoubleConvBlock(nn.Module):
    def __init__(self, in_depth, out_depth, pooling=False, return_before_pooling=False, normalization=True):
        super().__init__()
        self.pooling = pooling
        self.return_before_pooling = return_before_pooling
        layers = [nn.Conv2d(in_depth, out_depth, 3, padding=1)]
        if normalization: layers.append(nn.BatchNorm2d(out_depth))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Conv2d(out_depth, out_depth, 3, padding=1))
        if normalization: layers.append(nn.BatchNorm2d(out_depth))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.convs = nn.Sequential(*layers)
        if pooling: self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.convs(x)
        if self.pooling:
            x_pooled = self.pool(x)
            if self.return_before_pooling: return x_pooled, x
            return x_pooled
        return x


class FiLMLayer(nn.Module):
    def __init__(self, channels, condition_dim):
        super().__init__()
        self.fc = nn.Linear(condition_dim, 2 * channels)
        self.fc.weight.data.normal_(0, 0.02)
        self.fc.bias.data.fill_(0)
        self.fc.bias.data[:channels].fill_(1)

    def forward(self, x, condition):
        params = self.fc(condition)
        gamma, beta = torch.split(params, x.size(1), dim=1)
        return gamma.unsqueeze(2).unsqueeze(3) * x + beta.unsqueeze(2).unsqueeze(3)


class CBAM(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # [建议] 如果之前怀疑压缩太厉害，可以将这里的 reduction 改小，比如改为 8
        mid_channels = max(channels // reduction, 1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid_channels, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(mid_channels, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        self.conv_spatial = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        # ==========================================
        # 1. Channel Attention (通道注意力)
        # ==========================================
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_scale = self.sigmoid(avg_out + max_out)


        x = x * channel_scale

        # ==========================================
        # 2. Spatial Attention (空间注意力)
        # ==========================================
        avg_s = torch.mean(x, dim=1, keepdim=True)
        max_s, _ = torch.max(x, dim=1, keepdim=True)
        spatial_scale = self.sigmoid(self.conv_spatial(torch.cat([avg_s, max_s], dim=1)))


        return x * spatial_scale
# class CBAM(nn.Module):
#     def __init__(self, channels, reduction=16):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         mid_channels = max(channels // reduction, 1)
#         self.fc = nn.Sequential(nn.Conv2d(channels, mid_channels, 1, bias=False), nn.ReLU(),
#                                 nn.Conv2d(mid_channels, channels, 1, bias=False))
#         self.sigmoid = nn.Sigmoid()
#         self.conv_spatial = nn.Conv2d(2, 1, 7, padding=3, bias=False)
#
#     def forward(self, x):
#         avg_out = self.fc(self.avg_pool(x))
#         max_out = self.fc(self.max_pool(x))
#         x = x * self.sigmoid(avg_out + max_out)
#         avg_s = torch.mean(x, dim=1, keepdim=True)
#         max_s, _ = torch.max(x, dim=1, keepdim=True)
#         return x * self.sigmoid(self.conv_spatial(torch.cat([avg_s, max_s], dim=1)))


class CFEEncoder(nn.Module):
    def __init__(self, size, n_layers, feat_ch, max_depth):
        super().__init__()
        layers = [];
        in_d = 1;
        out_d = 16
        for i in range(n_layers):
            layers.append(DoubleConvBlock(in_depth=in_d, out_depth=out_d, pooling=True, normalization=True))
            in_d = out_d;
            out_d = min(out_d * 2, max_depth)
        self.features = nn.Sequential(*layers)
        final_size = size // (2 ** n_layers);
        flat_dim = in_d * (final_size ** 2)
        self.mlp = nn.Sequential(nn.Flatten(), nn.Linear(flat_dim, 128), nn.LeakyReLU(0.2, inplace=True),
                                 nn.Linear(128, feat_ch))

    def forward(self, x): return self.mlp(self.features(x))