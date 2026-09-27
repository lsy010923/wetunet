from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# DropPath import compatibility: timm versions differ.
try:
    from timm.layers import DropPath
except Exception:
    from timm.models.layers import DropPath


# ==========================================
# 1) MS-ADCAM: Multi-Scale Anisotropic Depthwise Context Aggregation Module
# ==========================================

class ChannelAttentionMSADCAM(nn.Module):
    """Average/max pooled channel recalibration used inside MS-ADCAM."""

    def __init__(self, input_channels: int, internal_neurons: int):
        super().__init__()
        internal_neurons = max(int(internal_neurons), 1)
        self.input_channels = input_channels

        self.fc1 = nn.Conv2d(input_channels, internal_neurons, kernel_size=1, stride=1, bias=True)
        self.fc2 = nn.Conv2d(internal_neurons, input_channels, kernel_size=1, stride=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_att = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        avg_att = self.fc1(avg_att)
        avg_att = F.relu(avg_att, inplace=True)
        avg_att = self.fc2(avg_att)
        avg_att = torch.sigmoid(avg_att)

        max_att = F.adaptive_max_pool2d(x, output_size=(1, 1))
        max_att = self.fc1(max_att)
        max_att = F.relu(max_att, inplace=True)
        max_att = self.fc2(max_att)
        max_att = torch.sigmoid(max_att)

        att = avg_att + max_att
        return att.view(-1, self.input_channels, 1, 1)


class MSADCAM(nn.Module):
    """
    MS-ADCAM from the paper-level naming.

    This keeps the original bottleneck computation: channel recalibration followed
    by 5x5 depthwise context and multi-scale anisotropic 7/11/21 strip branches.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channel_attention_reduce: int = 4,
        channelAttention_reduce: Optional[int] = None,
    ):
        super().__init__()
        assert in_channels == out_channels, "MSADCAM expects in_channels == out_channels"

        if channelAttention_reduce is not None:
            channel_attention_reduce = channelAttention_reduce

        reduce = max(int(channel_attention_reduce), 1)
        internal = max(in_channels // reduce, 1)

        self.ca = ChannelAttentionMSADCAM(in_channels, internal)

        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=True)
        self.act = nn.GELU()

        self.dconv5_5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels, bias=False)

        self.dconv1_7 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 7), padding=(0, 3), groups=in_channels, bias=False)
        self.dconv7_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(7, 1), padding=(3, 0), groups=in_channels, bias=False)

        self.dconv1_11 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 11), padding=(0, 5), groups=in_channels, bias=False)
        self.dconv11_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(11, 1), padding=(5, 0), groups=in_channels, bias=False)

        self.dconv1_21 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 21), padding=(0, 10), groups=in_channels, bias=False)
        self.dconv21_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(21, 1), padding=(10, 0), groups=in_channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.act(x)

        channel_gate = self.ca(x)
        x = channel_gate * x

        local_context = self.dconv5_5(x)

        strip_7 = self.dconv7_1(self.dconv1_7(local_context))
        strip_11 = self.dconv11_1(self.dconv1_11(local_context))
        strip_21 = self.dconv21_1(self.dconv1_21(local_context))

        context = local_context + strip_7 + strip_11 + strip_21

        spatial_gate = self.conv(context)
        out = spatial_gate * x
        out = self.conv(out)
        return out


# Backward-compatible names used by older scripts/checkpoints.
ChannelAttention_CPCA = ChannelAttentionMSADCAM
CPCA = MSADCAM


# ==========================================
# 2) GSCAM: Group-wise Shuffle Channel Attention Module
# ==========================================

class GSCAM(nn.Module):
    """
    Encoder channel recalibration module, exposed with the paper name GSCAM.

    The implementation keeps the original lightweight avg/max pooled channel
    gate used in this project to replace MobileNetV3 SE blocks.
    """

    def __init__(self, channels: int, r: int = 16):
        super().__init__()
        r = max(int(r), 1)
        mid = max(channels // r, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(channels, mid, 1, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(mid, channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def _branch(self, pooled: torch.Tensor) -> torch.Tensor:
        x = self.fc1(pooled)
        x = self.act(x)
        x = self.fc2(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_att = self._branch(self.avg_pool(x))
        max_att = self._branch(self.max_pool(x))
        weight = self.sigmoid(avg_att + max_att)
        return x * weight


CABBlock = GSCAM


# ==========================================
# 3) HFRM: Hybrid Feature Refinement Module
# ==========================================

class ChannelSpatialGate(nn.Module):
    """Complementary channel and spatial gates used by high-level HFRM stages."""

    def __init__(self, channels: int, r: int = 8):
        super().__init__()
        mid = max(channels // r, 1)

        self.cse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=True),
            nn.Sigmoid(),
        )
        self.sse = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.cse(x) + x * self.sse(x)


class LargeKernelRefinementBlock(nn.Module):
    """Large-kernel depthwise refinement block used inside HFRM."""

    def __init__(self, dim: int, k: int = 7, drop_path: float = 0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=k, padding=k // 2, groups=dim, bias=False)
        self.norm = nn.BatchNorm2d(dim)
        self.act = nn.GELU()
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, 1, bias=False)
        self.act2 = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, 1, bias=False)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.pwconv1(x)
        x = self.act2(x)
        x = self.pwconv2(x)
        return residual + self.drop_path(x)


class AdaptiveSkipFusion(nn.Module):
    """Adaptive decoder/encoder skip fusion used as the first step of HFRM."""

    def __init__(self, dim: int):
        super().__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        mid = max(dim // 4, 1)
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, mid, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, dim, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x_dec: torch.Tensor, x_enc: torch.Tensor) -> torch.Tensor:
        if x_dec.size()[2:] != x_enc.size()[2:]:
            x_dec = F.interpolate(x_dec, size=x_enc.shape[2:], mode="bilinear", align_corners=False)
        out = torch.cat([x_dec, x_enc], dim=1)
        out = self.fusion_conv(out)
        weight = self.att(out)
        return out * weight + x_dec


class DecoderRefinement(nn.Module):
    """Projection plus two large-kernel refinement blocks."""

    def __init__(self, in_channels: int, out_channels: int, drop_path: float = 0.0):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.block = nn.Sequential(
            LargeKernelRefinementBlock(out_channels, k=7, drop_path=drop_path),
            LargeKernelRefinementBlock(out_channels, k=7, drop_path=drop_path),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return self.block(x)


class HFRM(nn.Module):
    """
    Hybrid Feature Refinement Module.

    It bundles the project's existing adaptive skip fusion, large-kernel decoder
    refinement, and optional channel-spatial recalibration under the paper name.
    """

    def __init__(self, dim: int, drop_path: float = 0.0, gate_r: int = 8, use_channel_spatial_gate: bool = False):
        super().__init__()
        self.adaptive_fusion = AdaptiveSkipFusion(dim)
        self.large_kernel_refine = DecoderRefinement(dim, dim, drop_path=drop_path)
        self.channel_spatial_gate = (
            ChannelSpatialGate(dim, r=gate_r) if use_channel_spatial_gate else nn.Identity()
        )

    def forward(self, x_dec: torch.Tensor, x_enc: torch.Tensor) -> torch.Tensor:
        x = self.adaptive_fusion(x_dec, x_enc)
        x = self.large_kernel_refine(x)
        return self.channel_spatial_gate(x)


# Backward-compatible class names used by older utility scripts.
SCSEBlock = ChannelSpatialGate
CMUNeXtBlock = LargeKernelRefinementBlock
CCMFusion = AdaptiveSkipFusion
DoubleConv_CMU = DecoderRefinement


# ==========================================
# 4) WetBird-Net
# ==========================================

class WetBirdNet(nn.Module):
    """
    WetBird-Net for binary wetland waterbird segmentation.

    Architecture names follow the paper: MobileNetV3-Large encoder with GSCAM,
    MS-ADCAM bottleneck, and HFRM decoder refinement.
    """

    def __init__(
        self,
        n_classes: int = 1,
        pretrained: bool = False,
        normalize_input: bool = False,
        drop_path: float = 0.0,
        scse_r: int = 8,
        gscam_r: int = 16,
        cab_r: Optional[int] = None,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.normalize_input = normalize_input
        self.gscam_r = int(cab_r) if cab_r is not None else int(gscam_r)

        # Optional ImageNet normalization inside model. Keep disabled when the
        # dataset loader only scales pixels to [0, 1], as in this project.
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        self.backbone = models.mobilenet_v3_large(weights=weights)
        self._replace_se_with_gscam(self.backbone, r=self.gscam_r)

        self.extract_indices = [1, 3, 6, 12, 16]
        self.enc_channels = [16, 24, 40, 112, 960]

        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.skip_proj4 = nn.Conv2d(self.enc_channels[3], 256, 1, bias=False)
        self.conv_in4 = nn.Conv2d(self.enc_channels[4], 256, 1, bias=False)
        self.ms_adcam_bottleneck = MSADCAM(in_channels=256, out_channels=256, channel_attention_reduce=4)
        self.hfrm4 = HFRM(256, drop_path=drop_path, gate_r=scse_r, use_channel_spatial_gate=True)

        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.skip_proj3 = nn.Conv2d(self.enc_channels[2], 128, 1, bias=False)
        self.conv_in3 = nn.Conv2d(256, 128, 1, bias=False)
        self.hfrm3 = HFRM(128, drop_path=drop_path, gate_r=scse_r, use_channel_spatial_gate=True)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.skip_proj2 = nn.Conv2d(self.enc_channels[1], 64, 1, bias=False)
        self.conv_in2 = nn.Conv2d(128, 64, 1, bias=False)
        self.hfrm2 = HFRM(64, drop_path=drop_path, gate_r=scse_r, use_channel_spatial_gate=False)

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.skip_proj1 = nn.Conv2d(self.enc_channels[0], 64, 1, bias=False)
        self.conv_in1 = nn.Conv2d(64, 64, 1, bias=False)
        self.hfrm1 = HFRM(64, drop_path=drop_path, gate_r=scse_r, use_channel_spatial_gate=False)

        self.final_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.outc = nn.Conv2d(64, n_classes, 1, bias=True)

    def _replace_se_with_gscam(self, model: nn.Module, r: int = 16) -> None:
        from torchvision.ops.misc import SqueezeExcitation as SE

        for name, child in model.named_children():
            if isinstance(child, SE):
                in_channels = child.fc1.in_channels
                setattr(model, name, GSCAM(in_channels, r=r))
            else:
                self._replace_se_with_gscam(child, r=r)

    def _replace_se_with_cab(self, model: nn.Module, r: int = 16) -> None:
        self._replace_se_with_gscam(model, r=r)

    @staticmethod
    def _upgrade_legacy_state_dict_names(state_dict):
        if not isinstance(state_dict, (dict, OrderedDict)):
            return state_dict

        replacements = (
            ("cpca_bottleneck.", "ms_adcam_bottleneck."),
            ("ccm4.", "hfrm4.adaptive_fusion."),
            ("dec4.", "hfrm4.large_kernel_refine."),
            ("scse_d4.", "hfrm4.channel_spatial_gate."),
            ("ccm3.", "hfrm3.adaptive_fusion."),
            ("dec3.", "hfrm3.large_kernel_refine."),
            ("scse_d3.", "hfrm3.channel_spatial_gate."),
            ("ccm2.", "hfrm2.adaptive_fusion."),
            ("dec2.", "hfrm2.large_kernel_refine."),
            ("ccm1.", "hfrm1.adaptive_fusion."),
            ("dec1.", "hfrm1.large_kernel_refine."),
        )

        upgraded = OrderedDict()
        changed = False
        for key, value in state_dict.items():
            new_key = key
            for old, new in replacements:
                if old in new_key:
                    new_key = new_key.replace(old, new)
            changed = changed or (new_key != key)
            upgraded[new_key] = value
        return upgraded if changed else state_dict

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        state_dict = self._upgrade_legacy_state_dict_names(state_dict)
        try:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        except TypeError:
            return super().load_state_dict(state_dict, strict=strict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize_input:
            x = (x - self.img_mean) / self.img_std

        features = []
        t = x
        idx_set = set(self.extract_indices)

        for i, layer in enumerate(self.backbone.features):
            t = layer(t)
            if i in idx_set:
                features.append(t)

        if len(features) != 5:
            raise RuntimeError(
                f"Expected 5 features but got {len(features)}. "
                f"extract_indices={self.extract_indices}. "
                f"Your torchvision MobileNetV3 structure may differ."
            )

        x1, x2, x3, x4, x5 = features

        x5p = self.conv_in4(x5)
        x5p = self.ms_adcam_bottleneck(x5p)

        d4 = self.hfrm4(self.up4(x5p), self.skip_proj4(x4))
        d3 = self.hfrm3(self.up3(self.conv_in3(d4)), self.skip_proj3(x3))
        d2 = self.hfrm2(self.up2(self.conv_in2(d3)), self.skip_proj2(x2))
        d1 = self.hfrm1(self.up1(self.conv_in1(d2)), self.skip_proj1(x1))

        return self.outc(self.final_up(d1))


# Keep the old public name so existing training/evaluation scripts continue to run.
Wetland_WT_UNet = WetBirdNet
