# SPDX-License-Identifier: Apache-2.0
# Adapted from RepViT by Ao Wang et al., https://github.com/THU-MIG/RepViT
# Reference: third_party/repvit/model/repvit.py; license: third_party/repvit/LICENSE.
# SE rounding follows timm 0.5.4 (Copyright 2021 Ross Wightman, Apache-2.0).
# Modified for DGIM: grayscale input, exact 128/196/256/256 channels, strides
# 2/4/8/8, depths 2/4/28/2, Kaiming initialization, and no classifier or timm.
"""Fixed RepViT feature backbone; this is not the official RepViT-M2.3."""

import torch
from torch import nn
from torch.nn import functional as F


def _conv_like(conv):
    """Create a biased convolution on the same device and in the same dtype."""
    return nn.Conv2d(
        conv.in_channels, conv.out_channels, conv.kernel_size,
        stride=conv.stride, padding=conv.padding, dilation=conv.dilation,
        groups=conv.groups, bias=True, padding_mode=conv.padding_mode,
    ).to(device=conv.weight.device, dtype=conv.weight.dtype).eval()


class ConvBN(nn.Sequential):
    def __init__(self, inp, oup, kernel_size=1, stride=1, padding=0,
                 groups=1, zero_init=False):
        super().__init__()
        self.add_module('conv', nn.Conv2d(
            inp, oup, kernel_size, stride, padding, groups=groups, bias=False))
        self.add_module('bn', nn.BatchNorm2d(oup))
        self.zero_init = zero_init

    @torch.no_grad()
    def fuse(self):
        if self.training:
            raise RuntimeError('ConvBN fusion requires eval mode.')
        scale = self.bn.weight * torch.rsqrt(self.bn.running_var + self.bn.eps)
        fused = _conv_like(self.conv)
        fused.weight.copy_(self.conv.weight * scale[:, None, None, None])
        fused.bias.copy_(self.bn.bias - self.bn.running_mean * scale)
        return fused


class RepVGGDW(nn.Module):
    """Training: DW 3x3 + DW 1x1 + identity, then an outer BN."""
    def __init__(self, channels):
        super().__init__()
        self.conv3 = ConvBN(channels, channels, 3, padding=1, groups=channels)
        self.conv1 = nn.Conv2d(channels, channels, 1, groups=channels, bias=True)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        return self.bn(self.conv3(x) + self.conv1(x) + x)

    @torch.no_grad()
    def fuse(self):
        if self.training:
            raise RuntimeError('RepVGGDW fusion requires eval mode.')
        conv3 = self.conv3.fuse()
        identity = torch.zeros_like(conv3.weight)
        identity[:, 0, 1, 1] = 1
        kernel = conv3.weight + F.pad(self.conv1.weight, (1, 1, 1, 1)) + identity
        bias = conv3.bias + self.conv1.bias
        scale = self.bn.weight * torch.rsqrt(self.bn.running_var + self.bn.eps)
        fused = _conv_like(conv3)
        fused.weight.copy_(kernel * scale[:, None, None, None])
        fused.bias.copy_(self.bn.bias + (bias - self.bn.running_mean) * scale)
        return fused


class SqueezeExcite(nn.Module):
    """SE with ratio 1/4; only the internal SE width is rounded to eight."""
    def __init__(self, channels):
        super().__init__()
        reduced = max(8, int(channels * 0.25 + 4) // 8 * 8)
        self.fc1 = nn.Conv2d(channels, reduced, 1)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced, channels, 1)

    def forward(self, x):
        gate = x.mean((2, 3), keepdim=True)
        gate = self.fc2(self.act(self.fc1(gate))).sigmoid()
        return x * gate


class Residual(nn.Module):
    def __init__(self, branch):
        super().__init__()
        self.branch = branch

    def forward(self, x):
        return x + self.branch(x)


class RepViTBlock(nn.Module):
    def __init__(self, inp, oup, stride=1, use_se=False):
        super().__init__()
        if stride not in (1, 2):
            raise ValueError('RepViT block stride must be 1 or 2.')
        if stride == 1 and inp != oup:
            raise ValueError('A regular RepViT block must preserve channels.')
        if stride == 2:
            self.token_mixer = nn.Sequential(
                ConvBN(inp, inp, 3, stride=2, padding=1, groups=inp),
                SqueezeExcite(inp) if use_se else nn.Identity(),
                ConvBN(inp, oup),
            )
        else:
            self.token_mixer = nn.Sequential(
                RepVGGDW(inp),
                SqueezeExcite(inp) if use_se else nn.Identity(),
            )
        self.channel_mixer = Residual(nn.Sequential(
            ConvBN(oup, 2 * oup),
            nn.GELU(),
            ConvBN(2 * oup, oup, zero_init=True),
        ))

    def forward(self, x):
        return self.channel_mixer(self.token_mixer(x))


class RepViT_8_2(nn.Module):
    """Kaiming-initialized grayscale backbone with DGIM's feature interface.

    Stage 4 only groups the last two ordinary stride-one blocks of the
    30-block coarse sequence. Its SE pattern remains off/on; no additional
    transition, projection, tail block, or ImageNet classifier is inserted.
    """
    DEPTHS = (2, 4, 28, 2)
    STAGE_DIMS = (128, 196, 256, 256)
    OUT_STRIDES = (2, 4, 8, 8)

    def __init__(self, config, deploy=False):
        super().__init__()
        if tuple(config['block_dims']) != (128, 196, 256):
            raise ValueError('This RepViT backbone requires BLOCK_DIMS=[128,196,256].')
        self.deploy = False
        self.stem = nn.Sequential(
            ConvBN(1, 64, 3, stride=2, padding=1),
            nn.GELU(),
            ConvBN(64, 128, 3, stride=1, padding=1),
        )
        self.stage1 = nn.Sequential(
            RepViTBlock(128, 128, use_se=True),
            RepViTBlock(128, 128, use_se=False),
        )
        self.stage2 = nn.Sequential(
            RepViTBlock(128, 196, stride=2),
            RepViTBlock(196, 196, use_se=True),
            RepViTBlock(196, 196, use_se=False),
            RepViTBlock(196, 196, use_se=True),
        )
        # Keep the first 28 entries of the original coarse SE sequence:
        # transition(off), ordinary(on), ordinary(off), ... ordinary(on).
        self.stage3 = nn.Sequential(
            RepViTBlock(196, 256, stride=2),
            *[RepViTBlock(256, 256, use_se=(index % 2 == 1))
              for index in range(1, 28)],
        )
        self.stage4 = nn.Sequential(
            RepViTBlock(256, 256, use_se=False),
            RepViTBlock(256, 256, use_se=True),
        )
        self._initialize_weights()
        if deploy:
            self.eval()
            self.switch_to_deploy()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        # Do not overwrite RepViT's zero-initialized channel residual branch.
        for module in self.modules():
            if isinstance(module, ConvBN) and module.zero_init:
                nn.init.zeros_(module.bn.weight)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError('RepViT_8_2 expects grayscale input [B,1,H,W].')
        if x.shape[-2] % 8 or x.shape[-1] % 8:
            raise ValueError('DGIM input height and width must be divisible by eight.')
        x1 = self.stage1(self.stem(x))
        x2 = self.stage2(x1)
        coarse = self.stage4(self.stage3(x2))
        return {'feats_c': coarse, 'feats_f': None,
                'feats_x2': x2, 'feats_x1': x1}

    @torch.no_grad()
    def switch_to_deploy(self):
        """Permanently fuse trained BN/RepVGGDW layers for inference, in place."""
        if self.deploy:
            return self
        if self.training:
            raise RuntimeError('Call eval() before converting the RepViT backbone.')

        def fuse_children(parent):
            for name, child in list(parent.named_children()):
                # Fuse the entire DW block before visiting its inner ConvBN.
                if isinstance(child, (RepVGGDW, ConvBN)):
                    setattr(parent, name, child.fuse())
                else:
                    fuse_children(child)

        fuse_children(self)
        self.deploy = True
        return self
