# Copyright (c) Meta Platforms, Inc. and affiliates.
# This derivative is distributed under the DINOv3 License Agreement.
# See LICENSE.md and source.yml in this directory.
"""Offline ConvNeXt-Large teacher; official parameter names, stage-2 forward.

The complete parameter tree is retained for strict loading of the official
backbone checkpoint. Forward evaluates only stages 0 and 1; no hub or download
code, classifier, training recipe, or ViT implementation is included.
"""

import torch
from torch import nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format='channels_last'):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == 'channels_last':
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mean = x.mean(1, keepdim=True)
        variance = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(variance + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))
        self.drop_path = nn.Identity()

    def forward(self, x):
        residual = x
        x = self.dwconv(x).permute(0, 2, 3, 1)
        x = self.pwconv2(self.act(self.pwconv1(self.norm(x))))
        x = self.gamma * x
        return residual + self.drop_path(x.permute(0, 3, 1, 2))


class ConvNeXtLarge(nn.Module):
    def __init__(self):
        super().__init__()
        dims = [192, 384, 768, 1536]
        depths = [3, 3, 27, 3]
        self.downsample_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
                LayerNorm(dims[0], eps=1e-6, data_format='channels_first'),
            )
        ])
        for index in range(3):
            self.downsample_layers.append(nn.Sequential(
                LayerNorm(dims[index], eps=1e-6, data_format='channels_first'),
                nn.Conv2d(dims[index], dims[index + 1], kernel_size=2, stride=2),
            ))
        self.stages = nn.ModuleList([
            nn.Sequential(*[Block(dim) for _ in range(depth)])
            for dim, depth in zip(dims, depths)
        ])
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Identity()
        # The official state_dict contains this alias of the final norm too.
        self.norms = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity(), self.norm])

    def forward(self, x):
        for index in range(2):
            x = self.stages[index](self.downsample_layers[index](x))
        return x


def convnext_large():
    return ConvNeXtLarge()
