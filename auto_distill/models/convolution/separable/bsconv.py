from typing import Union, Dict

import math

import torch.nn as nn
from torch.nn.common_types import _size_2_t


'''
Adapted from: https://arxiv.org/abs/2003.13549
'''

class BSConvU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        normalization: nn.Module = None,
        norm_kwargs: Dict = None,
    ):
        # pointwise
        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        if normalization:
            self.norm = normalization(num_features=out_channels, **norm_kwargs)

        # depthwise
        self.depthwise = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=out_channels,
            bias=bias,
            padding_mode=padding_mode,
        )
        super().__init__()

    def forward(self, x):
        x = self.depthwise(x)
        return self.pointwise(x)


class BSConvS(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        p: int = 0.25,
        min_mid_channels:int = 4,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        normalization: nn.Module = None,
        norm_kwargs: Dict = None,
    ):
        assert 0.0 <= p <= 1.0

        mid_channels = min(in_channels, max(min_mid_channels, math.ceil(p * in_channels)))
        self.pointwise1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        self.pointwise2 = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        self.depthwise = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=out_channels,
            bias=bias,
            padding_mode=padding_mode,
        )

        if normalization:
            self.norm1 = normalization(num_features=mid_channels, **norm_kwargs)
            self.norm2 = normalization(num_features=out_channels, **norm_kwargs)

        super().__init__()

    def forward(self, x):
        x = self.pointwise1(x)
        if hasattr(self, "norm1"):
            x = self.norm1(x)

        x = self.pointwise2(x)
        if hasattr(self, "norm2"):
            x = self.norm2(x)

        x = self.depthwise(x)
        return x
