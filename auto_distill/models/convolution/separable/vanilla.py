from typing import Union, Dict

import torch.nn as nn
from torch.nn.common_types import _size_2_t


class Vanilla(nn.Module):
    '''
    Vanilla separable convolution, adapted from https://arxiv.org/pdf/1610.02357
    '''
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
        with_bn: bool = False,
        bn_kwargs: Dict = None,
    ):
        self.depthwise = nn.Conv2d(
            in_channels, 
            in_channels, 
            kernel_size=kernel_size, 
            groups=in_channels, 
            padding=padding, 
            padding_mode=padding_mode,
            dilation=dilation,
            stride=stride,
            bias=bias
        )

        if with_bn:
            self.norm = nn.BatchNorm2d(num_features=in_channels, **bn_kwargs)

        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            bias=False,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
        )
        super().__init__()

    def forward(self, x):
        x = self.depthwise(x)
        return self.pointwise(x)
