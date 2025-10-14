from typing import Callable, Tuple
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
from rust_enum import enum, Case

@enum
class Pooling:
    MAX = Case(dim=int, ceil_mode=bool, return_indices=bool, dilation=int, padding=int, stride=int, kernel_size=int)
    ADAPTIVE_MAX = Case(dim=int, output_size=int|Tuple[int, ...])
    AVG = Case(dim=int, ceil_mode=bool, count_include_pad=bool, padding=int, stride=int, kernel_size=int)
    ADAPTIVE_AVG = Case(dim=int, output_size=int|Tuple[int, ...])
    LP = Case(dim=int, ceil_mode=bool, stride=int, kernel_size=int)

    @staticmethod
    def avg_f(stride=1, padding=1, ceil_mode=False, dilation=1,  return_indices=False)->Callable:
        return partial(Pooling.AVG, stride=stride, padding=padding, ceil_mode=ceil_mode,  dilation=dilation, return_indices=return_indices)

    def get_layer(self) -> nn.Module:
        p : nn.Module
        allowed_dims = (1,2,3)
        def allowed_dim_assertion(dim):
            if dim not in allowed_dims:
                raise ValueError(f"dim must be in {allowed_dims}.")
        match self:
            case Pooling.MAX(dim, ceil_mode, return_indices, dilation, padding, stride, kernel_size):
                allowed_dim_assertion(dim)
                pooling = (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)[dim-1]
                p = pooling(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, ceil_mode=ceil_mode, return_indices=return_indices)
            case Pooling.ADAPTIVE_MAX(dim, output_size):
                allowed_dim_assertion(dim)
                pooling = (nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d)[dim-1]
                p = pooling(output_size=output_size)
            case Pooling.AVG(dim, ceil_mode, count_include_pad, padding, stride, kernel_size):
                allowed_dim_assertion(dim)
                pooling = (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)[dim-1]
                p = pooling(kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad)
            case Pooling.ADAPTIVE_AVG(dim, output_size):
                allowed_dim_assertion(dim)
                pooling = (nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d)[dim-1]
                p = pooling(output_size=output_size)
            case Pooling.LP(dim, ceil_mode, stride, kernel_size):
                allowed_dim_assertion(dim)
                pooling = (nn.LPPool1d, nn.LPPool2d, nn.LPPool3d)[dim-1]
                p = pooling(kernel_size=kernel_size, stride=stride, ceil_mode=ceil_mode)
        return p

# TODO: this can be introduced when the next pooling will result in an odd depth
"""
    Pooling with different kernel sizes and/or strides along each spatial dimensions
    to capture features at various scales (useful when data has inherent directional structure)
"""
def anisotropic_pooling():
    return nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))

"""
    Pooling with multiple poolings with a learnable scalar
"""
class MixedPooling(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, initial_alpha=0.5):
        super(MixedPooling, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        # Define alpha as a learnable parameter
        self.alpha = nn.Parameter(torch.tensor(initial_alpha))

    def forward(self, x):
        max_pool = F.max_pool3d(
            x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding
        )
        avg_pool = F.avg_pool3d(
            x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding
        )
        # Apply sigmoid to keep alpha between 0 and 1
        alpha = torch.sigmoid(self.alpha)
        return alpha * max_pool + (1 - alpha) * avg_pool