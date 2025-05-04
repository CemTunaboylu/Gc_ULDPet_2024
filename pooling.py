from enum import Enum

import torch
from torch import nn
from torch.nn import functional as F

from MONAI.monai.networks.layers.factories import Pool


# TODO: Adaptive pooling + LP max min
class Pooling(Enum):
    MAX = 0
    AVG = 1

    def __str__(self):
        return self.name

    def pool(self, spatial_dims:int, kernel_size:int=2):
        pools = [
            Pool[Pooling.MAX.name, spatial_dims](kernel_size=kernel_size),
            Pool[Pooling.AVG.name, spatial_dims](kernel_size=kernel_size)
            ]
        if self.value >= len(pools):
            raise NotImplementedError(f"pooling {self} is not implemented")

        return pools[self.value]

"""
    Pooling with different kernel sizes and/or strides along each spatial dimensions
    to capture features at various scales (useful when data has inherent directional structure)
"""
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