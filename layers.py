from typing import Callable, List, Optional, Sequence, Union
from functools import partial

import torch
from torch import dropout_, nn
from torch import Tensor
from torch.nn import functional as F
import numpy as np
import einops

from adn import Activation, ADN, Normalization
from film import FiLMIntercept
from pooling import Pooling

def __get_conv_for_dim(spatial_dims:int, conv_opts:List)->nn.Module:
    if spatial_dims <= 0:
        raise ValueError(f'Spatial dimensions can only be positive but got {spatial_dims}')
    if len(conv_opts) < spatial_dims:
        raise NotImplementedError(f'Convolutions for {spatial_dims}-D is not implemented')
    return conv_opts[spatial_dims-1]

def get_conv_for_dim(spatial_dims:int)->nn.Module:
    conv_opts = [nn.Conv1d, nn.Conv2d, nn.Conv3d]
    return __get_conv_for_dim(spatial_dims, conv_opts)

def get_f_conv_for_dim(spatial_dims:int)->nn.Module:
    conv_opts = [F.conv1d, F.conv2d, F.conv3d]
    return __get_conv_for_dim(spatial_dims, conv_opts)

def get_conv_for_dim_from(spatial_dims:int, from_module)->nn.Module:
    conv_opts : List   
    try:
        conv_opts = [from_module.Conv1d, from_module.Conv2d, from_module.Conv3d]
    except AttributeError:
        conv_opts = [from_module.conv1d, from_module.conv2d, from_module.conv3d]
    return __get_conv_for_dim(spatial_dims, conv_opts)

# stolen from monai.networks.layers.convutils
def same_padding(kernel_size: Sequence[int] | int, dilation: Sequence[int] | int = 1) -> tuple[int, ...] | int:
    """
    Return the padding value needed to ensure a convolution using the given kernel size produces an output of the same
    shape as the input for a stride of 1, otherwise ensure a shape of the input divided by the stride rounded down.

    Raises:
        NotImplementedError: When ``np.any((kernel_size - 1) * dilation % 2 == 1)``.
    """

    kernel_size_np = np.atleast_1d(kernel_size)
    dilation_np = np.atleast_1d(dilation)

    if np.any((kernel_size_np - 1) * dilation % 2 == 1):
        raise NotImplementedError(
            f"Same padding not available for kernel_size={kernel_size_np} and dilation={dilation_np}."
        )

    padding_np = (kernel_size_np - 1) / 2 * dilation_np
    padding = tuple(int(p) for p in padding_np)

    return padding if len(padding) > 1 else padding[0]

def validate_interception_point(p:int, len:int):
    neg_len = -len
    if p >= len or p < neg_len:
        raise ValueError(f'Layer injection index must be within bounds, but {p} is given')

class Convolution(nn.Sequential):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        strides: Sequence[int] | int = 1,
        kernel_size: Sequence[int] | int = 3,
        adn_ordering: str = 'NDA',
        act: Activation = Activation.lrelu(),
        norm: Callable = partial(Normalization.instance),
        dropout_dim: int | None = 1,
        dropout_prob: float | None = None,
        dilation: Sequence[int] | int = 1,
        groups: int = 1,
        bias: bool = True,
        padding: Sequence[int] | int | None = None,
        pooling : Callable | None = None,
        pooling_kernel_size = 2,
        output_padding: Sequence[int] | int | None = None,
    ) -> None:
        super().__init__()

        self.spatial_dims = spatial_dims
        conv_d = get_conv_for_dim(spatial_dims)

        self.in_channels = in_channels
        self.out_channels = out_channels

        if padding is None:
            padding = same_padding(kernel_size, dilation)

        conv: nn.Module = conv_d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=strides,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.add_module("conv", conv)

        if pooling:
            pool = pooling(dim=spatial_dims, kernel_size=pooling_kernel_size)
            self.add_module("pooling", pool)

        if not all(n is None for n in (act, norm, dropout_dim)):
            normalization = norm(num_features=out_channels) if norm else None
            self.add_module(
                "adn",
                ADN(
                    ordering=adn_ordering,
                    in_channels=out_channels,
                    act=act,
                    norm=normalization,
                    dropout_dim=dropout_dim,
                    dropout_prob=dropout_prob,
                ),
            )

class NConv(nn.Sequential):
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        bias: bool,
        act: Activation = Activation.lrelu(),
        norm: Callable = partial(Normalization.instance),
        num_conv : int = 2,
        dropout_dim : int | None = None,
        dropout_prob: float | None = None,
        padding: int = 1,
        kernel_size:int = 3,
        ):
        super().__init__()

        self.num_conv = num_conv
        for i in range(num_conv):
            c = Convolution(spatial_dims,
                    in_chns if i == 0 else out_chns,
                    out_chns,
                    kernel_size=kernel_size,
                    act=act,
                    norm=norm,
                    bias=bias,
                    dropout_dim=dropout_dim,
                    dropout_prob=dropout_prob,
                    # ! parameterize pooling
                    # pooling=pooling,
                    # pooling_kernel_size=pooling_kernel_size,
                    padding=padding)
            self.add_module(self.naming(i),c)


    # inject gamma and beta to the FiLM layer if existsyy
    def inject(self, gamma:Tensor, beta:Tensor):
        # TODO: this can emit an error
        if not self.is_filmed: return
        filmed_elm_key = FiLMIntercept.module_name()
        film_intercept: FiLMIntercept = self._modules.get(filmed_elm_key)
        film_intercept.inject(gamma, beta)

    def naming(self, i:int)->str:
        return f"conv-{i}"