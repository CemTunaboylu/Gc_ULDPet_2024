from typing import List, Optional, Sequence, Union

import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F

import numpy as np

import einops

from adn import Activation, ADN, Normalization
from pooling import Pooling

LAST_LAYER = -1

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
    conv_opts = [F.Conv1d, F.Conv2d, F.Conv3d]
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
    if p < len or p > neg_len:
        return
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
        act: tuple = Activation.LRELU.activation(),
        norm: tuple | str | None = "INSTANCE",
        dropout: tuple | str | float | None = None,
        dropout_dim: int | None = 1,
        dilation: Sequence[int] | int = 1,
        groups: int = 1,
        bias: bool = True,
        padding: Sequence[int] | int | None = None,
        pooling : Pooling | None = None,
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
            pooling = pooling.pool(spatial_dims, pooling_kernel_size)
            self.add_module("pooling", pooling)

        if act is None and norm is None and dropout is None:
            return
        self.add_module(
            "adn",
            ADN(
                ordering=adn_ordering,
                in_channels=out_channels,
                act=act,
                norm=norm,
                norm_dim=self.spatial_dims,
                dropout=dropout,
                dropout_dim=dropout_dim,
            ),
        )

class Encoder(nn.Sequential):
    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            actv: Activation,
            num_convs: int = 2,
            norm: Normalization = Normalization.INSTANCE,
            bias: bool = True,
            dropout: float = 0.0,
            pooling: Pooling = Pooling.AVG,
            pooling_kernel_size: int = 2,
        ):
        super().__init__()

        # pooling = pooling.pool(spatial_dims=spatial_dims, kernel_size=pooling_kernel_size)
        self.convs = NConv(
            spatial_dims,
            in_channels,
            out_channels,
            act=actv,
            norm=norm,
            bias=bias,
            num_conv=num_convs,
            dropout=dropout,
        )
        if pooling:
            pooling = pooling.pool(spatial_dims, pooling_kernel_size)
            self.add_module("pooling", pooling)

    def inject(self, gamma, beta):
        if self.is_filmed:
            self.convs.filmed.inject(gamma, beta)

    def encode(self, x: torch.Tensor)-> torch.Tensor:
        return self.convs(x)

    def down_sample(self, x: torch.Tensor)-> torch.Tensor:
        if pooling := self._modules.get("pooling"):
            return pooling(x)
        return x

""" Feature-wise Linear Modulation """
class FiLMIntercept(nn.Module):
    def __init__(self, seq:nn.Sequential, module_key='adn'):
        super().__init__()
        self.seq = seq
        """
            By the default, Convolution creates a nn.Sequential with ADN module within,
            forward pass pipeline thus includes ADN, and needs to be intercepted by the FiLM layer.
            The parent Block/Layer to contain this block, will have an interception module instead, which
            steals the ADN module from the desired convolution layer (default=last). Thus, during its forward pass,
            the parent will inject the current gamma and beta values to the wrapping intercept. Before the ADN module,
            which is removed from the _modules of Convolution module, will be applied after the intercept's application of FiLM.
                For 2Conv ->  x -> | Conv | ADN | Conv | FiLM | ADN(applied by intercept)
        """
        self.stolen_module = self.seq._modules.pop(module_key)
        self.gamma : torch.Tensor = torch.tensor([])
        self.beta : torch.Tensor = torch.tensor([])

    def inject(self, gamma:torch.Tensor, beta:torch.Tensor):
        self.gamma = gamma
        self.beta = beta

    # be lazy reshape when called
    def modulate(self, feature_map:torch.Tensor)->torch.Tensor:
        if 0 == self.gamma.shape[0] or 0 == self.beta.shape[0]:
            raise ValueError("Gamma and beta must be set before calling forward.")

        # gamma and beta have shape: (batch_size, num_features)
        # we want to reshape them to (batch_size, num_features, 1, 1, 1)
        reshape = 'b nf -> b nf 1 1 1'
        gamma = einops.rearrange(self.gamma, reshape)
        beta = einops.rearrange(self.beta, reshape)
        # reset them to be safe
        self.gamma = torch.tensor([])
        self.beta = torch.tensor([])

        return gamma * feature_map + beta

    def forward(self, x:torch.Tensor)->torch.Tensor:
        for m in [self.seq, self.modulate, self.stolen_module]:
            x = m(x)
        return x

    @staticmethod
    def module_name()->str:
        return 'filmed'

class NConv(nn.Sequential):
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: Activation,
        norm: Normalization,
        bias: bool,
        num_conv : int = 2,
        dropout: Union[float, tuple] = 0.0,
        padding: int = 1,
        kernel_size:int = 3,
        ):
        super().__init__()

        # ADN
        act = act.activation()
        norm = norm.norm()

        self.num_conv = num_conv
        for i in range(num_conv):
            # TODO: check ADN ordering default is NDA
            c = Convolution(spatial_dims,
                    in_chns if i == 0 else out_chns,
                    out_chns,
                    kernel_size=kernel_size,
                    act=act,
                    norm=norm,
                    bias=bias,
                    dropout=dropout,
                    # pooling=pooling,
                    # pooling_kernel_size=pooling_kernel_size,
                    padding=padding)
            self.add_module(self.naming(i),c)


    def inject(self, gamma:Tensor, beta:Tensor):
        # TODO: this can emit an error
        if not self.is_filmed: return
        filmed_elm_key = FiLMIntercept.module_name()
        film_intercept: FiLMIntercept = self._modules.get(filmed_elm_key)
        film_intercept.inject(gamma, beta)

    def naming(self, i:int)->str:
        return f"conv-{i}"

def inject_film_wrapper_to(block:Union[Encoder, NConv], to_intercept:int=LAST_LAYER, module_key='adn'):
    if isinstance(block, Encoder):
        inject_film_wrapper_to(block.convs, to_intercept, module_key)
        block.__setattr__('is_filmed', True)
    elif isinstance(block, NConv):
        validate_interception_point(to_intercept, block.num_conv)
        if to_intercept < 0:
            to_intercept += block.num_conv
        mod_name = block.naming(to_intercept)
        conv = block._modules.get(mod_name)
        block._modules.pop(mod_name)
        block.__setattr__('is_filmed', True)
        fi = FiLMIntercept(conv, module_key)
        block.add_module(FiLMIntercept.module_name(), fi)
    else:
        raise ValueError(f'FiLM Layer injection for {block} is not implemented')
