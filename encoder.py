from functools import partial 
from typing import Callable, Union

import torch
from torch import nn

from adn import Activation, Normalization
from layers import NConv, validate_interception_point
from film import FiLMIntercept, LAST_LAYER

class Encoder(nn.Sequential):
    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            pooling: Callable | None = None,
            act: Activation = Activation.lrelu(),
            norm: Callable = partial(Normalization.instance),
            num_convs: int = 2,
            bias: bool = True,
            dropout_dim: int | None = None,
            dropout_prob: float | None = None,
            pooling_kernel_size: int = 2,
        ):
        super().__init__()

        self.convs = NConv(
            spatial_dims,
            in_channels,
            out_channels,
            act=act,
            norm=norm,
            bias=bias,
            num_conv=num_convs,
            dropout_dim=dropout_dim,
            dropout_prob=dropout_prob,
        )
        if pooling:
            pool = pooling(dim=spatial_dims, kernel_size=pooling_kernel_size)
            self.add_module("pooling", pool)

    # inject gamma and beta to the FiLM layer if existsyy
    def inject(self, gamma, beta):
        if self.is_filmed:
            self.convs.filmed.inject(gamma, beta)

    def encode(self, x: torch.Tensor)-> torch.Tensor:
        return self.convs(x)

    def down_sample(self, x: torch.Tensor)-> torch.Tensor:
        if pooling := self._modules.get("pooling"):
            return pooling(x)
        return x

def inject_film_wrapper_to(block:Union[Encoder, NConv], to_intercept:int=LAST_LAYER, module_key='adn', module_name: str = 'is_filmed'):
    if isinstance(block, Encoder):
        inject_film_wrapper_to(block.convs, to_intercept, module_key, module_name)
        block.__setattr__(module_name, True)
    elif isinstance(block, NConv):
        validate_interception_point(to_intercept, block.num_conv)
        if to_intercept < 0:
            to_intercept += block.num_conv
        mod_name = block.naming(to_intercept)
        conv = block._modules.get(mod_name)
        block._modules.pop(mod_name)
        block.__setattr__(module_name, True)
        fi = FiLMIntercept(conv, module_key)
        block.add_module(FiLMIntercept.module_name(), fi)
    else:
        raise ValueError(f'FiLM Layer injection for {block} is not implemented')