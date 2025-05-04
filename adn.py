from typing import Tuple
from enum import Enum

import torch
from torch import nn
F = nn.functional

from sys import path 

from MONAI.monai.networks.layers.factories import Act,  Dropout, Norm, Pool, split_args
from MONAI.monai.utils import has_option

# from monai.networks.layers import Dropout, Act, Norm

def get_activation(activation:str):
    a = None
    if activation == 'relu':
        a= nn.ReLU(inplace=True)
    elif activation == 'leaky_relu':
        a= nn.LeakyReLU(inplace=True)
    elif activation == 'elu':
        a= nn.ELU(inplace=True)
    elif activation == 'prelu':
        a= nn.PReLU()
    elif activation == 'swish':
        a= nn.SiLU(inplace=True)  # Swish activation
    else:
        raise ValueError(f"Unsupported activation: {activation}")
    return a

class Activation(Enum):
    LRELU = 0
    PRELU = 1
    RELU = 2
    ELU = 3
    SWISH = 4

    def __str__(self):
        return self.name

    def activation(self, negative_slope=0.2, num_parameters=1, init=0.25)->Tuple:
        a = [("LeakyReLU", {"negative_slope": negative_slope, "inplace": True}),
             ("PReLU", {"num_parameters": num_parameters, "init": init}),
             "ReLU"]
        if self.value >= len(a):
            raise NotImplementedError(f"activation {self} is not implemented")

        return a[self.value]

class Normalization(Enum):
    INSTANCE = 0
    BATCH = 1
    INSTANCE_NVFUSER = 2
    LOCALRESPONSE = 3
    LAYER = 4
    GROUP = 5
    SYNCBATCH = 6

    def norm(self)->str:
        n = [('INSTANCE', {"affine":True}),
             'BATCH', 'INSTANCE_NVFUSER', 'LOCALRESPONSE', 'LAYER',
              ('GROUP', {"num_groups": 4}), 'SYNCBATCH']
        if self.value >= len(n):
            raise NotImplementedError(f"normalization {self} is not implemented")

        return n[self.value]

def get_norm_layer(name: tuple | str, spatial_dims: int | None = 1, channels: int | None = 1):
    """
        from monai.networks.layers import get_norm_layer

        g_layer = get_norm_layer(name=("group", {"num_groups": 1}))
        n_layer = get_norm_layer(name="instance", spatial_dims=2)

    Args:
        name: a normalization type string or a tuple of type string and parameters.
        spatial_dims: number of spatial dimensions of the input.
        channels: number of features/channels when the normalization layer requires this parameter
            but it is not specified in the norm parameters.
    """
    if name == "":
        return torch.nn.Identity()
    norm_name, norm_args = split_args(name)
    norm_type = Norm[norm_name, spatial_dims]
    kw_args = dict(norm_args)
    if has_option(norm_type, "num_features") and "num_features" not in kw_args:
        kw_args["num_features"] = channels
    if has_option(norm_type, "num_channels") and "num_channels" not in kw_args:
        kw_args["num_channels"] = channels
    return norm_type(**kw_args)

def get_act_layer(name: tuple | str):
    """
        from monai.networks.layers import get_act_layer

        s_layer = get_act_layer(name="swish")
        p_layer = get_act_layer(name=("prelu", {"num_parameters": 1, "init": 0.25}))

    Args:
        name: an activation type string or a tuple of type string and parameters.
    """
    if name == "":
        return torch.nn.Identity()
    act_name, act_args = split_args(name)
    act_type = Act[act_name]
    return act_type(**act_args)

def get_dropout_layer(name: tuple | str | float | int, dropout_dim: int | None = 1):
    """
        from monai.networks.layers import get_dropout_layer

        d_layer = get_dropout_layer(name="dropout")
        a_layer = get_dropout_layer(name=("alphadropout", {"p": 0.25}))

    Args:
        name: a dropout ratio or a tuple of dropout type and parameters.
        dropout_dim: the spatial dimension of the dropout operation.
    """
    if name == "":
        return torch.nn.Identity()
    if isinstance(name, (int, float)):
        # if dropout was specified simply as a p value, use default name and make a keyword map with the value
        drop_name = Dropout.DROPOUT
        drop_args = {"p": float(name)}
    else:
        drop_name, drop_args = split_args(name)
    drop_type = Dropout[drop_name, dropout_dim]
    return drop_type(**drop_args)


def get_pool_layer(name: tuple | str, spatial_dims: int | None = 1):
    """
    Create a pooling layer instance.

    For example, to create adaptiveavg layer:

    .. code-block:: python

        from monai.networks.layers import get_pool_layer

        pool_layer = get_pool_layer(("adaptiveavg", {"output_size": (1, 1, 1)}), spatial_dims=3)

    Args:
        name: a pooling type string or a tuple of type string and parameters.
        spatial_dims: number of spatial dimensions of the input.

    """
    if name == "":
        return torch.nn.Identity()
    pool_name, pool_args = split_args(name)
    pool_type = Pool[pool_name, spatial_dims]
    return pool_type(**pool_args)


class ADN(nn.Sequential):
    def __init__(
            self,
            ordering='NDA',
            in_channels: int | None = None,
            act: tuple | str | None = "RELU",
            norm: tuple | str | None = None,
            norm_dim: int | None = None,
            dropout: tuple | str | float | None = None,
            dropout_dim: int | None = None):
        super().__init__()
        op_dict = {"A": None, "D": None, "N": None}
        # define the normalization type and the arguments to the constructor
        if norm is not None:
            if norm_dim is None and dropout_dim is None:
                raise ValueError("norm_dim or dropout_dim needs to be specified.")
            op_dict["N"] = get_norm_layer(name=norm, spatial_dims=norm_dim or dropout_dim, channels=in_channels)

        # define the activation type and the arguments to the constructor
        if act is not None:
            op_dict["A"] = get_act_layer(act)

        if dropout is not None:
            if norm_dim is None and dropout_dim is None:
                raise ValueError("norm_dim or dropout_dim needs to be specified.")
            op_dict["D"] = get_dropout_layer(name=dropout, dropout_dim=dropout_dim or norm_dim)

        for item in ordering.upper():
            if item not in op_dict:
                raise ValueError(f"ordering must be a string of {op_dict}, got {item} in it.")
            if op_dict[item] is not None:
                self.add_module(item, op_dict[item])