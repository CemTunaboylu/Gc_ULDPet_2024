from re import U
from typing import Callable, List, Optional, Sequence, Tuple, Union
import torch
from torch import nn
F = nn.functional
import einops

from adn import Activation, ADN, Normalization 
from layers import Encoder, NConv, inject_film_wrapper_to
from monster import MetadataNetwork
from upscaler import UpSampling, ConcatenatingUpScaler

from collections import namedtuple

class Features:
    def __init__(self, encoder: Sequence[int]):
        self.encoder = encoder

    def __iter__(self):
        for in_f, out_f in zip(self.encoder[:-1], self.encoder[1:]):
            yield in_f, out_f

# TODO: I can also implement the loop with a method as an iterator to make it more readable

DEFAULT_FEATURES = Features(encoder=[32, 64, 128, 256])

class Unet3DEncoders(nn.Module):
    def __init__(
            self,
            spatial_dims: int = 3,
            in_channels: int = 1,
            features: Features = DEFAULT_FEATURES,
            actv: Activation = Activation.PRELU,
            norm: Normalization = Normalization.INSTANCE,
            bias: bool = True,
            dropout: float = 0.0,
            pre_init_f_inplace: Union[Callable[[Encoder], None],None] = None,
            ):
        super().__init__()
        encoders = nn.ModuleList()

        init = Encoder(spatial_dims, in_channels, features.encoder[0], actv, norm=norm, bias=bias, dropout=dropout)
        if pre_init_f_inplace is not None:
            pre_init_f_inplace(init)
        encoders.append(init)

        for in_f, out_f in features:
            e = Encoder(spatial_dims, in_f, out_f, actv, norm=norm, bias=bias, dropout=dropout)
            if pre_init_f_inplace is not None:
                pre_init_f_inplace(e)
            encoders.append(e)

        self.encoders = encoders

    def encode(self, x: torch.Tensor)-> Tuple[torch.Tensor, List[torch.Tensor]]:
        skips = [torch.empty(0)] * len(self.encoders)
        for ix, encoder in enumerate(self.encoders):
            x = encoder.encode(x)
            skips[ix] = x
            x = encoder.down_sample(x)

        return x, skips

    def forward(self, x: torch.Tensor)-> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        x: input should have spatially N dimensions
            ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N])``, N is defined by `dimensions`.
            It is recommended to have ``dim_n % 16 == 0`` to ensure all maxpooling inputs have
            even edge lengths.
        """
        return self.encode(x)

class Unet3DDecoders(nn.Module):
    def __init__(
            self,
            spatial_dims: int = 3,
            out_channels: int = 1,
            features: Features = DEFAULT_FEATURES,
            actv: Activation = Activation.PRELU,
            norm: Normalization = Normalization.INSTANCE,
            bias: bool = True,
            dropout: float = 0.0,
            up_sampling: UpSampling = UpSampling.DECONV,
            ):
        super().__init__()
        decoders = nn.ModuleList()
        for in_f, out_f in features:
            cus = ConcatenatingUpScaler(spatial_dims, in_chan = out_f, cat_chan = out_f, out_chan = in_f, actv = actv, 
                    up_sampling = up_sampling, norm = norm, bias = bias, dropout = dropout, halves = True)
                    # up_sampling=up_sampling, norm=norm, bias=bias, dropout=dropout, halves = False)
            decoders.insert(0, cus)

        # final = NConv(spatial_dims, features.encoder[0], out_channels, actv, norm, bias, dropout=dropout, pooling=None)
        final = ConcatenatingUpScaler(spatial_dims, in_chan = features.encoder[0], cat_chan = features.encoder[0], out_chan = out_channels, actv = actv, 
                    up_sampling = up_sampling, norm = norm, bias = bias, dropout = dropout, halves = True)
                    # up_sampling=up_sampling, norm=norm, bias=bias, dropout=dropout, halves=False)
        decoders.append(final)
        self.decoders = decoders

    def decode(self, x: torch.Tensor, skips: List[torch.Tensor])-> torch.Tensor:
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        return x

    def forward(self, x: torch.Tensor, skips: List[torch.Tensor])-> torch.Tensor:
        return self.decode(x,skips)

class UNet3D(nn.Module):
    def __init__(
            self,
            spatial_dims: int = 3,
            in_channels: int = 1,
            out_channels: int = 1,
            features: Features = DEFAULT_FEATURES,
            actv: Activation = Activation.PRELU,
            norm: Normalization = Normalization.INSTANCE,
            bias: bool = True,
            dropout: float = 0.0,
            up_sampling: UpSampling = UpSampling.DECONV,
            ):
        super().__init__()

        self.encoders = Unet3DEncoders(
            spatial_dims=spatial_dims, 
            in_channels=in_channels, 
            features=features, 
            actv=actv,
            norm=norm,
            bias=bias,
            dropout=dropout)
        self.add_module("encoders", self.encoders)

        bottleneck_chan = features.encoder[-1]
        self.bottleneck = NConv(spatial_dims, bottleneck_chan, bottleneck_chan, actv, norm, bias, dropout=dropout)
        self.add_module("bottleneck", self.bottleneck)

        self.decoders = Unet3DDecoders(
            spatial_dims=spatial_dims,
            out_channels=out_channels,
            features=features,
            actv=actv,
            norm=norm,
            bias=bias,
            dropout=dropout,
            up_sampling=up_sampling)
        self.add_module("decoders", self.decoders)

    def encode(self, x: torch.Tensor)-> Tuple[torch.Tensor, List[torch.Tensor]]:
        return self.encoders(x)

    def decode(self, x: torch.Tensor, skips: List[torch.Tensor])-> torch.Tensor:
        return self.decoders(x, skips)

    def forward(self, x: torch.Tensor):
        """
        x: input should have spatially N dimensions
            ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N])``, N is defined by `dimensions`.
            It is recommended to have ``dim_n % 16 == 0`` to ensure all maxpooling inputs have
            even edge lengths.
        """
        orig = x 

        x, skips = self.encode(x)
        x = self.bottleneck(x)
        x = self.decode(x, skips)

        denoised = orig - x
        return denoised

class FiLMed3DUNet(nn.Module):
    def __init__(
            self,
            metadata_size:int,
            spatial_dims: int = 3,
            # TODO: integrate image volume gradients on axis
            in_channels: int = 1,
            out_channels: int = 1,
            features: Features = DEFAULT_FEATURES,
            actv: Activation = Activation.PRELU,
            norm: Normalization = Normalization.INSTANCE,
            bias: bool = True,
            dropout: float = 0.0,
            up_sampling: UpSampling = UpSampling.DECONV,
            dimensions: Optional[int] = None,
            amp: bool = True,
            **kwargs
            ):

        super().__init__()

        self.spatial_dims = spatial_dims

        self.metadata_net = MetadataNetwork(metadata_size, features.encoder)

        self.filmed_layers = Unet3DEncoders(
            spatial_dims=spatial_dims, 
            in_channels=in_channels, 
            features=features, 
            actv=actv,
            norm=norm,
            bias=bias,
            dropout=dropout,
            pre_init_f_inplace=inject_film_wrapper_to,
            )
        self.add_module("filmed_layers", self.filmed_layers)

        bottleneck_chan = features.encoder[-1]
        self.bottleneck = NConv(spatial_dims, bottleneck_chan, bottleneck_chan, actv, norm, bias, dropout=dropout)
        self.add_module("bottleneck", self.bottleneck)

        self.decoders = Unet3DDecoders(
            spatial_dims=spatial_dims,
            out_channels=out_channels,
            features=features,
            actv=actv,
            norm=norm,
            bias=bias,
            dropout=dropout,
            up_sampling=up_sampling)
        self.add_module("decoders", self.decoders)

    def encode(self, x: torch.Tensor, meta:torch.Tensor)-> Tuple[torch.Tensor, List[torch.Tensor]]:
        gammas_betas = self.metadata_net(meta)
        for ix, filmed in enumerate(self.filmed_layers.encoders):
            (gamma, beta) = gammas_betas[ix]
            filmed.inject(gamma, beta)
        return self.filmed_layers(x)

    def decode(self, x: torch.Tensor, skips: List[torch.Tensor])-> torch.Tensor:
        return self.decoders(x, skips)

    def forward(self, x:torch.Tensor, meta:torch.Tensor):
        """
        meta: metadata tensor of shape (batch_size, metadata_size)
        """
        divisible, fail_indices = is_spatial_dims_divisible_by(x, self.spatial_dims, 2 ** self.spatial_dims)
        # TODO: handle this better
        if not divisible:
            m = f"Input tensor have dimensions that are not divisible by {2**self.spatial_dims}:{fail_indices}"
            raise ValueError(m)
            # logger.warn(m)

        _ = torch.initial_seed() % (2**32 -1)

        orig = x

        x, skips = self.encode(x, meta)
        x = self.bottleneck(x)
        x = self.decode(x, skips)

        denoised = orig - x
        return denoised


def is_spatial_dims_divisible_by(x: torch.Tensor, spatial_dims_to_check: int, by=16, panic=False) -> Tuple[bool, List]:
    x_shape = x.shape

    if len(x_shape) < spatial_dims_to_check:
        raise ValueError(f"Input tensor must have at least {spatial_dims_to_check} dimensions, but got shape {x_shape}")

    spatial_sizes = x_shape[-spatial_dims_to_check:]

    indivisible_dim_indices = []
    for i, size in enumerate(spatial_sizes):
        if size % by != 0:
            actual_dim_index = len(x_shape) - spatial_dims_to_check + i
            indivisible_dim_indices.append(actual_dim_index)

    if panic and indivisible_dim_indices:
        raise ValueError(f"Input tensor have dimensions that are not divisible by {by}:{indivisible_dim_indices}")

    return (not indivisible_dim_indices,  indivisible_dim_indices)


def print_unet(network:FiLMed3DUNet, to_print=['encoders', 'bottleneck', 'decoders'], dont_show=['filmed']):
    for k, m in network._modules.items():
        if k not in to_print: continue
        if not isinstance(m, nn.ModuleList): continue
        for mm in m:
            for n,sm in mm._modules.items():
                if n in dont_show: continue
                for _n, ssm in sm._modules.items():
                    if _n in dont_show: continue
                    print(ssm)