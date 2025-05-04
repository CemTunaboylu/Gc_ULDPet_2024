from enum import Enum
from typing import Optional, Tuple, Union

import torch 
from torch import nn

from adn import Activation, Normalization
from layers import NConv

from MONAI.monai.networks.blocks import UpSample

class UpSampling(Enum):
    # NEAREST_NEIGHBOR, BILINEAR
    DECONV = "deconv"
    PIXEL_SHUFFLE = "pixelshuffle"
    NON_TRAINABLE = "nontrainable"

    def __str__(self):
        return self.value

class ConcatenatingUpScaler(nn.Sequential):
    def __init__(
            self,
            spatial_dims: int,
            in_chan: int,
            cat_chan: int,
            out_chan: int,
            actv: Activation,
            norm: Normalization = Normalization.INSTANCE,
            bias: bool = True,
            dropout: float = 0.0,
            up_sampling: UpSampling = UpSampling.DECONV,
            # TODO: make this better
            # pre_conv: a conv block applied before upsampling. - only used in the "nontrainable" or "pixelshuffle" mode.
            pre_conv: Optional[Union[nn.Module, str]] = "default",
            # TODO: make this better
            # interp_mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``} - only used in the "nontrainable" mode.
            interp_mode:str = "linear",
            # Only used in the "nontrainable" mode.
            align_corners: Optional[bool] = True,
            # halves: whether to halve the number of channels during upsampling. 
            # this parameter does not work on ``nontrainable`` mode if ``pre_conv`` is `None`.
            halves:bool = True,
        ):

        super().__init__()
        self.spatial_dims = spatial_dims

        up_chan : int 
        if UpSampling.NON_TRAINABLE.value == up_sampling.value and pre_conv is None:
            up_chan = in_chan
        elif halves:
            up_chan = in_chan//2
        else:
            up_chan = in_chan 

        self.upsample = UpSample(
            spatial_dims,
            in_chan,
            up_chan,
            scale_factor=2 if halves else 1,
            kernel_size=3,
            mode=up_sampling.value,
            pre_conv = pre_conv,
            interp_mode=interp_mode,
            align_corners=align_corners
            )

        self.convs = NConv(spatial_dims, cat_chan + up_chan, out_chan, actv, norm, bias, dropout=dropout)

    def forward(self, x: torch.Tensor, skip:Union[torch.Tensor, None] = None):
        x = self.upsample(x)
        if skip is not None:
            x = concat_with_size_check(skip, x, dim=1, fix_if_dim_mismatch=True, fix_by_padding=True)
        x = self.convs(x)
        return x

# FIXME: handle errors
def concat_with_size_check(x1, x2, dim=1, fix_if_dim_mismatch=True, fix_by_padding=True)->Union[torch.Tensor, None]:
    same_dimensions = len(x1.shape) == len(x2.shape)
    if not same_dimensions:
        print(f"Dimensions do not match: {len(x1.shape)} vs {len(x2.shape)}")
        return None

    batch_sizes_match = x1.shape[0] == x2.shape[0]
    if not batch_sizes_match:
        print(f"Batch sizes do not match: {x1.shape[0]} vs {x2.shape[0]}")
        return None

    spatial_dims_match = x1.shape[2:] == x2.shape[2:]
    if spatial_dims_match:
        return torch.cat([x1, x2], dim=dim)

    print(f"Spatial dimensions do not match: {x1.shape[2:]} vs {x2.shape[2:]}")
    if not fix_if_dim_mismatch:
        return None
    if fix_by_padding:
        (x1,x2) = pad_to_larger(x1 ,x2)
    else:
        raise NotImplementedError(f"fixing with cropping is not implemented")
    return torch.cat([x1, x2], dim=dim)


def pad_to_larger(x1:torch.Tensor, x2:torch.Tensor, mode='replicate')->Tuple[torch.Tensor, torch.Tensor]:
    dims = len(x1.shape) - 2 # 2x max pooling with odd edge length handling
    D = dims * 2
    sp_1, sp_2 = [0] * D, [0] * D # padding pairs as p_<ith from last dim>_<axis>
    # ex: pad last and before last dims [p_1_x pp_1_y p_2_x p_2_y]

    r_shape = lambda t: reversed(t.shape[-dims:])
    for i,(x_1, x_2) in enumerate(zip(r_shape(x1), r_shape(x2))):
        if x_1 == x_2: continue
        s = sp_2 if x_1 > x_2 else sp_1
        factor = abs(x_1 - x_2)
        if factor > 2: raise ValueError(f'padding implementation only handles halving, but received a difference: {factor}')
        # difference in dims dictates in which dims the padding will be performed
        for ax in range(factor-1, -1, -1):
            s[i*2+ax] = 1

    _pad = lambda t, s: nn.functional.pad(t, tuple(s), mode)
    x1, x2 = _pad(x1, sp_1), _pad(x2, sp_2)
    return (x1,x2)
