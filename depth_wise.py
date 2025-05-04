import torch
from torch import nn
from torch.nn import functional as F

import einops

from layers import get_conv_for_dim, get_conv_for_dim_from


def channel_shuffle(x, groups):
    batch_size, num_chs, *spatial_dims = x.size()
    assert num_chs % groups == 0, f"Number of channels must be divisible by groups, got ch:{num_chs}, g:{groups}"

    x = einops.rearrange(x, 'b (g c) ... -> b (c g) ...', g=groups).contiguous()
    return x

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, spatial_dims, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()

        self.spatial_dims = spatial_dims
        conv_d = get_conv_for_dim(spatial_dims)

        self.depthwise = conv_d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=in_channels, bias=False
        )
        self.pointwise = conv_d(
            in_channels, out_channels, kernel_size=1, bias=False
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# TODO: handle logger
def prepare_overlapping_slices(depth:int, group:int, denom:int, adjust=True):
    if depth % group != 0:
        if not adjust: raise ValueError(
            f"depth:{depth} is not divisible by group:{group}, this will cause uncovered depth for last slice without adjusting the last slice"
        )
        # logger.warn(f"depth:{depth} is not divisible by group:{group}, if the last slice leaves uncovered depth, it will be elongated to cover the remaining.")

    l_0= depth // group
    # the following must hold for the new length l_1 : ((denom-1).group+1).l_1 = group.denom.l_0
    l_1 = int(group * denom *l_0//((denom-1)*group+1))
    o_1_len = int((l_1)//denom)
    g_ixs = [slice(0, l_1)]
    step_size = l_1 - o_1_len
    for i in range(1,group):
        prev = g_ixs[-1]
        s = prev.stop - o_1_len
        g_ixs.append( slice(s, min(s + l_1, depth)) )
    # TODO: maybe add an additional group if the uncovered is not small if its 27 and the length is 40, it may be better to add another group
    if adjust and g_ixs[-1].stop < depth:
        uncovered = depth - g_ixs[-1].stop
        g_ixs[-1] = slice(g_ixs[-1].start, depth)
        # logger.warn(f'    adjusted last slice to {[(s.start, s.stop, s.stop-s.start) for s in g_ixs[-1:]]} to cover last {uncovered}')
    return g_ixs, step_size


# Batch Processing: Ensure that custom layers support batch processing effectively.
class InterleavedDepthwiseConv(nn.Module):
    def __init__(self, spatial_dims, in_chs, out_chs, kernel_size, stride=1, padding=0, group_size=3, overlap_ratio=2, expected_depth_dim:int|None=None):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_chs = in_chs
        self.out_chs = out_chs

        self.kernel_size = kernel_size

        conv_d = get_conv_for_dim_from(spatial_dims, nn)
        self.vectorized_conv_d = get_conv_for_dim_from(spatial_dims, F)
        assert overlap_ratio > 1, f'overlap ratio as the denominator should be larger than 1, got {overlap_ratio}'

        # Calculate the size of each group
        self.group_size = group_size
        self.overlap_ratio = overlap_ratio
        # Define the indices for each group
        if in_chs < group_size and (not expected_depth_dim or expected_depth_dim < group_size) :
            raise ValueError(f'either input channels or depth to receive must be larger than group_size {group_size}, got input channels: {in_chs}, depth: {expected_depth_dim}')

        # TOOD: use enum
        depth = in_chs if in_chs >= group_size else expected_depth_dim
        self.working_on_depth = expected_depth_dim == depth

        if self.working_on_depth:
            self.slice_tensor = lambda t, slicing: t[:, :, slicing :, :, :] # slice the depth
            # TODO: check these out
            self.overlap_window_str = 'b c d ...  -> b c ... d'
            self.conv_prep_str = 'b ... num_groups group_size -> (b num_groups) ... group_size'
            self.reshape_out_back_str = '(b num_groups) c ... -> b (num_groups c) ...'
            self.rearrange_lastly = 'b c ... d -> b c d ... '
        else:
            self.slice_tensor = lambda t,slicing: t[:, slicing, :, :, :] # slice the channels
            self.overlap_window_str = 'b c ... -> b ... c' # dim agnostic
            self.conv_prep_str = 'b ... num_groups group_size -> (b num_groups) group_size ...'
            self.reshape_out_back_str = '(b num_groups) c ... -> b (num_groups c) ...'


        self.group_indices, self.step_size = prepare_overlapping_slices(depth, group_size, overlap_ratio, adjust=True)
        self.num_groups = len(self.group_indices)

        # TODO: this overlaps by vectorizing input?
        # shape: (total_out_channels, group_size, kernel_size, kernel_size, [kernel_size])
        self.vectorized_multiple_convolutions_shape = (out_chs, group_size, *([self.kernel_size] * self.spatial_dims))
        # stacked kernels for all groups
        self.vectorized_multiple_convolutions = nn.Parameter(
            torch.randn(self.vectorized_multiple_convolutions_shape, requires_grad=True)
        )
        # self.conv = conv_d(
        #     # in_channels = group_size * self.num_groups,
        #     in_channels = self.group_size,
        #     out_channels = (out_chs // self.num_groups),
        #     kernel_size=kernel_size,
        #     stride=stride,
        #     padding=padding,
        #     groups = self.num_groups,
        # )
        # Define convolutional layers for each group
        # Currently only supports depth dim
        self.total_output_channels = 0
        self.convs = nn.ModuleList()
        for ix, slc in enumerate(self.group_indices):
            group_out_channels = slc.stop - slc.start
            self.total_output_channels += group_out_channels
            c = conv_d(
            in_channels = group_out_channels,
            out_channels= group_out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
            )
            self.convs.append(c)
            # .add_module(f'conv_{ix}', c)
            # self.convs.add_module(f'conv_{ix}', c)

        # TODO: if depth == groups: point-wise conv, thus should not be allowed

    def vectorize_overlapping_groups(self, x):
        x = einops.rearrange(x, self.overlap_window_str)

        x = x.unfold(dimension=-1, size=self.group_size, step=self.step_size)
        # Shape: (b, d, h, w, num_groups, group_size) if channelwise
        # Shape: (b, c, h, w, num_groups, group_size) if depth dim wise

        x = einops.rearrange(x, self.conv_prep_str)
        def rearrange_back(t):
            t = einops.rearrange(t, self.reshape_out_back_str)
            if self.rearrange_lastly:
                t = einops.rearrange(t, self.rearrange_lastly)
            return t
        return x, rearrange_back

    def vectorized_forward(self, x):
        batch_size, in_channels, D, H, W = x.shape
        x, f_rearrange_back = self.vectorize_overlapping_groups(x)
        x = self.vectorized_conv_d(
            x,
            self.vectorized_multiple_convolutions,
            bias = self.bias,
            stride = self.stride,
            padding = self.padding,
            groups = self.num_groups,
            )
        x = f_rearrange_back(x)
        return x

    def forward(self, x):
        batch_size, in_channels, D, H, W = x.shape
        assert in_channels == self.in_chs, f'input channels must be of length {self.in_chs}, got {in_channels}'

        # Consider vectorizing operations where possible.
        # Extract overlapping groups
        concatanated_output = torch.empty(
            batch_size,
            self.total_output_channels,
            D_out,
            H_out,
            W_out,
            device=x.device,
            dtype=x.dtype
        )
        ch_offset = 0
        for conv, slc in zip(self.convs, self.group_indices):
            # Process group
            sliced_x = self.slice_tensor(x, slc)
            group_output = conv(sliced_x)
            channels = group_output.shape[1]
            concatanated_output[:, ch_offset:ch_offset+channels, ...] = group_output
            ch_offset += channels

            del sliced_x, group_output
        # for ix, slicing in enumerate(self.group_indices):
            # slice_tensor supports batching?
            # outputs[ix] = self.convs[ix](sliced_x)

        # Concatenate outputs along the channel dimension
        # output = torch.cat(outputs, dim=1)
        # if shuffle: output = channel_shuffle(output, self.group_size)
        return concatanated_output
