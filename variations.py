from enum import Enum
from itertools import product
from typing import List, Dict, Tuple, Union

class Grouping(Enum):
    SPATIAL = False
    CHANNEL_WISE = True

"""
    Equation used:
        output = ((input + 2*Padding - dilation*(kernel_size-1) - 1) // stride) + 1
        Rearranged to compute padding:
        Padding = ((stride * (output- 1) - input + dilation*(kernel_size - 1) + 1) / 2)

        note: arranged formula loses the flooring with '// 2', thus we need to ensure whether we are 
        mistaken or not by plugging the values back into the original equation

"""
def calculate_padding_for(in_dim:int, out_dim:int, kernel_size:int, stride:int=1, dilation:int=1) -> int:
    padding = int((stride*(out_dim - 1) - in_dim + dilation * (kernel_size - 1) + 1) / 2)
    if padding < 0:
        raise ValueError(f'Padding cannot be negative, got {padding}')
    original = lambda P : ((input + 2*P- dilation*(kernel_size-1) - 1) // stride) + 1

    match original(padding):
        case o if o == out_dim:  
            return padding
        case o if o < out_dim and original(padding - 1) == out_dim:
            return padding - 1
        case _ if original(padding + 1) == out_dim:
            return padding + 1
            
    raise ValueError(f'Cannot find a padding for given parameters')

"""
Visual Comparison of Two Convolution Configurations
---------------------------------------------------

Both configurations produce the same output shape from a (16x16) input,
but they cover the input differently :

Config A: kernel=3, dilation=1, padding=1 (regular convolution)
Receptive Field (3x3, contiguous)

Sliding kernel pattern:

    █ █ █
    █ █ █   ← local, dense coverage
    █ █ █

Effective Receptive Field = 3 x 3 = 9

Config B: kernel=5, dilation=2, padding=4 (dilated convolution)
Receptive Field (5x5, dilated)


   Dilation=2 means there are 1-pixel gaps between sampled points.
   Kernel covers a much larger area, with fewer values used.

    █ . █ . █
    . . . . .
    █ . █ . █   ← sparse sampling with wide coverage
    . . . . .
    █ . █ . █

Effective Receptive Field = 1 + d·(k−1) = 1 + 2·(5−1) = 9

Even though the output size is the same, the effective receptive field is much larger in the second case.
"""

def generate_conv_variations(
    input_size: Union[Tuple[int, int], Tuple[int, int, int]],
    output_size: Union[Tuple[int, int], Tuple[int, int, int]],
    kernel_sizes: List[int], # assuming that kernel has same dims for now
    strides: List[int],
    dilations: List[int],
    groups: List[int],
    grouping_type: Grouping = Grouping.CHANNEL_WISE,
    padding_limit: int = 3,
    depthwise_options: bool|List[bool] = False,
) -> List[List[Dict]]:
    """
    Generate all valid convolutional layer configurations based on the provided constraints.

    Parameters:
    - input_size & output_size: (height, width) for 2D or (depth, height, width) for 3D.
    - kernel_sizes: List of possible kernel sizes. Can be integers or tuples.
            If integer and dim > 1, it is assumed to be the same across all spatial dimensions.
    - strides: List of possible strides.
    - dilations: List of possible dilations.
    - groups: List of possible groups. Groups can be either on channels or on spatial dimensions.
    - padding_limit: int imposes a max value limit on padding, default is 3
    - depthwise_options: either a bool or a list of bools to control which options use depthwise convolution

    assumes symmetric padding
    Returns:
    - List of dictionaries containing convolution parameters as keyword arguments for partial
    """
    assert len(input_size) == len(output_size), f'Input and output sizes must match, got i:{len(input_size)}, o:{len(output_size)}'
    dims = len(input_size)
    if dims not in [2, 3]: raise ValueError("Dimension must be either 2 or 3.")

    variations_by_dims = [[] for _ in range(dims)]
    for (s, k, d) in product(strides, kernel_sizes, dilations):
        for x in range(dims):
            try:
                calc_p = calculate_padding_for(input_size[x], output_size[x], k, s, d)
                if calc_p > padding_limit: continue

                # only this dim has non-zero padding
                paddings = [0] * dims
                paddings[x] = calc_p

                variations_by_dims[x].append({
                    'kernel_size' : k,
                    'stride' : s,
                    'dilation' : d,
                    'padding' : tuple(paddings)}
                )
            except:
                continue
    return variations_by_dims

def is_of_same_type(*values)->bool: return all([type(values[0]) == type(v)for v in values[1:]])

def merge_variations(dim_ordered_list_of_dicts:List[Dict])->Dict:
    args = dict()
    for k in dim_ordered_list_of_dicts[0].keys():
        values = [ d[k] for d in dim_ordered_list_of_dicts]
        if not is_of_same_type(*values): raise TypeError(f'{values} are not of same type')
        match values[0]:
            case str() | int() | float():
                args[k] = values
            case list() | tuple():
                args[k] = merge_by_dim(*values)
            case _:
                print(f'unexptected type {type(values[0])} -> {k}:{values}')
                args[k] = values
    return args


def merge_by_dim(*iterables)->List:
    merged = [0] * len(iterables)
    for dim, iter in enumerate(iterables):
        merged[dim] = iter[dim]

    return merged


def print_conv_variations(configs: List[List[Dict]]):
    for dim_index, dim_configs in enumerate(configs):
        print(f"\n[Dimension {dim_index}] — {len(dim_configs)} configurations found")
        for ix, cfg in enumerate(dim_configs):
            print(f"  #{ix+1:02d}: "
                  f"kernel={cfg['kernel_size']}, "
                  f"stride={cfg['stride']}, "
                  f"dilation={cfg['dilation']}, "
                  f"padding={cfg['padding']}")
