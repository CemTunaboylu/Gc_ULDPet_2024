import unittest

import torch 

from depth_wise import DepthwiseSeparableConv, InterleavedDepthwiseConv


batch_size = 4
in_channels = 8
out_channels = 11
depth = 16
height = width = 32

# class TestDepthWiseConv2D(unittest.TestCase):
    # def test_interleaved_depthwise_conv(self):
    #     conv_module = InterleavedDepthwiseConv(
    #         spatial_dims=3,
    #         in_chs=in_channels,
    #         out_chs=out_channels,
    #         kernel_size=3,
    #         stride=1,
    #         padding=1,
    #         group_size=4,
    #         overlap_ratio=1.5,
    #         expected_depth_dim=depth,
    #     )
    #     x = torch.randn(batch_size, in_channels, depth, height, width)
    #     # Test the forward pass
    #     output = conv_module(x)
    #     self.assertEqual(output.shape, (batch_size, out_channels, depth, height, width))

    # def setUp(self):
        # Set up a simple test case
        # self.input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
        # self.conv = DepthWiseConv2D(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1)

    # def test_forward(self):
        # Test the forward pass
        # output = self.conv(self.input_tensor)
        # self.assertEqual(output.shape, (1, 6, 32, 32))  # Check output shape

    # def test_weight_initialization(self):
        # Test weight initialization
        # weight = self.conv.weight
        # self.assertEqual(weight.shape, (6, 1, 3, 3))  # Check weight shape


# if __name__ == "__main__":
#     unittest.main()