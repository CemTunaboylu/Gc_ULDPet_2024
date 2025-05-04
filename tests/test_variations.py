import unittest

import torch

from variations import generate_conv_variations

class TestDepthWiseConv(unittest.TestCase):
    def test_generate_conv_variations_basic(self):
        test_case = {
            "input_size": (8, 8),
            "output_size": (8, 8),
            "kernel_sizes": [3, 5, 7],
            "strides": [1, 2],
            "dilations": [1, 2],
            "groups": [1, 2, 3, 5, 8],
        }
        configurations = generate_conv_variations(**test_case)
        self.assertIsInstance(configurations, list)
        self.assertEqual(len(configurations), 2)
        for dim_cfgs in configurations:
            self.assertIsInstance(dim_cfgs, list)
            for cfg in dim_cfgs:
                self.assertIn('kernel_size', cfg)
                self.assertIn('stride', cfg)
                self.assertIn('dilation', cfg)
                self.assertIn('padding', cfg)
                self.assertIsInstance(cfg['padding'], tuple)
                self.assertEqual(len(cfg['padding']), 2)
                self.assertIn(cfg['kernel_size'], test_case['kernel_sizes'])
                self.assertIn(cfg['stride'], test_case['strides'])
                self.assertIn(cfg['dilation'], test_case['dilations'])
                for p in cfg['padding']:
                    self.assertGreaterEqual(p, 0)
                # Check output size is as expected
                in_dim = test_case["input_size"][0]
                out_dim = test_case["output_size"][0]
                k = cfg['kernel_size']
                s = cfg['stride']
                d = cfg['dilation']
                p = cfg['padding'][0]
                calc_out = ((in_dim + 2 * p - d * (k - 1) - 1) // s) + 1
                self.assertEqual(calc_out, out_dim)

    def test_3d_input(self):
        test_case = {
            "input_size": (8, 8, 8),
            "output_size": (8, 8, 8),
            "kernel_sizes": [3,5,7],
            "strides": [1,2],
            "dilations": [1,2],
            "groups": [1,2],
        }
        configurations = generate_conv_variations(**test_case)
        self.assertEqual(len(configurations), 3)
        for dim_cfgs in configurations:
            self.assertTrue(all('kernel_size' in cfg for cfg in dim_cfgs))
            for cfg in dim_cfgs:
                in_dim = test_case["input_size"][0]
                out_dim = test_case["output_size"][0]
                k = cfg['kernel_size']
                s = cfg['stride']
                d = cfg['dilation']
                p = cfg['padding'][0]
                calc_out = ((in_dim + 2 * p - d * (k - 1) - 1) // s) + 1
                self.assertEqual(calc_out, out_dim)

    def test_same_output_with_different_configs(self):
        # Test that different kernel/stride/dilation/padding combos can yield same output
        test_cases = [
            # (input_size, output_size, kernel_size, stride, dilation)
            ((16, 16), (16, 16), 3, 1, 1),
            ((16, 16), (8, 8), 3, 2, 1),
            ((16, 16), (16, 16), 5, 1, 2),
            ((32, 32), (16, 16), 3, 2, 1),
        ]
        for in_size, out_size, k, s, d in test_cases:
            test_case = {
                "input_size": in_size,
                "output_size": out_size,
                "kernel_sizes": [k],
                "strides": [s],
                "dilations": [d],
                "groups": [1],
            }
            configurations = generate_conv_variations(**test_case)
            for dim_cfgs in configurations:
                for cfg in dim_cfgs:
                    in_dim = in_size[0]
                    out_dim = out_size[0]
                    k = cfg['kernel_size']
                    s = cfg['stride']
                    d = cfg['dilation']
                    p = cfg['padding'][0]
                    calc_out = ((in_dim + 2 * p - d * (k - 1) - 1) // s) + 1
                    self.assertEqual(calc_out, out_dim)


    def test_generate_conv_variations_real(self):
        test_case = {
            "input_size": (360, 256,256),
            "output_size": (180, 128,128),
            "kernel_sizes": [3, 5, 7, 9],
            "strides": [1, 2, 3],
            "dilations": [1, 2, 3],
            "groups": [5, 8, 11, 15, 20],
        }
        configurations = generate_conv_variations(**test_case)
        self.assertIsInstance(configurations, list)
        self.assertEqual(len(configurations), 3)
        for dim_cfgs in configurations:
            self.assertIsInstance(dim_cfgs, list)
            for cfg in dim_cfgs:
                self.assertIn('kernel_size', cfg)
                self.assertIn('stride', cfg)
                self.assertIn('dilation', cfg)
                self.assertIn('padding', cfg)
                self.assertIsInstance(cfg['padding'], tuple)
                self.assertEqual(len(cfg['padding']), 2)
                self.assertIn(cfg['kernel_size'], test_case['kernel_sizes'])
                self.assertIn(cfg['stride'], test_case['strides'])
                self.assertIn(cfg['dilation'], test_case['dilations'])
                for p in cfg['padding']:
                    self.assertGreaterEqual(p, 0)
                # Check output size is as expected
                in_dim = test_case["input_size"][0]
                out_dim = test_case["output_size"][0]
                k = cfg['kernel_size']
                s = cfg['stride']
                d = cfg['dilation']
                p = cfg['padding'][0]
                calc_out = ((in_dim + 2 * p - d * (k - 1) - 1) // s) + 1
                self.assertEqual(calc_out, out_dim)

    def test_invalid_padding(self):
        test_case = {
            "input_size": (8, 8),
            "output_size": (8, 8),
            "kernel_sizes": [9],
            "strides": [1],
            "dilations": [1],
            "groups": [1],
        }
        configurations = generate_conv_variations(**test_case)
        self.assertTrue(all(len(cfgs) == 0 for cfgs in configurations))


    def test_invalid_dimension(self):
        test_case = {
            "input_size": (8,),
            "output_size": (8,),
            "kernel_sizes": [3],
            "strides": [1],
            "dilations": [1],
            "groups": [1],
        }
        with self.assertRaises(ValueError):
            generate_conv_variations(**test_case)

    def test_padding_limit(self):
        # Should not return configs with padding > padding_limit
        test_case = {
            "input_size": (8, 8),
            "output_size": (8, 8),
            "kernel_sizes": [7],
            "strides": [1],
            "dilations": [2],
            "groups": [1],
            "padding_limit": 1,
        }
        configurations = generate_conv_variations(**test_case)
        for dim_cfgs in configurations:
            for cfg in dim_cfgs:
                for p in cfg['padding']:
                    self.assertLessEqual(p, 1)

if __name__ == "__main__":
    unittest.main()