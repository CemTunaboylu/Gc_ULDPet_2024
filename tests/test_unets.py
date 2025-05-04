from functools import partial
import unittest

import torch

from unets import Features, FiLMed3DUNet,UNet3D, print_unet

def various_sized_unets(cls=FiLMed3DUNet, args=(12,3,21,21)):
    fs = [
        Features(encoder=[64, 128, 256, 512],final=32),            # params: 35_539_024
        Features(encoder=[32, 64, 128, 256, 512],final=32),        # params: 35_788_436
        Features(encoder=[32, 64, 128, 256],final=32),             # params: 9_019_024
        Features(encoder=[16, 32, 64, 128, 256, 512],final=32),    # params: 35_850_936
        Features(encoder=[16, 32, 64, 128, 256],final=32),         # params: 9_081_524
    ]
    for f in fs:
        u = cls(*args, f)
        # p = count_parameters(u)
        # many_gbs(p)

class TestUnets(unittest.TestCase):
    # def test_different_sized_unets(self):
        # various_sized_unets(UNet3D, (3,21,21))
        # various_sized_unets()

    def test_forward_pass(self):
        # encoders = [64, 128, 256, 512]
        encoders = [16,32]
        features = Features(encoder=encoders)
        batch = 2
        meta_size = 21
        channel_size = 6 
        # h = w = 440
        h = w = 32
        # d = 100
        d = 32 
        model_classes = [partial(FiLMed3DUNet, metadata_size = meta_size), UNet3D]
        for cls in model_classes: 
            model = cls(
                    spatial_dims = 3, 
                    in_channels = channel_size, 
                    out_channels = channel_size, 
                    features = features
                    )

            model.train()
            i= torch.ones([batch,channel_size,d,h,w])
            o : torch.Tensor
            if cls == UNet3D:
                o = model.forward(i)
            else:
                meta = torch.ones([batch,meta_size])
                o = model.forward(i,meta)
            self.assertEqual(o.shape, (batch, channel_size, d, h, w))
        


if __name__ == "__main__":
    unittest.main()