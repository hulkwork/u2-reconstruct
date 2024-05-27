import unittest

import torch

from src.model import UNet


class TestLayers(unittest.TestCase):
    def test_double_conv(self):
        obj = UNet(n_channels=3, n_classes=2, bilinear=True)
        img = torch.rand(5, 3, 28, 28)
        self.assertEqual(obj.forward(img).shape, (5, 2, 28, 28))


if __name__ == "__main__":
    unittest.main()
