import unittest 
import torch 
from src.layer import DoubleConv, Down, Up, OutConv

class TestLayers(unittest.TestCase):
    def test_double_conv(self):
        obj = DoubleConv(in_channels=3, out_channels=1)
        img = torch.rand(1,3,28,28)
        self.assertEqual(obj.forward(img).shape, (1,1,28,28))

    def test_down_layer(self):
        pool_size = 4
        size_in = 28
        pool_out_size = size_in // pool_size
        obj = Down(in_channels=2, out_channels=1, pool=pool_size)
        img = torch.rand(1,2,size_in,size_in)
        self.assertEqual(obj.forward(img).shape, (1,1,pool_out_size,pool_out_size))
        
        pool_size = 5
        size_in = 28
        pool_out_size = size_in // pool_size 
        obj = Down(in_channels=2, out_channels=1, pool=pool_size)
        img = torch.rand(1,2,size_in,size_in)
        self.assertEqual(obj.forward(img).shape, (1,1,pool_out_size,pool_out_size))
        
    def test_up_layer(self):
        
        size_in = 280
        size_2 = 56
        channel_1 = 2
        img = torch.rand(1,channel_1,size_in +2 ,size_in)
        channel_2 = 2
        img2 = torch.rand(1,channel_2,size_2,size_2 + 2)
        obj = Up(in_channels=channel_1 + channel_2, out_channels=2, bilinear=True)
        # input size : (batch_size, channel_1, size_1,size_1) x (batch_size, channel_2, size_2,size_2) --> (batch_size, channel_1 + channel_2, size_2,size_2)
        self.assertEqual(obj.forward(img, img2).shape, torch.Size([1,2,56,58]))

        channel_1 = 4
        channel_2 = 2
        obj = Up(in_channels=channel_1, out_channels=channel_1 - channel_2, bilinear=False)
        size_in = 280
        size_2 = 56
        
        img = torch.rand(1,channel_1,size_in +2 ,size_in)
        img2 = torch.rand(1,channel_2,size_2,size_2 + 2)
        # input size : (batch_size, channel_1, size_1,size_1) x (batch_size, channel_2, size_2,size_2) --> (batch_size, channel_1 - channel_2, size_2,size_2)
        self.assertEqual(obj.forward(img, img2).shape, torch.Size([1,channel_1-channel_2,size_2, size_2+2]))

    def test_out_conv(self):
        channel_1 = 4
        size_in = 280
        img = torch.rand(1,channel_1,size_in +2 ,size_in)

        obj = OutConv(in_channels=channel_1, out_channels=2)
        self.assertEqual(obj.forward(img).shape, torch.Size([1,2,size_in +2 , size_in]))


        


if __name__ == "__main__":
    unittest.main()