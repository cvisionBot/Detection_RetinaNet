from ..layers.convolution import Conv2dBnAct, Conv2dBn

import torch
from torch import nn

class Residual_LiteBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, out_channels, stride=1):
        super(Residual_LiteBlock, self).__init__()
        self.out_channels = out_channels
        self.res_conv1 = Conv2dBnAct(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=1)
        self.res_conv2 = Conv2dBn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        self.identity = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)
        
    def forward(self, input):
        output = self.res_conv1(input)
        output = self.res_conv2(output)
        if input.size() != output.size():
            input = self.identity(input)
        output = input + output
        return output

    def get_channel(self):
        return self.out_channels