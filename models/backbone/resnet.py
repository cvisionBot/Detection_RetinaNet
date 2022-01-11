from ..layers.convolution import Conv2dBnAct, Conv2dBn
from ..layers.blocks import Residual_LiteBlock
from ..initialize import weight_initialize

import torch
from torch import nn

class ModelStem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ModelStem, self).__init__()
        self.conv = Conv2dBnAct(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=2)
        
    def forward(self, input):
        output = self.conv(input)
        return output
    
class Make_LiteLayer(nn.Module):
    def __init__(self, layers_configs, pre_layer_ch):
        super(Make_LiteLayer, self).__init__()
        self.stage_channels = []
        self.pre_ch = pre_layer_ch
        self.layers_configs = layers_configs
        self.layer = self.residual_litelayer(self.layers_configs)
        
    def forward(self, input):
        return self.layer(input)
    
    def residual_litelayer(self, cfg):
        layers = []
        input_ch = cfg[0]
        for i in range(cfg[-1]):
            if i == 0:
                layer = Residual_LiteBlock(in_channels=self.pre_ch, kernel_size=cfg[1], out_channels=cfg[2], stride=cfg[3])
            else:
                layer = Residual_LiteBlock(in_channels=input_ch, kernel_size=cfg[1], out_channels=cfg[2])
            layers.append(layer)
            input_ch = layer.get_channel()
        return nn.Sequential(*layers)
    
class _ResNet18(nn.Module):
    def __init__(self, in_channels, classes):
        super(_ResNet18, self).__init__()
        self.Stem = ModelStem(in_channels=in_channels, out_channels=64)
        self.stage_channels = [64, 64, 128, 256, 512]
        # configs : in_channels, kernel_size, out_channels, stride, iter_cnt
        conv2_x = [64, 3, 64, 2, 2]
        conv3_x = [128, 3, 128, 2, 2]
        conv4_x = [256, 3, 256, 2, 2]
        conv5_x = [512, 3, 512, 2, 2]
        
        self.layer1 = Make_LiteLayer(conv2_x, 64)
        self.layer2 = Make_LiteLayer(conv3_x, 64)
        self.layer3 = Make_LiteLayer(conv4_x, 128)
        self.layer4 = Make_LiteLayer(conv5_x, 256)
        self.classification = nn.Sequential(
            Conv2dBnAct(in_channels=512, out_channels=1280, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1280, classes, 1)
        )
        
    def forward(self, input):
        stem= self.Stem(input)
        print('stem shape : ', stem.shape)
        s1 = self.layer1(stem)
        print('stage 1 shape : ', s1.shape)
        s2 = self.layer2(s1)
        print('stage 2 shape : ', s2.shape)
        s3 = self.layer3(s2)
        print('stage 3 shape : ', s3.shape)
        s4 = self.layer4(s3)
        print('stage 4 shape : ', s4.shape)
        pred = self.classification(s4)
        b, c, _, _ = pred.size()
        pred = pred.view(b, c)
        stages = [stem, s1, s2, s3, s4]
        return {'stages' : stages, 'pred':pred}
    
    
def ResNet(in_channels, classes=1000):
    model = _ResNet18(in_channels=in_channels, classes=classes)
    weight_initialize(model)
    return model


if __name__ == '__main__':
    model = ResNet(in_channels=3, classes=1000)
    model(torch.rand(1, 3, 320, 320))