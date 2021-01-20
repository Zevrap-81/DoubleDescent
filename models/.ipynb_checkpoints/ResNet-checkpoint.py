import torch.nn as nn
from functools import partial

# adding an auto padding functionality in pytorch
class Conv2dWithAutoPadding(nn.Conv2d):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.padding= (self.kernel_size[0]//2, self.kernel_size[1]//2)


class Conv2dWithBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv= nn.Sequential(
                                    Conv2dWithAutoPadding(in_channels=in_channels, out_channels=out_channels, **kwargs),
                                    nn.BatchNorm2d(out_channels)
                                )
    
    def forward(self, val):
        return self.conv(val)



class Residual_Base(nn.Module):
    def __init__(self, in_channels, out_channels, activation):
        super().__init__()
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        self.shortcut= nn.Identity()
        self.block= nn.Identity()

    def forward(self, val):
        val= self.block(val)+self.residual(val)
        return self.activation_function(val)
    
    @property
    def activation_function(self):
        return nn.ModuleDict([['relu', nn.ReLU(inplace=True)],
                              ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
                              ['selu', nn.SELU(inplace=True)],
                              ['none', nn.Identity()]])[self.activation]

class Residual_Shortcut(Residual_Base):
    def __init__(self, in_channels, out_channels, activation, downsampling):
        super().__init__(in_channels, out_channels, activation)
        self.shortcut= Conv2dWithBatchNorm(self.in_channels,self.out_channels, kernel_size=1,stride=downsampling)



class Residual_Block(Residual_Shortcut):
    def __init__(self, in_channels, out_channels, activation, downsampling):
        super().__init__(in_channels, out_channels, activation, downsampling)
        self.conv3x3= partial(Conv2dWithBatchNorm, kernel_size=3)
        self.block= nn.Sequential(
                                    self.conv3x3(self.in_channels,self.out_channels),
                                    self.activation_function,
                                    self.conv3x3(self.out_channels,self.out_channels)
                                 )
    
    

class Resnet_Layer(nn.Module):
    def __init__(self):
        super().__init__(self)