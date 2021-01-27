import torch.nn as nn
from torchsummary import summary
from functools import partial

# adding an auto padding functionality 
class Conv2dWithAutoPadding(nn.Conv2d):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.padding= (self.kernel_size[0]//2, self.kernel_size[1]//2)



class Residual_Base(nn.Module):
    def __init__(self, in_channels, out_channels, activation):
        super().__init__()
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        self.shortcut= nn.Identity()
        self.block= nn.Identity()

    def forward(self, val):
        val= self.block(val)+self.shortcut(val)
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
   
        self.shortcut= self.conv_bn(in_channels, out_channels, kernel_size=1, stride=downsampling)

    @staticmethod
    def conv_bn(in_channels, out_channels, **kwargs):
        return nn.Sequential(
                            Conv2dWithAutoPadding(in_channels=in_channels, out_channels=out_channels,bias=False, **kwargs),
                            nn.BatchNorm2d(out_channels)
                            )


class Residual_Block(Residual_Shortcut):
    def __init__(self, in_channels, out_channels, activation, downsampling):
        super().__init__(in_channels, out_channels, activation, downsampling)
        self.conv3x3= partial(self.conv_bn, kernel_size=3)
        self.block= nn.Sequential(
                                    self.conv3x3(self.in_channels, self.out_channels, stride=downsampling),
                                    self.activation_function,
                                    self.conv3x3(self.out_channels,self.out_channels, stride=1)
                                 )


class Resnet_Layer(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, activation='relu', downsampling=1):
        super().__init__()

        self.block= Residual_Block
        # downsampling= 2 if in_channels!=out_channels else 1

        self.blocks= nn.Sequential(
                                    self.block(in_channels, out_channels, activation, downsampling= downsampling),
                                    *[ self.block(out_channels, out_channels, activation, downsampling=1) for _ in range(n-1)]
                                  )
                            
    def forward(self, val):
        return self.blocks(val)

class Resnet_Conv(nn.Module):
    """
    Convolution Layers.
    """
    def __init__(self, in_channels=3, k=64, n_blocks=[2,2,2,2], activation= 'relu', downsampling=2):
        super().__init__()
        self.block_sizes=[k, 2*k, 4*k, 8*k]

        self.conv_layer1= nn.Sequential(
                                    nn.Conv2d(in_channels,self.block_sizes[0], kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(self.block_sizes[0]),
                                    nn.ReLU(inplace=True)
                                  )   

        self.conv_layer2= Resnet_Layer(self.block_sizes[0], self.block_sizes[0], n_blocks[0], activation)
        self.conv_layer3= Resnet_Layer(self.block_sizes[0], self.block_sizes[1], n_blocks[1], activation, downsampling)
        self.conv_layer4= Resnet_Layer(self.block_sizes[1], self.block_sizes[2], n_blocks[2], activation, downsampling)
        self.conv_layer5= Resnet_Layer(self.block_sizes[2], self.block_sizes[3], n_blocks[3], activation, downsampling)
       

        self.conv_block= nn.Sequential( 
                                        self.conv_layer1, self.conv_layer2, self.conv_layer3, self.conv_layer4, self.conv_layer5
                                      )
    def forward(self, val):
        val= self.conv_block(val)
        return val

class ResNet_FC(nn.Module):
    """
    Fully connected layer
    """
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features, n_classes)

    def forward(self, val):
        val = self.avg(val)
        val = val.view(val.size(0), -1)
        val = self.fc(val)
        return val

class ResNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=10, *args, **kwargs):
        super().__init__()
        self.conv_layers = Resnet_Conv(in_channels, *args, **kwargs)
        self.fc_layers = ResNet_FC(self.conv_layers.block_sizes[-1], n_classes)
        
    def forward(self, val):
        val= self.conv_layers(val)
        val = self.fc_layers(val)
        return val


if __name__== '__main__':

    resnet18= ResNet()
    print(resnet18)

    summary(resnet18, (3,32,32))
