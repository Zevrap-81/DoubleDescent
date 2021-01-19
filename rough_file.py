import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn



class Residual_Base(nn.Module):
    def __init__(self, in_channels, out_channels, activation):
        super().__init__()
        self.activation= activation
        self.shortcut= nn.Identity()
        self.block= nn.Identity()

    def forward(self, val):
        val= self.block(val)+self.shortcut(val)
        return self.act_func(val)
    
    @property
    def act_func(self):
        return nn.ModuleDict([['relu', nn.ReLU(inplace=True)],
                              ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
                              ['selu', nn.SELU(inplace=True)],
                              ['none', nn.Identity()]])[self.activation]
                        
a=Residual_Base(3,10,'relu')
b=torch.randn((1,3)) 
print(a(b))