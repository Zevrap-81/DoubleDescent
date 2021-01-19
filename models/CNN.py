
import torch.nn as nn
import torch

class CNN(nn.Module):
    def __init__(self,k, n_classes):
        super().__init__()
        self.k= k
        self.conv_block= nn.Sequential(
                    #Layer0: input batchsize x inputchannel(3) x 32 x 32
                        nn.Conv2d(3,self.k, (3,3), stride=1, padding=1, bias=True),
                        nn.ReLU(),
                        nn.BatchNorm2d(self.k),
                            #output batchsize x outputchannel(k) x 32 x 32
                        
                    #Layer1: input batchsize x inputchannel(3) x 32 x 32
                        nn.Conv2d(self.k,2*self.k, (3,3), stride=1, padding=1, bias=True),
                        nn.ReLU(),
                        nn.MaxPool2d((2,2),stride=2),
                        nn.BatchNorm2d(2*self.k),
                            #output batchsize x outputchannel(2k) x 16 x 16

                    #Layer2: input batchsize x inputchannel(2k) x 16 x 16
                        nn.Conv2d(2*self.k,4*self.k, (3,3), stride=1, padding=1, bias=True),
                        nn.ReLU(),
                        nn.MaxPool2d((2,2),stride=2),
                        nn.BatchNorm2d(4*self.k),
                            #output batchsize x outputchannel(4k) x 8 x 8 

                    #Layer3: input batchsize x inputchannel(3) x 8 x 8
                        nn.Conv2d(4*self.k,8*self.k, (3,3), stride=1, padding=1, bias=True),
                        nn.ReLU(),
                        nn.MaxPool2d((2,2), stride=2),
                        nn.BatchNorm2d(8*self.k),
                            #output batchsize x outputchannel(k) x 4 x 4
                        nn.MaxPool2d((4,4), stride=1))
                            #output batchsize x outputchannel(k) x 1 x 1

        self.fc_block=nn.Linear(8*self.k, n_classes, bias=True)

    def forward(self, data):
        data= self.conv_block(data)
        data= self.fc_block(data.view(-1, 8*self.k))
        return data