import math
import numpy as np  
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from torchvision.utils import make_grid
# import horovod.torch as hvd
import time
import pickle

from models.ResNet import *

use_horovod= False
use_gpu= torch.cuda.is_available()
seed= 2009

if use_gpu:
    if torch.cuda.device_count() >= 1: # change to >1
        use_horovod= use_gpu and use_horovod

if use_horovod:
    # Initialize horovod
    hvd.init()
    # pin GPU to local rank
    torch.cuda.set_device(hvd.local_rank())

    # Using seeds to repreduce the same result
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)



train_set= datasets.CIFAR10(root='data', 
                            download=True,
                            train= True, 
                            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))]))
test_set= datasets.CIFAR10(root='data', 
                           download=True, 
                           train=False, 
                           transform= transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))]))

print(f"train_set length:{len(train_set)}")
print(f"test_set length:{len(test_set)}")



n_classes=10
batch_size=256
train_sampler, val_sampler= None, None
kwargs={}

if use_horovod:
    kwargs= {'num_workers': 1, 'pin_memory': True}
    batch_size=batch_size*hvd.size()
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, 
                                                                   num_replicas=hvd.size(), 
                                                                   rank=hvd.rank())

    val_sampler = torch.utils.data.distributed.DistributedSampler(test_set, 
                                                                  num_replicas=hvd.size(), 
                                                                  rank=hvd.rank())

train_loader= DataLoader(train_set, batch_size=batch_size, sampler= train_sampler, **kwargs)
val_loader= DataLoader(test_set, batch_size=batch_size, sampler= val_sampler, **kwargs)





def denorm(img):
    return (img/2+0.5).clamp_(0,1)
def plot_img(img_batch):
    grid= make_grid(img_batch, nrow=math.ceil(math.sqrt(batch_size)), pad_value=2)
    fig = plt.figure(figsize=(10,10))
    plt.imshow(denorm(grid).permute(1,2,0))
    plt.show()


'''
    Gets the value from all the devices and averages it using all-reduce
'''
def average_loss(val,name):
        tensor= torch.tensor(val)
        avg_loss=hvd.allreduce(tensor,name=name)
        return avg_loss.item() 


def train(k, epochs):
    
    model= ResNet(k=k)
    opt= torch.optim.Adam(model.parameters(), lr= 1e-4)
    criterion= nn.CrossEntropyLoss()
    
    if use_horovod:
        model.to('cuda')
        # broadcast parameters and optimizer state from root device to other devices
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(opt, root_rank=0)
        
        # Wraps the opimizer for multiGPU operation
        opt = hvd.DistributedOptimizer(opt,  
                                      named_parameters=model.named_parameters(),
                                      op = hvd.Adasum)

    
    loss_dict= {'epoch':[],'train':[], 'val':[]}
    
    for epoch in range(epochs):
        train_loss= 0
        val_loss= 0
        
        # train block
        for img_batch, labels_batch in train_loader:
            if use_gpu:
                img_batch= img_batch.to('cuda')
                labels_batch= labels_batch.to('cuda')
            
            pred= model(img_batch)
            
            opt.zero_grad()
            loss= criterion(pred, labels_batch)
            loss.backward()
            opt.step()
            train_loss+=loss.item()
            
        #val block
        with torch.no_grad():
            for img_batch, labels_batch in val_loader:
                if use_gpu:
                    img_batch= img_batch.to('cuda')
                    labels_batch= labels_batch.to('cuda')
                    
                pred= model(img_batch)
                loss= criterion(pred, labels_batch)
                val_loss+=loss.item()
        
        if use_horovod:
            train_loss= average_loss(train_loss, 'avg_train_loss')
            val_loss= average_loss(val_loss, 'avg_val_loss')
        
                    
        loss_dict['epoch'].append(epoch+1)
        loss_dict['train'].append(train_loss)
        loss_dict['val'].append(val_loss)
        
        print(",".join(["{}:{:.2f}".format(key, val[epoch]) for key, val in loss_dict.items()]))
    
    torch.save(model.state_dict(), "models/modelsdata/ResNet18_Cifar10_d{}.ckpt".format(k))
    save_obj(loss_dict, "models/modelsdata/losses/ResNet18_Cifar10_d{}".format(k))
    return loss_dict        

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def plot(dd_loss_dict):
    ks= list(dd_loss_dict.keys())
    train_loss=[ dd_loss_dict[k]['train'][-1] for k in ks]
    val_loss=[ dd_loss_dict[k]['val'][-1] for k in ks]
    
    fig = plt.figure(figsize=(10,8))
    plt.plot(ks, train_loss,label='Train')
    plt.plot(ks, val_loss, label='Val')

    plt.ylabel('Cross-Entropy Loss')
    plt.xlabel('Model-Width-Parameter')
    plt.legend(fontsize=20, ncol=3) 

    plt.title('Generalisation Loss', pad=12) 


def resnet18_mdl_dd():
    width_scales= [8,16,32,64,128,256,512]
    dd_loss_dict={}
    epochs=30
    for k in width_scales:
        dd_loss_dict[k]= train(k, epochs)
    plot(dd_loss_dict)
    
    return dd_loss_dict

resnet18_mdl_dd()
# dd_loss_dict={}
# for k in [8,16,32,64,128,256]:
#     dd_loss_dict[k]= load_obj("models/modelsdata/losses/ResNet18_Cifar10_d{}".format(k))
# plot(dd_loss_dict)