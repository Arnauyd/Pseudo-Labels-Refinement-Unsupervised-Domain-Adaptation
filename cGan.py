# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 23:46:07 2021

@author: ArnaudA
"""

##############################################################################
#Import
##############################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
import IPython.display

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


##############################################################################
#cGAN Simplified
##############################################################################


class Generator_S(nn.Module):
    def __init__(self, g_input_dim, n_channels, n_classes=10):
        super(Generator_S, self).__init__()       
        #label encoding
        self.label_emb = nn.Embedding(n_classes, 50)
        self.fc_emb = nn.Linear(50, 8*8)
        #latent space
        self.input = nn.Linear(g_input_dim, 128*8*8)
        self.ups2D = nn.UpsamplingNearest2d(scale_factor=2)
        #latent and label space
        self.conv0 = nn.Conv2d(129, 128, kernel_size=(5,5), stride=(1,1), padding=(2,2), padding_mode ='replicate')
        self.conv1 = nn.Conv2d(128, 128, kernel_size=(5,5), stride=(1,1), padding=(2,2), padding_mode ='replicate')
        self.conv2 = nn.Conv2d(128, 64, kernel_size=(5,5), stride=(1,1), padding=(2,2), padding_mode ='replicate')
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(5,5), stride=(1,1), padding=(2,2), padding_mode ='replicate')
        self.conv4 = nn.Conv2d(64, n_channels, kernel_size=(5,5), stride=(1,1), padding=(2,2), padding_mode ='replicate')
        self.BN0 = nn.BatchNorm2d(129,momentum=0.1)
        self.BN1 = nn.BatchNorm2d(128,momentum=0.1)
        self.BN2 = nn.BatchNorm2d(64,momentum=0.1)
        self.BN3 = nn.BatchNorm2d(n_channels,momentum=0.1)
        self.dro = nn.Dropout(p=0.3)
    
    # forward method
    def forward(self, x, label): 
        
        #latent space and label encoding concatenation
        x = self.input(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = x.view(-1,128,8,8)
        label = self.label_emb(label)
        label = self.fc_emb(label)
        label = label.view(-1,1,8,8)
        x = torch.cat((x,label),1)
        x = self.BN0(x)
        
        #First Convolution + upsampling
        x = self.dro(x)
        x = self.conv0(x)
        x = self.BN1(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        
        #First convolution 
        x = self.dro(x)
        x = self.conv1(x)
        x = self.BN1(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        
        #Second convolution + upsampling
        x = self.ups2D(x)
        x = self.dro(x)
        x = self.conv2(x)
        x = self.BN2(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        
        #First convolution 
        x = self.dro(x)
        x = self.conv3(x)
        x = self.BN2(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        
        #Second convolution + upsampling
        x = self.ups2D(x)
        x = self.dro(x)
        x = self.conv4(x)
        x = self.BN3(x)
        x = torch.tanh(x)

        return x
    
class Discriminator_S(nn.Module):
    def __init__(self, d_input_dim, n_channels, n_classes=10):
        super(Discriminator_S, self).__init__()
        #label encoding
        self.label_emb = nn.Embedding(n_classes, 50)
        self.fc_emb = nn.Linear(50, 32*32)
        #latent and label space
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d((n_channels+1), 64,kernel_size=(5,5), stride=(2,2), padding=(2,2), padding_mode ='replicate'))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(64, 128,kernel_size=(5,5), stride=(2,2), padding=(2,2), padding_mode ='replicate'))
        self.conv3 = nn.utils.spectral_norm(nn.Conv2d(128, 128,kernel_size=(5,5), stride=(1,1), padding=(2,2), padding_mode ='replicate'))
        self.dro = nn.Dropout(p=0.3)
        self.fla = nn.Flatten()
        self.fc = nn.Linear(8*8*128, 1)
        self.BN1 = nn.BatchNorm2d(64,momentum=0.1)
        self.BN2 = nn.BatchNorm2d(128,momentum=0.1)

    
    # forward method
    def forward(self, x, label):
        
        #label encoding
        label = self.label_emb(label)
        label = self.fc_emb(label)
        label = label.view(-1,1,32,32)
        x = torch.cat((x,label),1)
        #Classification
        x = self.conv1(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.BN1(x) 
        x = self.dro(x)      
        x = self.conv2(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.BN2(x)  
        x = self.dro(x)      
        x = self.fla(x)
        x = self.fc(x)
        x = torch.sigmoid(x)

        return x