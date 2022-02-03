# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 22:26:18 2021

@author: ArnaudA
"""
##############################################################################
#Import
##############################################################################
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
import os
import CNN as Models
from torchvision import models
import Predict as predict
import cGan as cGans
import Train_cGan_NMIST as Train_cGans

##############################################################################
#Load Pretrained C0 
##############################################################################

#Define n_channels and adequate transformation
n_channels = 1
device =torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')

#Load C0
C0=Models.ResNet(Models.ResidualBlock, num_classes=10, n_channels = n_channels)
C0.load_state_dict(torch.load('./outputs/Model_Trained_SVHN_C1.pth'))
C0.to(device)

##############################################################################
#Load NMIST data and transform data
##############################################################################
    
n_channels = 1
    
if n_channels == 1: 
    mnist_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=n_channels),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (1.0))
        ])
else:
    mnist_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=n_channels),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
        ])
        
        
trainset = datasets.MNIST('./dataset/MNIST', train=True, transform=mnist_transform,download=True)
testset = datasets.MNIST('./dataset/MNIST', train=False, transform=mnist_transform,download=True)
batch_size=128
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)
            
#Batch description
dataiter = iter(trainloader)
images, labels=dataiter.next()
print('size of batch of labels: {}'.format(labels.shape))
print('size of batch of images: {}'.format(images.shape))

##############################################################################
#Predict Y_tild & target_datasets
##############################################################################

#y_tild = predict.predict_from_dataset(C0, trainset)

print('Target dataset calculation')

targetCtrainset = predict.predict_target_dataset(C0, trainset)

#target_testset = predict.predict_target_dataset(C0, testset)

print('Target loader calculation')

targetCloader = torch.utils.data.DataLoader(targetCtrainset, batch_size=batch_size,shuffle=True)

##############################################################################
#Pretrained G0,D0
##############################################################################

#dimensions
z_dim = 100
n_classes = 10
images_dim = trainset[0][0].shape[1] * trainset[0][0].shape[2]
    
#model
    
G0 = cGans.Generator_S(g_input_dim = z_dim, n_channels = n_channels, n_classes = n_classes).to(device)
D0 = cGans.Discriminator_S(images_dim, n_channels = n_channels, n_classes = n_classes).to(device)
    
# loss
loss = nn.BCELoss()
    
# optimizer
lr_G = 0.0001
lr_D = 0.0002
G0_optimizer = optim.Adam(G0.parameters(), lr = lr_G)
D0_optimizer = optim.Adam(D0.parameters(), lr = lr_D)    

###########################################################################
#Train
###########################################################################

print('start training GO and D0')

Train_cGans.GAN_train(G0, D0, G0_optimizer, D0_optimizer, targetCloader, n_channels = n_channels, n_epoch =25, criterion='BCELoss', noise = True, loss = loss)

print('training done')

torch.save(G0.state_dict(), './outputs/G0_MNIST.pth')
torch.save(D0.state_dict(), './outputs/D0_MNIST.pth') 