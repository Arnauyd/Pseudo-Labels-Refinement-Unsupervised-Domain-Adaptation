# -*- coding: utf-8 -*-
"""
Created on Fri May 14 16:11:17 2021

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
import Predict as Predict


##############################################################################
#Main 
##############################################################################

if __name__ == "__main__":

    ###########################################################################
    #Load Model
    ###########################################################################
    
    #Define n_channels and adequate transformation
    n_channels = 1
    
    
    device =torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')
    model=Models.ResNet(Models.ResidualBlock, num_classes=10, n_channels = n_channels)
    model.load_state_dict(torch.load('./outputs/C_MNIST_CNN.pth'))
    #model=models.resnet50(pretrained=True)
    #model.eval()
    model.to(device)
    
    ###########################################################################
    #Load NMIST data and transform data
    ###########################################################################
    
    if n_channels == 1: 
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=n_channels),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (1.0))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=n_channels),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
        ])
    
    
    
    trainset = datasets.SVHN('./dataset/SVHN', split='train', transform=transform,download=True)
    testset = datasets.SVHN('./dataset/SVHN', split='test', transform=transform,download=True)
    batch_size=128
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)
    
    #Sample description
    print('Number of training samples: {}'.format(len(trainset)))
    print('Number of test samples: {}'.format(len(testset)))
    print('sample size: {}'.format(trainset[0][0].shape))
    print('image rows: {}'.format(trainset[0][0].shape[1]))
    print('image cols: {}'.format(trainset[0][0].shape[2]))
    print('Classes: {}'.format(np.unique(trainloader.dataset.labels)))
    
    #Batch description
    dataiter = iter(trainloader)
    images, labels=dataiter.next()
    print('size of batch of labels: {}'.format(labels.shape))
    print('size of batch of images: {}'.format(images.shape))
    
    ###########################################################################
    #Display
    ###########################################################################
    
    plt.imshow(images[0].permute(1,2,0).numpy().squeeze(),cmap='gray_r')
    
    figure = plt.figure()
    num_of_images = 60
    for index in range(1, num_of_images + 1):
        plt.subplot(6, 10, index)
        plt.axis('off')
        plt.imshow(images[index].permute(1,2,0).numpy().squeeze(), cmap='gray_r')
    
    figure = plt.figure()
    plt.title('Balance of classes')    
    plt.hist(trainloader.dataset.labels)
    
    ###########################################################################
    #Evaluation of model 
    ###########################################################################
    
    criterion = nn.CrossEntropyLoss()
    print('Performance of model : {}'.format(
        Predict.test( model, testloader, criterion=criterion)))
    
    print('Prédictions :{}'.format(Predict.predict_from_dataset(model, trainset)))