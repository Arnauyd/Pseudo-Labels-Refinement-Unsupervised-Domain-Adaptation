# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 22:42:12 2021

@author: ArnaudA

This script permits to train one CNN model on a SVHN or NMIST dataset with
parameters n_channels  and grayscale.

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


##############################################################################
#helper functions 
##############################################################################

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))    
    

def train( model, train_loader, optimizer, epoch, log_interval=10, criterion = nn.CrossEntropyLoss() ):
    device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')
    model.train()
    loss_cpu = 0
    acc_cpu = 0
    for batch_idx, data in enumerate(train_loader, 0):

        # get the inputs; data is a list of [inputs, labels]
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = model(images)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        
        #record losses and accuracies
        loss_cpu += loss.cpu().item()
        acc_cpu += accuracy(outputs, labels)

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(labels), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            #n_iter=epoch * len(train_loader) + batch_idx
    #scheduler.step(loss_cpu / len(train_loader.dataset))
    #print(scheduler._last_lr)
    return loss_cpu/len(train_loader), acc_cpu/len(train_loader)

def test( model, test_loader, epoch, criterion = nn.CrossEntropyLoss()):
    model.eval()
    loss =0
    acc =0 
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader, 0):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs  = model(images)
            loss += criterion(outputs, labels)
            acc += accuracy(outputs, labels)

    test_loss_MSE = loss.item()/ len(test_loader)
    acc_loss_MSE = acc.item()/ len(test_loader)
    print('Test Epoch: {} \tMSE LOSS: {:.6f} \tMSE Accuracy: {:.6f}'.format(
        epoch, test_loss_MSE, acc_loss_MSE))

    return test_loss_MSE, acc_loss_MSE 

colors = [[31, 120, 180], [51, 160, 44]]
colors = [(r / 255, g / 255, b / 255) for (r, g, b) in colors]

def plot_losses(train_history, val_history):
    x = np.arange(1, len(train_history) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(x, train_history, color=colors[0], label="Training loss", linewidth=2)
    plt.plot(x, val_history, color=colors[1], label="Validation loss", linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title("Evolution of the training and validation loss")
    plt.show()

##############################################################################
#Main 
##############################################################################

if __name__ == "__main__":

    ###########################################################################
    #Load and preprocessing dataset
    ###########################################################################
    
    #Define n_channels and adequate transformation
    n_channels = 1
    
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
    #Evaluation of basic model
    ###########################################################################

    #Hypeerparameters
    device =torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')
    num_epochs =30
    num_classes=10
    learning_rate=0.01
    momentum=0.9
    
    #Choice of models
    #model=models.resnet50(pretrained=False)
    #model.eval().to(device)
    model= Models.Conv_adv_bn(num_classes, n_channels).to(device)
    model=Models.ResNet(Models.ResidualBlock,n_channels=n_channels).to(device)
    
    
    optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=momentum)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=0.1,verbose=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.8) 
    criterion = nn.CrossEntropyLoss()
    
    
    history_tr=[]
    history_te=[]
    
    for epoch in range(num_epochs):
        history_tr.append(train( model, trainloader, optimizer, epoch, log_interval=50, criterion=criterion ))
        scheduler.step()
        print(scheduler._last_lr)
        history_te.append(test( model, testloader, epoch, criterion=criterion))
    
    print('Training Done')
    plot_losses(history_tr, history_te)
    print('Best accuracy model basique :{}'.format(history_te[-1][1]))
    
    torch.save(model.state_dict(), './outputs/ResNet50_Trained_SVHN_C1.pth')

