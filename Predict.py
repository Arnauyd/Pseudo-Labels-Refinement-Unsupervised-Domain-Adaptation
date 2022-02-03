# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 21:53:13 2021

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


##############################################################################
#Helper Functions
##############################################################################

device =torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))  

def test(model, test_loader, criterion = nn.CrossEntropyLoss()):
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
    print('tMSE LOSS: {:.6f} \tMSE Accuracy: {:.6f}'.format(test_loss_MSE, acc_loss_MSE))

    return test_loss_MSE, acc_loss_MSE 

def predict_from_loader(model, trainloader):
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(trainloader, 0):
            images, _ = data
            images = images.to(device)
            outputs  = model(images)
            batch_results= torch.argmax(outputs,axis=1)
            if batch_idx ==0:
                results = batch_results
            else:
                results=torch.cat((results,batch_results),axis=0)

    print('Prediction complete')

    return results 


def predict_from_dataset(model, trainset):
    model.eval()
    loader=torch.utils.data.DataLoader(trainset, batch_size=1,shuffle=False)
    results= torch.zeros((len(loader),1), dtype=torch.int32)
    with torch.no_grad():
        for idx, (image,_) in enumerate(loader,0):
          image = image.to(device)
          results[idx] = torch.argmax(model(image),axis=1)
    return results

    print('Prediction complete')

    return results 

class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels=None):
        self.X = images
        self.y = labels
         
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        data = self.X[i, :]
            
        if self.y is not None:
            return (data, self.y[i])
        else:
            return data

def predict_target_dataset(model, dataset):
    
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=1,shuffle=False)
    y_tilds = torch.zeros((len(loader),),dtype=torch.long)

    with torch.no_grad():
        for idx, (image,label) in enumerate(loader,0):
          y_tilds[idx] = torch.argmax(model(image.to(device)),axis=1)
          if idx == 0:
              images = image
          else:
              images = torch.cat((images,image),dim=0)
        
    data = MNISTDataset(images, y_tilds)
    
    print('Prediction complete')
    
    return data

def predict_sampled_target_dataset(model, dataset, samples):
    
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=1,shuffle=False)
    y_tilds = torch.zeros((len(samples),),dtype=torch.long)
    it=0
    with torch.no_grad():
        for idx, (image,label) in enumerate(loader,0):
            if idx in samples:
                y_tilds[it] = torch.argmax(model(image.to(device)),axis=1)
                if it == 0:
                    images = image
                else:
                    images = torch.cat((images,image),dim=0)
                it +=1
        
    data = MNISTDataset(images, y_tilds)
    
    print('Prediction complete')
    
    return data

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
    model.load_state_dict(torch.load('./outputs/Model_Trained_SVHN_C1.pth'))
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
    
    
    
    trainset = datasets.MNIST('./dataset/MNIST', train=True, transform=transform,download=True)
    testset = datasets.MNIST('./dataset/MNIST', train=False, transform=transform,download=True)
    batch_size=128
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)
    
    #Sample description
    print('Number of training samples: {}'.format(len(trainset)))
    print('Number of test samples: {}'.format(len(testset)))
    print('sample size: {}'.format(trainset[0][0].shape))
    print('image rows: {}'.format(trainset[0][0].shape[1]))
    print('image cols: {}'.format(trainset[0][0].shape[2]))
    print('Classes: {}'.format(np.unique(trainloader.dataset.targets)))
    
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
    plt.hist(trainloader.dataset.targets)
    

    ###########################################################################
    #Evaluation of model 
    ###########################################################################
    
    criterion = nn.CrossEntropyLoss()
    print('Performance of model : {}'.format(
        test( model, testloader, criterion=criterion)))
    
    print('Prédictions :{}'.format(predict_from_dataset(model, trainset)))




