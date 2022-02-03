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
import Train_CNN_SVHN as Train_Cnn

##############################################################################
#Load Pretrained C0 ->C
##############################################################################

#Define n_channels and adequate transformation
n_channels = 1
device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')

#Load C0
C = Models.ResNet(Models.ResidualBlock, num_classes=10, n_channels = n_channels)
C.load_state_dict(torch.load('./outputs/Model_Trained_SVHN_C1.pth'))
C.to(device)

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
#Pretrained G0,D0 -> G,D
##############################################################################

#dimensions
z_dim = 100
n_classes = 10
images_dim = trainset[0][0].shape[1] * trainset[0][0].shape[2]
    
#model
G = cGans.Generator_S(g_input_dim = z_dim, n_channels = n_channels, n_classes = n_classes)
G.load_state_dict(torch.load('./outputs/G0_MNIST.pth'))
G.to(device)

D = cGans.Discriminator_S(images_dim, n_channels = n_channels, n_classes = n_classes)
D.load_state_dict(torch.load('./outputs/D0_MNIST.pth'))
D.to(device)
    
# loss
loss = nn.BCELoss()
    
# optimizer
lr_G = 0.0001
lr_D = 0.0002

G_optimizer = optim.Adam(G.parameters(), lr = lr_G)
D_optimizer = optim.Adam(D.parameters(), lr = lr_D) 

##############################################################################
#learning paramters
##############################################################################

n_samples = 128*60
batch_GAN_size = 128
n_iter = 5

##############################################################################
#learning of C phase 1
##############################################################################
for i in range(n_iter):

    print('Step 1 - improving classifier C iteration {}'.format(i))
    
    print('Generation of training dataset from cGAN Generator')
    # samples training dataset from cGAN generator
    targetGANtrainset = Train_cGans.generate_target_samples_from_Gan(G, n_samples)
    targetGANloader = torch.utils.data.DataLoader(targetGANtrainset, batch_size=batch_GAN_size, shuffle=True)
    
    #Hyperparameters
    num_epochs = 25
    learning_rate=0.01
    momentum=0.9
    optimizer = torch.optim.SGD(C.parameters(), learning_rate, momentum=momentum)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=0.1,verbose=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.8) 
    criterion = nn.CrossEntropyLoss()
    
    #learning
    print('Training of C')
    
    history_tr=[]
    for epoch in range(num_epochs):
        history_tr.append(Train_Cnn.train(C, targetGANloader, optimizer, epoch, log_interval=5, criterion=criterion))
        scheduler.step()
        print(scheduler._last_lr)
    
    #plot Loss
    x = np.arange(1, len(history_tr) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(x, history_tr, color='k', label = "Training loss", linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title("Evolution of the training and validation loss")
    plt.show()
    print('Training of C Done')
    
    #record current status of C
    torch.save(C.state_dict(), './outputs/C_MNIST_CNN.pth')
    
    
    ##############################################################################
    #Learning of G,D phase 2
    ##############################################################################
    
    print('Step 2 - improving cGAN G and D iteration {}'.format(i))
    
    #generate learning dataset for GAN
    samples = torch.randint(0, len(trainset), (n_samples,))
    
    print('Target dataset calculation')
    
    targetCtrainset = predict.predict_sampled_target_dataset(C, trainset, samples)
    
    print('Target loader calculation')
    
    targetCloader = torch.utils.data.DataLoader(targetCtrainset, batch_size=batch_size,shuffle=True)
    
    print('start training G and D')
    
    Train_cGans.GAN_train(G, D, G_optimizer, D_optimizer, targetCloader, n_channels = n_channels, n_epoch =25, criterion='BCELoss', noise = True, loss = loss)
    torch.save(G.state_dict(), './outputs/G_MNIST.pth')
    torch.save(D.state_dict(), './outputs/D_MNIST.pth') 
    
    print('training G and D done')

###########################################################################
#Evaluation of model 
###########################################################################
    
criterion = nn.CrossEntropyLoss()
print('Performance of model trained on NMIST on SVHN dataset : {}'.format(
        predict.test(C, testloader, criterion=criterion)))
    
print('Prédictions :{}'.format(predict.predict_from_dataset(C, trainset)))
