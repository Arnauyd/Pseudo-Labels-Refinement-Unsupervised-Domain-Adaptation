# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 23:46:07 2021

@author: ArnaudA

In this section I will apply an unconditionnal Gan to the the dataset SVHN 
and / or to the NMIST dataset with three channels of grayscale.

The main objectif is to obtain a good understanding and clear view of the performance
of a Gan.
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
import cGan as cGans
import math


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


##############################################################################
#Helper function
##############################################################################

def D_train(G, D, G_optimizer, D_optimizer, x, label, epoch, criterion='BCELoss', noise = False, p_flip=5, loss = nn.BCELoss(), z_dim=100): 
    '''
    
    Parameters
    ----------
    x : batch of samples
    label : labels of images.
    epoch : current epoch.
    criterion : TYPE, optional
        type of loss ('BCE or MSELoss'). The default is 'BCELoss'.
    noise : TYPE, optional
        Boolean to add noise on images fed to the discriminator. The default is False.
    p_flip : TYPE, optional
        Percent of labels to flip to lure dicriminator . The default is 5.

    Returns
    -------
    Trained D model.

    '''
    D.zero_grad()
    
    #label smoothing
    real_smoothing = torch.FloatTensor(label.shape[0],1).uniform_(0.7, 0.9)
    fake_smoothing = torch.FloatTensor(label.shape[0],1).uniform_(0.0, 0.3)
    
    #noise
    #print('Noise status {}'.format(noise))
    mean = 0
    std = 1 / epoch
    if (noise and epoch<20):
        noise_mask = ((torch.randn(label.shape[0], x.shape[1], 32, 32) + mean) * std)
    else:
        noise_mask = torch.zeros_like(x)
    
    # we use 1 for real and 0 for fake
    # train discriminator on real
    x_real = x + noise_mask
    y_real = (torch.ones((label.shape[0],1))*real_smoothing).to(device)

    #flip of positive label
    samples = torch.randint(label.shape[0],(int(label.shape[0]*p_flip/100),))
    y_real[samples] = 0
    x_real, y_real, label = Variable(x_real.to(device)), Variable(y_real.to(device)), label.to(device)
    
    
    D_output = D(x_real, label)
    if criterion == 'BCELoss':
        D_real_loss = loss(D_output, y_real)
    if criterion == 'LSLoss':
        D_real_loss = 0.5 * torch.mean((D_output-y_real)**2)
    

    # train discriminator on facke
    z = Variable(torch.randn(label.shape[0], z_dim).to(device))
    x_fake = G(z, label) + noise_mask.to(device)
    y_fake = (torch.zeros((label.shape[0],1))+fake_smoothing).to(device)
    
    #flip of positive label
    samples = torch.randint(label.shape[0],(int(label.shape[0]*p_flip/100),))
    y_fake[samples] = 1
    
    D_output = D(x_fake, label)
    if criterion == 'BCELoss':
        D_fake_loss = loss(D_output, y_fake)
    if criterion == 'LSLoss':
        D_fake_loss = 0.5 * torch.mean((D_output-y_fake)**2)
    

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()
        
    return  D_loss.data.item()

def G_train(G, D, G_optimizer, criterion='BCELoss', loss = nn.BCELoss(), z_dim=100, batch_size = 128):
    
    G.zero_grad()

    z = Variable(torch.randn(batch_size, z_dim).to(device))
    label = Variable(torch.randint(0, 10, (batch_size,)).to(device))
    y = torch.ones((batch_size,1)).to(device)

    G_output =  G(z, label)
    D_output =  D(G_output, label)
    if criterion == 'BCELoss':
        G_loss =  loss(D_output, y)
    if criterion == 'LSLoss':
        G_loss = 0.5 * torch.mean((D_output - y)**2)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()
        
    return G_loss.data.item()

# Plot the loss from each batch
def plotLoss(epoch, dLosses, gLosses):
    plt.figure(figsize=(10, 8))
    plt.plot(dLosses, label='Discriminitive loss')
    plt.plot(gLosses, label='Generative loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('gan_loss_epoch_%d.png' % epoch)

# Create a wall of generated images
def plotGeneratedImages(generatedImages,epoch, dim=(10, 10), figsize=(10, 10)):
    generatedImages=generatedImages.cpu().permute(0,2,3,1).numpy()
    print(generatedImages.shape)
    plt.figure(figsize=figsize)
    for i in range(100):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(np.squeeze(generatedImages[i]), interpolation='nearest', cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('./outputs/gan_generated_image_epoch_%d.png' % epoch)
    IPython.display.display(IPython.display.Image(data=('./outputs/gan_generated_image_epoch_%d.png' % epoch)))
    


def GAN_train(G, D, G_optimizer, D_optimizer, trainloader,n_channels = 1, n_epoch =20, criterion='BCELoss', noise = True,
              loss = nn.BCELoss(), z_dim =100, batch_size = 128, n_classes =10):
    
    for epoch in range(1, n_epoch+1):
        D_losses, G_losses = [], []
        for batch_idx, (x, label) in enumerate(trainloader):
            loss_D= D_train(G, D, G_optimizer, D_optimizer, x, label, epoch, criterion, noise = noise, loss = loss)
            D_losses.append(loss_D)
            loss_G= G_train(G, D, G_optimizer, criterion, loss = loss )
            G_losses.append(loss_G)
        
        print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
            (epoch), n_epoch, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))
                
        if batch_idx % 5 == 0:
            plotLoss(epoch, D_losses, G_losses)
             
        with torch.no_grad():
            test_z = Variable(torch.randn(batch_size, z_dim).to(device))
            test_label = Variable(torch.randint(0, n_classes, (batch_size,)).to(device))
            generated = G(test_z, test_label)
            plotGeneratedImages(generated.view(generated.size(0), n_channels, 32, 32),epoch)    


class TargetDataset(torch.utils.data.Dataset):
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

def generate_target_samples_from_Gan(G, n_samples, z_dim = 100, n_classes =10):
    '''
    Parameters
    ----------
    G : 
        cGan generator.
    n_samples : 
        number of samples to generate.
    zdim : TYPE, optional
        dimension of latent space. The default is 100.
    n_classes : TYPE, optional
        number of classes. The default is 10.

    Returns
    -------
    data : torch.utils.data.Dataset
        dataset of target samples consisting of generated images and labels.

    '''
    batch_size = 128
    n_batchs = int(n_samples/batch_size)
    
    for i in range(n_batchs):
    
        with torch.no_grad():
            samples_z = Variable(torch.randn(batch_size, z_dim).to(device))
            samples_label = Variable(torch.randint(0, n_classes, (batch_size,)).to(device))
            generated_images = G(samples_z, samples_label)
            
        if i == 0:
            labels = samples_label
            images = generated_images
        else:
            labels = torch.cat((labels,samples_label))
            images = torch.cat((images,generated_images))
    
        
    data = TargetDataset(images.cpu(), labels.cpu())
    
    print('generation of target dataset complete')
    
    return data


##############################################################################
#Main
##############################################################################

if __name__ == "__main__":
    
    ###########################################################################
    #Load NMIST data and transform data
    ###########################################################################
    
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
    #Model parameters
    ###########################################################################
    
    #dimensions
    z_dim = 100
    n_classes = 10
    images_dim = trainset[0][0].shape[1] * trainset[0][0].shape[2]
    
    #model
    
    G = cGans.Generator_S(g_input_dim = z_dim, n_channels = n_channels, n_classes = n_classes).to(device)
    D = cGans.Discriminator_S(images_dim, n_channels = n_channels, n_classes = n_classes).to(device)
    
    # loss
    loss = nn.BCELoss()
    
    # optimizer
    lr_G = 0.0001
    lr_D = 0.0002
    G_optimizer = optim.Adam(G.parameters(), lr = lr_G)
    D_optimizer = optim.Adam(D.parameters(), lr = lr_D)    
    
    ###########################################################################
    #Train
    ###########################################################################

    GAN_train(G, D, G_optimizer, D_optimizer, trainloader, n_channels = n_channels, n_epoch =25, criterion='BCELoss', noise = True)

    torch.save(G.state_dict(), './outputs/cGanS_G0_MNIST.pth')
    torch.save(D.state_dict(), './outputs/cGanS_D0_MNIST.pth')    
