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
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

##############################################################################
#Display
##############################################################################

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


##############################################################################
#GAN
##############################################################################


class Generator(nn.Module):
    def __init__(self, g_input_dim, n_channels):
        super(Generator, self).__init__()       
        self.input = nn.Linear(g_input_dim, 256*4*4)
        self.ups2D = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv0 = nn.Conv2d(256, 128, kernel_size=(5,5), stride=(1,1), padding=(2,2), padding_mode ='replicate')
        self.conv1 = nn.Conv2d(128, 64, kernel_size=(5,5), stride=(1,1), padding=(2,2), padding_mode ='replicate')
        self.conv2 = nn.Conv2d(64, n_channels, kernel_size=(5,5), stride=(1,1), padding=(2,2), padding_mode ='replicate')
        self.BN0 = nn.BatchNorm2d(256,momentum=0.1)
        self.BN1 = nn.BatchNorm2d(128,momentum=0.1)
        self.BN2 = nn.BatchNorm2d(64,momentum=0.1)
        self.BN3 = nn.BatchNorm2d(n_channels,momentum=0.1)
    
    # forward method
    def forward(self, z): 
        x = self.input(z)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = x.view(-1,256,4,4)
        x = self.BN0(x)
        
        x = self.ups2D(x)
        x = self.conv0(x)
        x = self.BN1(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        
        x = self.ups2D(x)
        x = self.conv1(x)
        x = self.BN2(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        
        x = self.ups2D(x)
        x = self.conv2(x)
        x = self.BN3(x)
        x = torch.tanh(x)

        return x
    
class Discriminator(nn.Module):
    def __init__(self, d_input_dim, n_channels):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, 64,kernel_size=(5,5), stride=(2,2), padding=(2,2), padding_mode ='zeros')
        self.conv2 = nn.Conv2d(64, 128,kernel_size=(5,5), stride=(2,2), padding=(2,2), padding_mode ='zeros')
        self.conv3 = nn.Conv2d(128, 256,kernel_size=(5,5), stride=(2,2), padding=(2,2), padding_mode ='zeros')
        self.dro = nn.Dropout(p=0.3)
        self.fla = nn.Flatten()
        self.fc = nn.Linear(4096, 1)
        self.BN1 = nn.BatchNorm2d(64,momentum=0.1)
        self.BN2 = nn.BatchNorm2d(128,momentum=0.1)
        self.BN3 = nn.BatchNorm2d(256,momentum=0.1)

    
    # forward method
    def forward(self, x):

        x = self.conv1(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.BN1(x)
        
        x = self.dro(x)      
        x = self.conv2(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.BN2(x)
        
        x = self.dro(x)      
        x = self.conv3(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.BN3(x)
        
        x = self.dro(x)      
        x = self.fla(x)
        x = self.fc(x)
        x = torch.sigmoid(x)

        return x

##############################################################################
#Model parameters
##############################################################################

#dimensions
z_dim = 100
images_dim = trainset[0][0].shape[1] * trainset[0][0].shape[2]

#model

G = Generator(g_input_dim = z_dim, n_channels = n_channels).to(device)
D = Discriminator(images_dim, n_channels = n_channels).to(device)

# loss
criterion = nn.BCELoss()

# optimizer
lr = 0.0002
G_optimizer = optim.Adam(G.parameters(), lr = lr)
D_optimizer = optim.Adam(D.parameters(), lr = lr)    

##############################################################################
#Helper function
##############################################################################

def D_train(x, criterion='BCELoss'): # x is a batch composed of images
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # we use 1 for real and 0 for fake
    # train discriminator on real
    x_real = x
    y_real = torch.ones((x_real.shape[0],1)).to(device)
    x_real, y_real = Variable(x_real.to(device)), Variable(y_real.to(device))
    
    D_output = D(x_real)
    if criterion == 'BCELoss':
        D_real_loss = criterion(D_output, y_real)
    if criterion == 'LSLoss':
        D_real_loss = 0.5 * torch.mean((D_output-y_real)**2)
    

    # train discriminator on facke
    z = Variable(torch.randn(batch_size, z_dim).to(device))
    x_fake = G(z)
    y_fake = torch.zeros((batch_size,1)).to(device)
    
    D_output = D(x_fake)
    if criterion == 'BCELoss':
        D_fake_loss = criterion(D_output, y_fake)
    if criterion == 'LSLoss':
        D_fake_loss = 0.5 * torch.mean((D_output-y_fake)**2)
    

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()
        
    return  D_loss.data.item()

def G_train(criterion='BCELoss'):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = Variable(torch.randn(batch_size, z_dim).to(device))
    y = torch.ones((batch_size,1)).to(device)

    G_output =  G(z)
    D_output =  D(G_output)
    if criterion == 'BCELoss':
        G_loss =  criterion(D_output, y)
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
    
##############################################################################
#Train model
##############################################################################

n_epoch = 20
for epoch in range(1, n_epoch+1):           
    D_losses, G_losses = [], []
    for batch_idx, (x, _) in enumerate(trainloader):
        loss_D= D_train(x)
        D_losses.append(loss_D)
        loss_G= G_train()
        G_losses.append(loss_G)

    print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
            (epoch), n_epoch, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))
    
    if batch_idx % 5 == 0:
        plotLoss(epoch, D_losses, G_losses)
    

    with torch.no_grad():
        test_z = Variable(torch.randn(batch_size, z_dim).to(device))
        generated = G(test_z)
        plotGeneratedImages(generated.view(generated.size(0), n_channels, 32, 32),epoch)    

torch.save(G.state_dict(), './outputs/Gan_G_MNIST.pth')
torch.save(D.state_dict(), './outputs/Gan_D_MNIST.pth')

