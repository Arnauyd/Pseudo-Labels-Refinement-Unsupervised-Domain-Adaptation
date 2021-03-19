# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 22:25:22 2021

@author: ArnaudA
"""

##############################################################################
#Import
##############################################################################
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import os

##############################################################################
#Load and preprocessing
##############################################################################

transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(0.5,1.0)])

trainset = datasets.MNIST('./dataset/MNIST', train=True, transform=transform,download=True)
testset = datasets.MNIST('./dataset/MNIST', train=False, transform=transform,download=True)
batch_size=64
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

#Sample description
print('Number of training samples: {}'.format(len(trainset)))
print('Number of test samples: {}'.format(len(trainset)))
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

plt.imshow(images[0].numpy().squeeze(),cmap='gray_r')

figure = plt.figure()
num_of_images = 60
for index in range(1, num_of_images + 1):
    plt.subplot(6, 10, index)
    plt.axis('off')
    plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')

figure = plt.figure()
plt.title('Balance of classes')    
plt.hist(trainloader.dataset.targets)
    
##############################################################################
#Build simple Neural Network 
##############################################################################

input_size=trainset[0][0].shape[1]*trainset[0][0].shape[2]
hidden_sizes = [128,64]
output_size= 10

basic_model=nn.Sequential(nn.Linear(input_size,hidden_sizes[0]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[1], output_size),
                          nn.LogSoftmax(dim=1))

criterion= nn.NLLLoss()
images, labels=next(iter(trainloader))
images=images.view(images.shape[0],-1)

logps = basic_model(images)
loss = criterion(logps,labels)
print(loss)

##############################################################################
#Weights 
##############################################################################

print('Before backward pass: \n', basic_model[0].weight.grad)
loss.backward()
print('After backward pass: \n', basic_model[0].weight.grad)

##############################################################################
#Core Training
##############################################################################

optimizer=optim.SGD(basic_model.parameters(),lr=0.003,momentum=0.9)
time0=time()
epochs=15
for e in range(epochs):
    running_loss=0
    for images,labels in trainloader:
        #flatten images
        images=images.view(images.shape[0],-1)
        
        #training step
        optimizer.zero_grad()
        output=basic_model(images)
        loss=criterion(output,labels)
        
        #backpropagation
        loss.backward()
        
        #Optimize weight
        optimizer.step()
        
        running_loss +=loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))
        
print("\nTraining Time (in minutes) =",(time()-time0)/60)
        
##############################################################################
#Test result
##############################################################################

def view_classify(img, ps):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.cpu().data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()


images,labels=next(iter(testloader))

img=images[0].view(1,784)

with torch.no_grad() :
    logps=basic_model(img)

ps=torch.exp(logps)
probab=list(ps.numpy()[0])
print("Predicted Digit =", probab.index(max(probab)))
view_classify(img.view(1, 28, 28), ps)

##############################################################################
#Test results
##############################################################################        
        
correct_count, all_count = 0, 0
for images,labels in testloader:
  for i in range(len(labels)):
    img = images[i].view(1, 784)
    with torch.no_grad():
        logps = basic_model(img)

    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    pred_label = probab.index(max(probab))
    true_label = labels.numpy()[i]
    if(true_label == pred_label):
      correct_count += 1
    all_count += 1

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))

torch.save(basic_model, './my_mnist_model.pt')
