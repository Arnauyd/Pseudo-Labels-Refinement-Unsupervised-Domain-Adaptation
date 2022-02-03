# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 00:32:09 2021

@author: ArnaudA
"""
##############################################################################
#Import
##############################################################################

import torch
from torch import nn
import torch.nn.functional as F

##############################################################################
#Build simple Neural Network 
##############################################################################

class Conv_basic(nn.Module):
    def __init__(self,num_classes, n_channels):
        super(Conv_basic,self).__init__()
        self.layer1=nn.Conv2d(n_channels, 16, kernel_size= 5,stride=1, padding=2)
        self.relu=nn.ReLU()
        self.layer2=nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc=nn.Linear(16*16*16,num_classes)
                
    def forward(self,x):
        out=self.layer1(x)
        out=self.relu(out)
        out=self.layer2(out)
        out=out.reshape(out.size(0),-1)
        out=self.fc(out)
        return out
    
##############################################################################
#Build advanced Neural Network 
##############################################################################


class Conv_adv(nn.Module):
    def __init__(self,num_classes, n_channels):
        super(Conv_adv,self).__init__()
        self.layer1=nn.Sequential(nn.Conv2d(n_channels, 16, kernel_size= 5,stride=1, padding=2),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=2,stride=2))
        self.layer2=nn.Sequential(nn.Conv2d(16, 32, kernel_size= 5,stride=1, padding=2),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=2,stride=2))
        self.fc=nn.Linear(8*8*32,num_classes)
        self.fc1=nn.Linear(num_classes,num_classes)
                
    def forward(self,x):
        out=self.layer1(x)
        out=self.layer2(out)
        out=out.reshape(out.size(0),-1)
        out=self.fc(out)
        out=F.relu(out)
        out=self.fc1(out)
        return out


class Conv_adv_dro(nn.Module):
    def __init__(self,num_classes, n_channels):
        super(Conv_adv_dro,self).__init__()
        self.layer1=nn.Sequential(nn.Conv2d(n_channels, 16, kernel_size= 5,stride=1, padding=2),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=2,stride=2))
        self.layer2=nn.Sequential(nn.Conv2d(16, 32, kernel_size= 5,stride=1, padding=2),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=2,stride=2))
        self.fc=nn.Linear(8*8*32,num_classes)
        self.fc1=nn.Linear(num_classes,num_classes)
    
    def forward(self,x):
        out=self.layer1(x)
        out=self.layer2(out)
        out=F.dropout(out,p=0.25)
        out=out.reshape(out.size(0),-1)
        out=self.fc(out)
        #out=F.dropout(out,p=0.5)
        out=F.relu(out)
        out=self.fc1(out)
        return out    
    
class Conv_adv_bn(nn.Module):
    def __init__(self,num_classes, n_channels):
        super(Conv_adv_bn,self).__init__()
        self.layer1=nn.Sequential(nn.Conv2d(n_channels, 16, kernel_size= 5,stride=1, padding=2),
                                  nn.BatchNorm2d(16),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=2,stride=2))
        self.layer2=nn.Sequential(nn.Conv2d(16, 32, kernel_size= 5,stride=1, padding=2),
                                  nn.BatchNorm2d(32),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=2,stride=2))
        self.fc=nn.Linear(8*8*32,num_classes)
        self.fc1=nn.Linear(num_classes,num_classes)
                
    def forward(self,x):
        out=self.layer1(x)
        out=self.layer2(out)
        out=F.dropout(out,p=0.25)
        out=out.reshape(out.size(0),-1)
        out=self.fc(out)
        #out=F.dropout(out,p=0.5)
        out=F.relu(out)
        out=self.fc1(out)
        return out 

##############################################################################
#Resnet for black and white image (MNIST) Change 1 to 3 in first conv to adapt
#to color images 
##############################################################################


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel,stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size = 3, stride = stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size = 3, stride = 1, padding=1)  #we change the size only once
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample
    
    def forward(self,x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample: #to be used when input size does not match output size
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return(out)

class ResNet(nn.Module):
    def __init__(self, block, num_classes=10, n_channels=1):
        super(ResNet, self).__init__()
        self.in_channel = 16
        self.conv1 = nn.Conv2d(n_channels,16, stride =1, kernel_size = 3, padding = 1)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.block1 = self.make_layer(block, 16, 1)
        self.block2 = self.make_layer(block, 16, 1)
        self.block3 = self.make_layer(block, 32, 2)
        self.block4 = self.make_layer(block, 32, 1)
        self.block5 = self.make_layer(block, 64, 2)
        self.block6 = self.make_layer(block, 64, 1)
        self.avg_pool = nn.AvgPool2d(7) #8 is the kernel size so it is taking average of 8x8
        self.fc = nn.Linear(64, num_classes)
    def make_layer(self, block, out_channel, stride=1):
        downsample = None
        if(stride!=1) or (self.in_channel != out_channel):#input size not equal to output size only when stride not 1 or input channel and output channel are not same 
            downsample = nn.Sequential(
            nn.Conv2d(self.in_channel, out_channel, kernel_size = 3, stride = stride, padding = 1),
            nn.BatchNorm2d(out_channel))
        out_layer = block(self.in_channel, out_channel, stride, downsample)
        self.in_channel = out_channel
        return(out_layer)
    def forward(self,x):
        out = self.conv1(x)

        out = self.bn(out)
        out = self.relu(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out