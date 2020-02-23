# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 19:28:31 2020

@author: 12899
"""
import torch as t
import torch.nn as nn

class NumberClassifier(nn.Module):
    def __init__(self):
        super(NumberClassifier,self).__init__()
        self.layer1=nn.Sequential(nn.Conv2d(1,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2,stride=2))
        self.layer2=nn.Sequential(nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2,stride=2))
        self.layer3=nn.Sequential(nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),nn.MaxPool2d(2,stride=2))
        self.layer4=nn.Sequential(nn.Linear(3*3*128,10),nn.Sigmoid())
    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=x.view(-1,3*3*128)
        x=self.layer4(x)
        return x