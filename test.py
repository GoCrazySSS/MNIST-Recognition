# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 12:30:31 2020

@author: 12899
"""
from dataset import MyDataSet,MyDataLoader
from model import NumberClassifier
import torch as t
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.autograd import Variable

BATCH_SIZE=32
NUM_WORKERS=8
sample_number=10000 if 10000%BATCH_SIZE==0 else 10000-(10000%BATCH_SIZE)
device=t.device('cuda:0' if t.cuda.is_available() else 'cpu')
net=t.load('./output/model_ep49.pth')
net.eval()

mydataset=MyDataSet('./data',training=False)
mydataloader=MyDataLoader(mydataset,BATCH_SIZE,NUM_WORKERS)

cnt=0
for it,(batch_image,batch_label) in enumerate(mydataloader):
    print('Testing iteration:{0:d}/{1:d}.'.format(it,10000//BATCH_SIZE))
    batch_image=batch_image.to(device).float()
    batch_label=batch_label.to(device).long()
    output=net(batch_image)
    for b in range(BATCH_SIZE):
        v=output[b][0]
        maxid=0
        for i in range(10):
            if output[b][i]>v:
                v=output[b][i]
                maxid=i
        if maxid==batch_label[b]:
            cnt+=1
print('Test Accur: {0:.5f}'.format(cnt/sample_number))
print('Test Completed.')