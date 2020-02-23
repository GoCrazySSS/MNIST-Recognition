# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 19:27:31 2020

@author: 12899
"""
from dataset import MyDataSet,MyDataLoader
from model import NumberClassifier
import torch as t
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.autograd import Variable
import os

EPOCH=50
LR=0.001
BATCH_SIZE=32
NUM_WORKERS=8
sample_number=60000 if 60000%BATCH_SIZE==0 else 60000-(60000%BATCH_SIZE)
device=t.device('cuda:0' if t.cuda.is_available() else 'cpu')
net=NumberClassifier().to(device)
optimizer=optim.Adam(net.parameters(),lr=LR)
loss_func=nn.CrossEntropyLoss().to(device)

mydataset=MyDataSet('./data')
mydataloader=MyDataLoader(mydataset,BATCH_SIZE,NUM_WORKERS)
last_ep_accur=0.
for ep in range(EPOCH):
    cnt=0
    for it,(batch_image,batch_label) in enumerate(mydataloader):
#        if ep==0 and it==0:
        batch_image=batch_image.to(device).float()
        batch_label=batch_label.to(device).long()
        output=net(batch_image)
#            print(output)
        loss=loss_func(output,batch_label)
        print(loss.__float__(),end=', ')
        for b in range(BATCH_SIZE):
            v=output[b][0]
            maxid=0
            for i in range(10):
                if output[b][i]>v:
                    v=output[b][i]
                    maxid=i
            if maxid==batch_label[b]:
                cnt+=1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch {0:d}, Accur: {1:.5f}'.format(ep,last_ep_accur))
        model_dir='./output'
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
    t.save(net,model_dir+'/model_ep{0:02d}.pth'.format(ep))
    last_ep_accur=cnt/sample_number