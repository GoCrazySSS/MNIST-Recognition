# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 19:49:20 2020

@author: 12899
"""
from torch.utils.data import Dataset,DataLoader
import numpy as np
import torch as t
import struct
import matplotlib.pyplot as plt

class MyDataSet(Dataset):
    def __init__(self,dataroot,training=True):
        super(MyDataSet,self).__init__()
        imagepath=dataroot+'/train-images.idx3-ubyte' if training else dataroot+'/t10k-images.idx3-ubyte'
        labelpath=dataroot+'/train-labels.idx1-ubyte' if training else dataroot+'/t10k-labels.idx1-ubyte'
        image_reader=open(imagepath,'rb')
        label_reader=open(labelpath,'rb')
        image_magic=struct.unpack('>I',image_reader.read(4))
        label_magic=struct.unpack('>I',label_reader.read(4))
        image_number=struct.unpack('>I',image_reader.read(4))[0]
        label_number=struct.unpack('>I',label_reader.read(4))[0]
        
        image_width=struct.unpack('>I',image_reader.read(4))
        image_hight=struct.unpack('>I',image_reader.read(4))
        
        self.len=image_number
#        print(image_number)
        self.image_data=np.empty((image_number,1,28,28),dtype=np.ubyte)
        self.label_data=np.empty(label_number,dtype=np.ubyte)
        for i in range(image_number):
            image=np.array(struct.unpack('>784B',image_reader.read(784)),dtype=np.ubyte)
            label=struct.unpack('>B',label_reader.read(1))[0]
            self.image_data[i]=image.reshape(1,28,28)
            self.label_data[i]=label
        label_reader.close()
        image_reader.close()
    def __len__(self):
        return self.len
    def __getitem__(self,index):
        return self.image_data[index]/255.,self.label_data[index]

class MyDataLoader(DataLoader):
    def __init__(self,mydataset,b_size,num_worker):
        super(MyDataLoader,self).__init__(mydataset,shuffle=True,batch_size=b_size,drop_last=True,num_workers=num_worker)

#mydataset=MyDataSet('./data')
#mydataloader=MyDataLoader(mydataset)
#for it,(image,label) in enumerate(mydataloader):
#    print(it,label)
#i0,l0=mydata.__getitem__(0)
#plt.imshow(i0,cmap='gray')
#print(l0)
