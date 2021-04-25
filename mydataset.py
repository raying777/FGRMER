
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 18:37:19 2019

@author: 
"""
import numpy as np
import torch
#from skimage import io
#from skimage import transform
#import matplotlib.pyplot as plt
#import os
#import torch
#import torchvision
from torch.utils.data import Dataset
#from torchvision.transforms import transforms
#from PIL import Image
import scipy.io as scio

'''class MyData(Dataset):

    def __init__(self, mat_root, data_root, transform):
        self.transform = transform 
        self.mat_root = mat_root
        self.data_root = data_root
        datas = []
        labels = []             
        mat = scio.loadmat(mat_root)
        datamat = mat['data_features_5000_gai'] 
        with open(data_root,'r') as f:
            lines = f.readlines()

        for line in lines:
            splited = line.strip().split(' ')
            data_line = splited[0]
            data = datamat[int(data_line)].reshape(1,1372)
            datas.append(data)
            
            label = splited[-1]
            labels.append(label)
        self.datas1 = np.array(datas)
        self.labels1 = np.array(labels).reshape(len(lines),1).astype(np.int64)
        
    def __getitem__(self, index):
        data, target = self.datas1[index], int(self.labels1[index])       
        
        data = torch.from_numpy(data)
        

        #sample = {'img': img, 'target': target}
        return data,target
    def __len__(self):
        return len(self.datas1)'''
    
class MyData1(Dataset):

    def __init__(self, mat_root, data_root, transform):
        self.transform = transform 
        self.mat_root = mat_root
        self.data_root = data_root
        datas = []
        labels = []             
        mat = scio.loadmat(mat_root)
        datamat = mat['feature_9000x1470_shape'] 
        with open(data_root,'r') as f:
            lines = f.readlines()

        for line in lines:
            splited = line.strip().split(' ')
            data_line = splited[0]
            data = datamat[int(data_line)].reshape(30,7,7)
            datas.append(data)
            
            label = splited[-1]
            labels.append(label)
        self.datas1 = np.array(datas)
        self.labels1 = np.array(labels).reshape(len(lines),1).astype(np.int64)
        
    def __getitem__(self, index):
        data, target = self.datas1[index], int(self.labels1[index])       
        
        data = torch.from_numpy(data)
        

        #sample = {'img': img, 'target': target}
        return data,target
    def __len__(self):
        return len(self.datas1)
