#!/usr/bin/env python
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import interpolate
from torch.utils import data

# def split_keys(L):
#     random.shuffle(L,cut)
#     test_key = L[:cut]
#     train_key = L[cut:]
#     return test_key, train_key 

class Dataset(data.Dataset):
    def __init__(self, file, batch_size, list_IDs):
        'Initialization'
        self.f = file
        self.bSize = batch_size
        self.list_IDs = list_IDs
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        # Order for the 1st dimension: ['patches','heatmaps']
        ID = self.list_IDs[index]
        dset = self.f.require_dataset(ID,(2,self.bSize,1,208,208),dtype='f')    
        patches = np.array(dset[0,:,:,:,:])
        heatmaps = np.array(dset[1,:,:,:,:])  

        X_norm = torch.from_numpy(patches)
        y_norm = torch.from_numpy(heatmaps)
        
        ## patch size (Input Dim)
        psize = X_norm.shape[1]
        
        # Reshaping: nn.Conv2d expects an input of the shape [batch_size, channels, height, width].
        X_norm = X_norm.view(X_norm.shape[0], 1, psize, psize)
        y_norm = y_norm.view(y_norm.shape[0], 1, psize, psize)

        return X_norm, y_norm

