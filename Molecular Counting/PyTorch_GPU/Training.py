#!/usr/bin/env python
import numpy as np
from collections import defaultdict
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.spatial import distance
import re, sys, os
from torchvision import datasets
import h5py
import numpy

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as data_utils
from torch.optim.lr_scheduler import ReduceLROnPlateau

from CNN_Model import project_01, normalize_im, NeuralNet, L1L2loss, weight_init, Checkpoint
from DataLoader import Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")         
print(device)   

# For reproducibility
np.random.seed(123)

# Load training & test data as training and validation sets
h5Train = h5py.File('TrainingSplit2.h5', 'r')
h5Test = h5py.File('TestSplit2.h5', 'r')

partition = {'train': list(h5Train.keys()), 'test': list(h5Test.keys())}

P = dict()
P['numEpoch'] = 350
P['learning_rate'] = 0.001
P['batchSize'] = 16

# Generators
training_set = Dataset(h5Train,P['batchSize'],partition['train'])
train_loader = DataLoader(training_set, shuffle=True)

validation_set = Dataset(h5Test,P['batchSize'],partition['test'])
test_loader = DataLoader(validation_set, shuffle=False)

myModel = NeuralNet()

if torch.cuda.device_count() > 1:                                               
    print("Let's use", torch.cuda.device_count(), "GPUs!")                      
    myModel = nn.DataParallel(myModel)                                              
                                                                                
myModel.to(device) 

psize = 208
optimizer = optim.Adam(myModel.parameters(), lr=P['learning_rate'])
criterion = L1L2loss((1, psize, psize))

myModel.apply(weight_init)
scheduler = ReduceLROnPlateau(optimizer, verbose=True,factor=0.1, patience=5, min_lr=0.00005, threshold=0, eps=0)

# Define the loss history recorder
df = pd.DataFrame(columns=('epoch', 'loss', 'loss_test'))
Weight_file = os.path.abspath(os.path.normpath(os.path.join(os.getcwd(),'Checkpointing/Checkpointer.h5')))
meanstd_name = os.path.abspath(os.path.normpath(os.path.join(os.getcwd(),'meanstd_name.mat')))
path = os.path.abspath(os.path.normpath(os.path.join(os.getcwd(),'output.csv')))

loss_list = {'train':[np.inf],'test':[np.inf]}
loss_c = {'train':np.inf,'test':np.inf}
data_loader = {'train':train_loader,'test':test_loader}

for epoch in range(P['numEpoch']):
    for phase in ['train', 'test']:
        if phase == 'train':
            myModel.train()
        else:
            myModel.eval()

        running_loss = 0.0
        counter = 0
        for i, (images, labels) in enumerate(data_loader[phase]):
            # Run the forward pass
            counter = counter+1
            images = images.reshape(P['batchSize'], 1, psize, psize)
            labels = labels.reshape(P['batchSize'], 1, psize, psize) 
            
            images = images.to(device)
            labels = labels.to(device)
            
            if phase == 'train':
                optimizer.zero_grad()
            
            density_pred = myModel(images)
            loss = criterion(labels,density_pred)

            if phase == 'train':
                loss.backward()
                optimizer.step()

            running_loss += loss.cpu().detach().numpy()
    
            if phase=='train' and counter==400:
                break
        
        running_loss *= P['batchSize']/counter
        if phase=='test':                                                       
            scheduler.step(running_loss)  

        # Save the model weights after each epoch if the validation loss decreased
        if phase=='test':
            is_best = bool(running_loss < min(loss_list['test']))
            Checkpoint(myModel, optimizer, epoch, is_best, Weight_file)

        loss_list[phase].append(running_loss)
        loss_c[phase] = running_loss

    # B.3) stat for this epoch
    print(str(epoch)+'    loss:'+str(loss_c['train'])+' '+str(loss_c['test']))
    # save
    df.loc[epoch] = [epoch, loss_c['train'], loss_c['test']]
    df.to_csv(path)
