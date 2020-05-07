#!/usr/bin/env python 
import numpy as np
from collections import defaultdict
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.spatial import distance
import re, sys, os
from torchvision import datasets
from IPython.display import Image
import h5py
import numpy

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as data_utils
from torch.optim.lr_scheduler import ReduceLROnPlateau
from CNN_Model import project_01, normalize_im, NeuralNet, L1L2loss, weight_init, Checkpoint
import gc 
  
collected = gc.collect() 

# For reproducibility
np.random.seed(123)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Load training data and divide it to training and validation sets
matfile = h5py.File('TrainingSet.mat', 'r')
patches = np.array(matfile['patches'])
heatmaps = 100.0*np.array(matfile['heatmaps'])   # Why multiply by 100

X_train, X_test, y_train, y_test = train_test_split(patches, heatmaps, test_size=0.3, random_state=42)
print('Number of Training Examples: %d' % X_train.shape[0])
print('Number of Validation Examples: %d' % X_test.shape[0])

#===================== Training set normalization ==========================
# normalize training images to be in the range [0,1] and calculate the
# training set mean and std
mean_train = np.zeros(X_train.shape[0],dtype=np.float32)
std_train = np.zeros(X_train.shape[0], dtype=np.float32)
for i in range(X_train.shape[0]):
    X_train[i, :, :] = project_01(X_train[i, :, :])
    mean_train[i] = X_train[i, :, :].mean()
    std_train[i] = X_train[i, :, :].std()

# resulting normalized training images
mean_val_train = mean_train.mean()
std_val_train = std_train.mean()
X_train_norm = np.zeros(X_train.shape, dtype=np.float32)
for i in range(X_train.shape[0]):
    X_train_norm[i, :, :] = normalize_im(X_train[i, :, :], mean_val_train, std_val_train)

# patch size (Input Dim)
psize =  X_train_norm.shape[1]

# ===================== Test set normalization ==========================
# normalize test images to be in the range [0,1] and calculate the test set
# mean and std

mean_test = np.zeros(X_test.shape[0],dtype=np.float32)
std_test = np.zeros(X_test.shape[0], dtype=np.float32)
for i in range(X_test.shape[0]):
    X_test[i, :, :] = project_01(X_test[i, :, :])
    mean_test[i] = X_test[i, :, :].mean()
    std_test[i] = X_test[i, :, :].std()

# resulting normalized test images
mean_val_test = mean_test.mean()
std_val_test = std_test.mean()
X_test_norm = np.zeros(X_test.shape, dtype=np.float32)
for i in range(X_test.shape[0]):
    X_test_norm[i, :, :] = normalize_im(X_test[i, :, :], mean_val_test, std_val_test)

P = dict()
P['numEpoch'] = 100
P['learning_rate'] = 0.001
P['batchSize'] = 16

X_train_norm = torch.from_numpy(X_train_norm.astype('float32')).to(device)
X_test_norm = torch.from_numpy(X_test_norm.astype('float32')).to(device)
y_train = torch.from_numpy(y_train.astype('float32')).to(device)
y_test = torch.from_numpy(y_test.astype('float32')).to(device)

# Reshaping: nn.Conv2d expects an input of the shape [batch_size, channels, height, width].
X_train_norm = X_train_norm.view(X_train.shape[0], 1, psize, psize)
X_test_norm = X_test_norm.view(X_test.shape[0], 1, psize, psize)
y_train = y_train.view(y_train.shape[0], 1, psize, psize)
y_test = y_test.view(y_test.shape[0], 1, psize, psize)

train = data_utils.TensorDataset(X_train_norm, y_train)
test = data_utils.TensorDataset(X_test_norm, y_test)
train_loader = DataLoader(train,shuffle=True,batch_size=P['batchSize'],num_workers=8)
test_loader = DataLoader(test,shuffle=False,batch_size=P['batchSize'],num_workers=8)

N_train = len(train)
N_test = len(test)

myModel = NeuralNet()

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

myModel.to(device)

optimizer = optim.Adam(myModel.parameters(), eps=1e-07, lr=P['learning_rate'])
criterion = L1L2loss((1, psize, psize))

# Weight Initialization
#myModel.apply(weight_init)
scheduler = ReduceLROnPlateau(optimizer, verbose=True,factor=0.1, patience=5, min_lr=0.00005, threshold=0, eps=0)

df = pd.DataFrame(columns=('epoch', 'loss', 'loss_test'))
Weight_file = os.path.abspath(os.path.normpath(os.path.join(os.getcwd(),'Checkpointing/Checkpointer.h5')))
meanstd_name = os.path.abspath(os.path.normpath(os.path.join(os.getcwd(),'meanstd_name.mat')))
path = os.path.abspath(os.path.normpath(os.path.join(os.getcwd(),'output.csv')))

len_dataset = {'train':N_train,'test':N_test}
loss_list = {'train':[np.inf],'test':[np.inf]}
loss_c = {'train':np.inf,'test':np.inf}
data_loader = {'train':train_loader,'test':test_loader}

for epoch in range(P['numEpoch']): 
    phase = 'train'
    myModel.train()
    running_loss = 0.0
    counter = 0
    for i, (images, labels) in enumerate(data_loader[phase]):
        # Run the forward pass
        counter = counter+1
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        density_pred = myModel(images)
        loss = criterion(labels,density_pred)
        
        loss.backward()
        optimizer.step() 

        running_loss += loss.detach().numpy()
        if counter==400:                                                        
            break  

    running_loss *= P['batchSize']/6400 #N_train 
    loss_list[phase].append(running_loss) 
    loss_c[phase] = running_loss

    phase = 'test'
    myModel.eval()
    running_loss = 0.0                                                          
    for i, (images, labels) in enumerate(data_loader[phase]):                   
        density_pred = myModel(images) 
        loss = criterion(labels,density_pred)                                                                                                                                     
        running_loss += loss.detach().numpy()                                   

    running_loss *= P['batchSize']/N_test                                 
    loss_list[phase].append(running_loss) 

    # Save the model weights after each epoch if the validation loss decreased
    loss_c[phase] = running_loss
#    scheduler.step(running_loss)
    
    is_best = bool(loss_c['test'] < min(loss_list['test']))    
    Checkpoint(myModel, optimizer, epoch, is_best, Weight_file) 
    
    loss_list[phase].append(running_loss) 

    # B.3) stat for this epoch
    print(str(epoch)+'    loss:'+str(loss_c['train'])+' '+str(loss_c['test']))
    # save
    df.loc[epoch] = [epoch, loss_c['train'], loss_c['test']]
    df.to_csv(path)  

mdict = {"mean_test": mean_val_test, "std_test": std_val_test}                  
sio.savemat(meanstd_name, mdict)  

