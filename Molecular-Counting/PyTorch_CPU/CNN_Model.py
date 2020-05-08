#!/usr/bin/env python
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import interpolate

def project_01(im):
    im = np.squeeze(im)
    min_val = im.min()
    max_val = im.max()
    return (im - min_val)/(max_val - min_val)

# normalize image given mean and std
def normalize_im(im, dmean, dstd):
    im = np.squeeze(im)
    im_norm = np.zeros(im.shape,dtype=np.float32)
    im_norm = (im - dmean)/dstd
    return im_norm
    
#  Define a matlab like gaussian 2D filter
def matlab_style_gauss2D(shape=(7,7),sigma=1):
    """ 
    2D gaussian filter - should give the same result as:
    MATLAB's fspecial('gaussian',[shape],[sigma]) 
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h.astype(dtype=np.float32)
    h[ h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    h = h*2.0
    h = h.astype('float32')
    return h

# Expand the filter dimensions
psf_heatmap = matlab_style_gauss2D(shape=(7,7),sigma=1)
gfilter = torch.from_numpy(np.reshape(psf_heatmap, [1, 1, 7, 7]))

# Combined MSE + L1 loss
def L1L2loss(input_shape):
    def bump_mse(heatmap_true, spikes_pred):
        # generate the heatmap corresponding to the predicted spikes
        heatmap_pred = F.conv2d(spikes_pred, gfilter, stride=1, padding=3)
        # heatmaps MSE        
        loss = nn.MSELoss()
        loss_heatmaps = loss(heatmap_true,heatmap_pred)
        
        # l1 on the predicted spikes
        l1_loss = nn.L1Loss()
        loss_spikes = l1_loss(spikes_pred,torch.zeros(input_shape))
        return loss_heatmaps + loss_spikes
    return bump_mse

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight)

def Checkpoint(model, optimizer, epoch, is_best, Weight_file):
   """Save checkpoint if a new best is achieved"""
   state = {'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()}
    
   if is_best:
       print ("=> Saving a new best")
       torch.save(state, Weight_file)
   else:
       print ("=> Validation Accuracy did not improve")
    
    
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()   
        # how do we determine tyoe of the kernel here or how is it determined?
        self.conv1 = nn.Conv2d(1,32,3,stride=(1,1),padding=(1,1),bias=False)
        self.conv2 = nn.Conv2d(32,64,3,stride=(1,1),padding=(1,1),bias=False)
        self.conv3 = nn.Conv2d(64,128,3,stride=(1,1),padding=(1,1),bias=False)
        self.conv4 = nn.Conv2d(128,512,3,stride=(1,1),padding=(1,1),bias=False)
        self.conv5 = nn.Conv2d(512,128,3,stride=(1,1),padding=(1,1),bias=False)
        self.conv6 = nn.Conv2d(128,64,3,stride=(1,1),padding=(1,1),bias=False) 
        self.conv7 = nn.Conv2d(64,32,3,stride=(1,1),padding=(1,1),bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(2)
        self.conv8 = nn.Conv2d(32,1,1,stride=(1,1),padding=(0,0),bias=False)

    def forward(self, x):
        z1 = F.relu(self.bn1(self.conv1(x)))
        pool1 = self.pool(z1)
        z2 = F.relu(self.bn2(self.conv2(pool1)))
        pool2 = self.pool(z2)
        z3 = F.relu(self.bn3(self.conv3(pool2)))
        pool3 = self.pool(z3)
        z4 = F.relu(self.bn4(self.conv4(pool3)))
        up5 = interpolate(z4, scale_factor=2)
        z5 = F.relu(self.bn3(self.conv5(up5)))
        up6 = interpolate(z5, scale_factor=2)
        z6 = F.relu(self.bn2(self.conv6(up6)))
        up7 = interpolate(z6, scale_factor=2)
        z7 = F.relu(self.bn1(self.conv7(up7)))
        z8 = self.conv8(z7)
        return z8  
