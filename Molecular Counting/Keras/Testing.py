# Import Libraries and model
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import argparse
from CNN_Model import buildModel, project_01, normalize_im
import scipy.io as sio
import time
import os
from os.path import abspath


"""
This function tests a trained model on the desired test set, given the
tiff stack of test images, learned weights, and normalization factors.

# Inputs
datafile          - the tiff stack of test images
weights_file      - the saved weights file generated in train_model
meanstd_file      - the saved mean and standard deviation file generated in train_model
savename          - the filename for saving the recovered SR image
upsampling_factor - the upsampling factor for reconstruction (default 8)
debug             - boolean whether to save individual frame predictions (default 0)

# Outputs
function saves a mat file with the recovered image, and optionally saves
individual frame predictions in case debug=1. (default is debug=0)
"""

datafile = os.path.abspath(os.path.normpath(os.path.join(os.getcwd(),'ArtificialDataset.tif')))
weights_file = os.path.abspath(os.path.normpath(os.path.join(os.getcwd(),'Checkpointing/Checkpointer.h5')))
meanstd_file = os.path.abspath(os.path.normpath(os.path.join(os.getcwd(),'meanstd_name.mat')))
savename = os.path.abspath(os.path.normpath(os.path.join(os.getcwd(),'recovery.mat')))

upsampling_factor = 8
# load the tiff data
Images = io.imread(datafile)

# get dataset dimensions
(K, M, N) = Images.shape

# upsampling using a simple nearest neighbor interp.
Images_upsampled = np.zeros((K, M*upsampling_factor, N*upsampling_factor))
for i in range(Images.shape[0]):
    Images_upsampled[i,:,:] = np.kron(Images[i,:,:], np.ones((upsampling_factor,upsampling_factor)))

Images = Images_upsampled

# upsampled frames dimensions
(K, M, N) = Images.shape
            
# Build the model for a bigger image
model = buildModel((M, N, 1))

# Load the trained weights
model.load_weights(weights_file)

# load mean and std
matfile = sio.loadmat(meanstd_file)
test_mean = np.array(matfile['mean_test'])
test_std = np.array(matfile['std_test'])

# Setting type
Images = Images.astype('float32')

# Normalize each sample by it's own mean and std
Images_norm = np.zeros(Images.shape,dtype=np.float32)
for i in range(Images.shape[0]):
    Images_norm[i,:,:] = project_01(Images[i,:,:])
    Images_norm[i,:,:] = normalize_im(Images_norm[i,:,:], test_mean, test_std)

# Reshaping
Images_norm = np.expand_dims(Images_norm,axis=3)
                        
# Make a prediction and time it
start = time.time()
predicted_density = model.predict(Images_norm, batch_size=1)
end = time.time()
print(end - start)
                            
# threshold negative values
predicted_density[predicted_density < 0] = 0

# resulting sum images
WideField = np.squeeze(np.sum(Images_norm, axis=0))
Recovery = np.squeeze(np.sum(predicted_density, axis=0))

# Save predictions to a matfile to open later in matlab
mdict = {"Recovery": Recovery}
sio.savemat(savename, mdict)

# save predicted density in each frame for debugging purposes
mdict = {"Predictions": predicted_density}
sio.savemat(savename + '_predictions.mat', mdict)
                                
# Look at the sum image
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True)
ax1.imshow(WideField)
ax1.set_title('Wide Field')
ax2.imshow(Recovery)
ax2.set_title('Sum of Predictions')
f.subplots_adjust(hspace=0)
f.savefig("output.pdf", bbox_inches='tight')


                                    
                                        

