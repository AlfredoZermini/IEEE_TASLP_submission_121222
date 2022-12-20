import numpy as np
import os
import time
import pickle
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import h5py

# hyperparameters
batch_size = 10

# neighbour parameters
frame_neigh = 1
n_frames = frame_neigh*2+1


# custom cost functions
def cost(y_true, y_pred):

   #return K.mean(K.sigmoid())  (K.square(y_true-y_pred)) )
   return K.mean(K.square(y_true-y_pred))


# creates spectrograms with neighbour frames
def neighbour1(X, frame_neigh):

   [n_samples1,n_freq,n_time,n_features] = X.shape

   # define tensors
   X_neigh = np.zeros([n_samples1*n_time, n_freq, n_frames, n_features ])
   X_zeros = np.zeros([n_samples1, n_freq, n_time+frame_neigh*2, n_features ])

   print(X_neigh.shape)
   

   # create X_zeros, who is basically X with an empty border of 2*frame_neigh frames
   X_zeros[:,:,frame_neigh:X_zeros.shape[2]-frame_neigh,:] = X

   for sample in range(0,n_samples1 ):
      for frame in range(0,n_time ):
         X_neigh[sample*n_time+frame, :, :, :] = X_zeros[sample, :, frame:frame+n_frames, :]

   return X_neigh


def test_NN(Xtest, load_NN_name,features):

   # define parameters
   [n_samples, n_freq, n_time, n_features] = Xtest.shape

   # include neighbour frames
   X_in = neighbour1(Xtest,frame_neigh)

   del Xtest

   # load CNN model
   model = load_model(load_NN_name, custom_objects={'cost': cost})
   model.summary()

   # fill output_tensor with input data from the audio mixtures
   output = model.predict(X_in, batch_size = batch_size, verbose=1)
   del model

   # define mask tensor
   masks = np.zeros([n_samples,n_freq,n_time,len(output)])

   # fill masks
   for sample in range(0, n_samples):
      for source in range(0,len(output)):
         masks[sample,:,0:n_time,source] = output[source][sample*n_time:sample*n_time+n_time,:].T


   return masks