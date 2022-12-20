import numpy as np
import soundfile as sf
import os
import random
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt

def read_wav(filename):
   sample_rate, samples = wavfile.read(filename)
   if samples.dtype==np.dtype('int16'):
      samples = samples.astype(np.float64) / np.iinfo(np.dtype('int16')).min
   return sample_rate, samples
   
def cortex_selector(Room):
        
   if Room == 'A':
      cortex = 'CortexBRIR_0_32s_'
   elif Room == 'B':
      cortex = 'CortexBRIR_0_47s_'
   elif Room == 'C':
      cortex = 'CortexBRIR_0_68s_'
   elif Room == 'D':
      cortex = 'CortexBRIR_0_89s_'
   return cortex
    
def evaluate_length(list, min_path_pos, brirs_path, cortex, Wlength, overlap, fft_size):
   data, fs = sf.read(list[min_path_pos] )
   brir_angle = read_wav(os.path.join( brirs_path, cortex + str(0) + 'deg_16k.wav') )
   wav_l = signal.convolve(brir_angle[1][:,0],data)
   [_, _, Xl] = signal.stft(wav_l, fs, 'hann', Wlength, overlap, fft_size)
   n_frames = Xl.shape[1]
   n_bins = len(wav_l)
   
   return n_frames, n_bins

def factorial(n):
   for i in range((n-1), 0, -1):        
      n = n * i 
   return n
   
def mag(x):
   
   x = 20 * np.log10(np.abs(x))
   return x

def neighbour(X, frame_neigh):
   frame_total = frame_neigh*2+1
    
   # define tensors
   X_neigh = np.zeros([X.shape[0], X.shape[1], X.shape[2]*frame_total], dtype=np.complex64)
   X_zeros = np.zeros([X.shape[0], X.shape[1], X.shape[2]+frame_neigh*2], dtype=np.complex64)
   #Xnew = np.zeros([X.shape[0]*X.shape[2], X.shape[1], frame_total])

   # fill tensor with zeros on the edges
   X_zeros[:,:,frame_neigh:X_zeros.shape[2]-frame_neigh] = X[:,:,:]
    
   for frame in range(0,X.shape[2]):
      X_neigh[:,:,frame*frame_total:frame*frame_total+frame_total] = X_zeros[:,:,frame:frame+frame_total]

   '''
   # create new X with neighbour frames
   for sample in range(0,X.shape[0]): #n_samples
      for frame_group in range(0, X.shape[2]): #n_time
         Xnew[sample*X.shape[2]+frame_group,:,:,:] = X[sample,:,frame_group*frame_total:frame_group*frame_total+frame_total,:]
   '''
   return X_neigh

def getdim(wav,fs, win, Wlength, overlap, fft_size):
   n_bins = len(wav)
   [_, _, X] = signal.stft(wav, fs, win, Wlength, overlap, fft_size)
   n_frames = len(X.T)
   return n_bins, n_frames
    
def expand_source(original, n_softmax):

   [n_samples, n_freq, n_time] = original.shape 
   expanded = np.zeros([n_samples*n_softmax, n_freq, n_time],np.complex64)
   [n_samples_expanded,_,_] = expanded.shape 
    
   # redifine target with right dimensions
   for image in range(0, n_samples):
      for direction_set in range(0, n_softmax):
         expanded[image*n_softmax : image*n_softmax+n_softmax, :, :] = [original[image, :, :],]*n_softmax

   return expanded
 
