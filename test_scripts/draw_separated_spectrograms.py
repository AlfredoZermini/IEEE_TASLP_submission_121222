import numpy as np
import os
import time
import sys
from scipy import signal
from pylab import *
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.stats.stats import pearsonr
import pickle
from train_utilities import expand_source


angles_list = [ [0,10,20,30], [0,20,40,60], [0,30,60,90], [0,40,80,120], [0,50,100,150],  [0,60,120,180], [0,70,140,210], [0,80,160,240], [0,90,180,270]  ]
n_doas = 9

def draw_separated_spectrograms(or0L, or0R, sep0L, sep0R, or1L, or1R, sep1L, sep1R, or2L, or2R, sep2L, sep2R, X_left, X_right, masks, sound_path):

    im=0
    mask_target0 = masks[:,:,:,0]
    mask_target1 = masks[:,:,:,1]
    mask_target2 = masks[:,:,:,2]

    index = 0
    for angle in angles_list:

        #print angle
        #image = index*len(or0L)/n_doas
        #print image


	#print sep0L[image,:,:].shape

        for image in range(im,im+10):
            path = os.path.join(sound_path, str(angle[0]) + 'deg_' + str(angle[1]) + 'deg_' + str(angle[2]) + 'deg' )

            ### LEFT
            fig = plt.figure(figsize=(20,20))
            fig.suptitle('Original vs separated targets - left: angle=' + str(angle[0]) + 'deg_' + str(angle[1]) + 'deg_' + str(angle[2]) + 'deg')
            plt.title('Original vs separated targets - left: angle=' + str(angle[0]) + 'deg_' + str(angle[1]) + 'deg_' + str(angle[2]) + 'deg', fontsize=30)

            plt.subplot(3, 4, 1, aspect='equal')
            plt.ylabel('Frequency', fontsize=10)
            title('Separated mask: source 0', fontsize=13)
            plt.imshow(mask_target0[image,:,:], cmap=plt.cm.spectral, aspect='auto', interpolation='none')
            plt.gca().invert_yaxis()

            plt.subplot(3, 4, 2, aspect='equal')
            title('Original spectrogram', fontsize=13)
            plt.imshow(np.abs(X_left[image,:,:]), cmap=plt.cm.spectral, aspect='auto', interpolation='none')
            plt.gca().invert_yaxis()

            plt.subplot(3, 4, 3, aspect='equal')
            title('Separated target: source 0', fontsize=13)
            plt.imshow(np.abs(sep0L[image,:,:]), cmap=plt.cm.spectral, aspect='auto', interpolation='none')
            plt.gca().invert_yaxis()

            plt.subplot(3, 4, 4, aspect='equal')
            title('Original target: source 0', fontsize=13)
            plt.imshow(np.abs(or0L[image,:,:]), cmap=plt.cm.spectral, aspect='auto', interpolation='none')
            plt.gca().invert_yaxis()

            plt.subplot(3, 4, 5, aspect='equal')
            plt.ylabel('Frequency', fontsize=10)
            title('Separated mask: source 1', fontsize=13)
            plt.imshow(mask_target1[image,:,:], cmap=plt.cm.spectral, aspect='auto', interpolation='none')
            plt.gca().invert_yaxis()

            plt.subplot(3, 4, 6, aspect='equal')
            title('Original spectrogram', fontsize=13)
            plt.imshow(np.abs(X_left[image,:,:]), cmap=plt.cm.spectral, aspect='auto', interpolation='none')
            plt.gca().invert_yaxis()

            plt.subplot(3, 4, 7, aspect='equal')
            title('Separated target: source 1', fontsize=13)
            plt.imshow(np.abs(sep1L[image,:,:]), cmap=plt.cm.spectral, aspect='auto', interpolation='none')
            plt.gca().invert_yaxis()

            plt.subplot(3, 4, 8, aspect='equal')
            title('Original target: source 1', fontsize=13)
            plt.imshow(np.abs(or1L[image,:,:]), cmap=plt.cm.spectral, aspect='auto', interpolation='none')
            plt.gca().invert_yaxis()

            plt.subplot(3, 4, 9, aspect='equal')
            plt.ylabel('Frequency', fontsize=10)
            title('Separated mask: source 2', fontsize=13)
            plt.imshow(mask_target2[image,:,:], cmap=plt.cm.spectral, aspect='auto', interpolation='none')
            plt.gca().invert_yaxis()

            plt.subplot(3, 4, 10, aspect='equal')
            title('Original spectrogram', fontsize=13)
            plt.imshow(np.abs(X_left[image,:,:]), cmap=plt.cm.spectral, aspect='auto', interpolation='none')
            plt.gca().invert_yaxis()

            plt.subplot(3, 4, 11, aspect='equal')
            title('Separated target: source 2', fontsize=13)
            plt.imshow(np.abs(sep2L[image,:,:]), cmap=plt.cm.spectral, aspect='auto', interpolation='none')
            plt.gca().invert_yaxis()

            plt.subplot(3, 4, 12, aspect='equal')
            title('Original target: source 2', fontsize=13)
            plt.imshow(np.abs(or2L[image,:,:]), cmap=plt.cm.spectral, aspect='auto', interpolation='none')
            plt.gca().invert_yaxis()

            plt.savefig(os.path.join(path, 'comparison_left_'+ str(angle[0]) + 'deg_' + str(angle[1]) + 'deg_' + str(angle[2]) + 'deg' + '_' + str(image)))
        im = im+10

    '''
	### RIGHT
	fig = plt.figure(figsize=(20,20))
        fig.suptitle('Original vs separated targets - right: angle=' + str(angle[0]) + 'deg_' + str(angle[1]) + 'deg_' + str(angle[2]) + 'deg')
        plt.title('Original vs separated targets - right: angle=' + str(angle[0]) + 'deg_' + str(angle[1]) + 'deg_' + str(angle[2]) + 'deg', fontsize=30)

        plt.subplot(3, 4, 1, aspect='equal')
        plt.ylabel('Frequency', fontsize=10)
        title('Separated mask: source 0', fontsize=13)
        plt.imshow(mask_target0[image,:,:], cmap=plt.cm.spectral, aspect='auto', interpolation='none')
        plt.gca().invert_yaxis()

        plt.subplot(3, 4, 2, aspect='equal')
        title('Original spectrogram', fontsize=13)
        plt.imshow(np.abs(X_right[image,:,:]), cmap=plt.cm.spectral, aspect='auto', interpolation='none')
        plt.gca().invert_yaxis()

        plt.subplot(3, 4, 3, aspect='equal')
        title('Separated target: source 0', fontsize=13)
        plt.imshow(np.abs(sep0R[image,:,:]), cmap=plt.cm.spectral, aspect='auto', interpolation='none')
        plt.gca().invert_yaxis()

        plt.subplot(3, 4, 4, aspect='equal')
        title('Original target: source 0', fontsize=13)
        plt.imshow(np.abs(or0R[image,:,:]), cmap=plt.cm.spectral, aspect='auto', interpolation='none')
        plt.gca().invert_yaxis()

        plt.subplot(3, 4, 5, aspect='equal')
        plt.ylabel('Frequency', fontsize=10)
        title('Separated mask: source 1', fontsize=13)
        plt.imshow(mask_target1[image,:,:], cmap=plt.cm.spectral, aspect='auto', interpolation='none')
        plt.gca().invert_yaxis()

        plt.subplot(3, 4, 6, aspect='equal')
        title('Original spectrogram', fontsize=13)
        plt.imshow(np.abs(X_right[image,:,:]), cmap=plt.cm.spectral, aspect='auto', interpolation='none')
        plt.gca().invert_yaxis()

        plt.subplot(3, 4, 7, aspect='equal')
        title('Separated target: source 1', fontsize=13)
        plt.imshow(np.abs(sep1R[image,:,:]), cmap=plt.cm.spectral, aspect='auto', interpolation='none')
        plt.gca().invert_yaxis()

        plt.subplot(3, 4, 8, aspect='equal')
        title('Original target: source 1', fontsize=13)
        plt.imshow(np.abs(or1R[image,:,:]), cmap=plt.cm.spectral, aspect='auto', interpolation='none')
        plt.gca().invert_yaxis()

        plt.subplot(3, 4, 9, aspect='equal')
        plt.ylabel('Frequency', fontsize=10)
        title('Separated mask: source 2', fontsize=13)
        plt.imshow(mask_target2[image,:,:], cmap=plt.cm.spectral, aspect='auto', interpolation='none')
        plt.gca().invert_yaxis()

        plt.subplot(3, 4, 10, aspect='equal')
        title('Original spectrogram', fontsize=13)
        plt.imshow(np.abs(X_right[image,:,:]), cmap=plt.cm.spectral, aspect='auto', interpolation='none')
        plt.gca().invert_yaxis()

        plt.subplot(3, 4, 11, aspect='equal')
        title('Separated target: source 2', fontsize=13)
        plt.imshow(np.abs(sep2R[image,:,:]), cmap=plt.cm.spectral, aspect='auto', interpolation='none')
        plt.gca().invert_yaxis()

        plt.subplot(3, 4, 12, aspect='equal')
        title('Original target: source 2', fontsize=13)
        plt.imshow(np.abs(or2R[image,:,:]), cmap=plt.cm.spectral, aspect='auto', interpolation='none')
        plt.gca().invert_yaxis()

        plt.savefig(os.path.join(path, 'comparison_right_'+ str(angle[0]) + 'deg_' + str(angle[1]) + 'deg_' + str(angle[2]) + 'deg'))


        index += 1
        '''

