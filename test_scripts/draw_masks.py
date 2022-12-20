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
sys.path.append("..")


angles_list = [ [0,10,20,30], [0,20,40,60], [0,30,60,90], [0,40,80,120], [0,50,100,150],  [0,60,120,180], [0,70,140,210], [0,80,160,240], [0,90,180,270]  ]
n_doas = 9

def draw_masks(masks, IRM_tensor, sound_path):

    # define masks
    mask_target0 = masks[:,:,:,0]
    mask_target1 = masks[:,:,:,1]
    mask_target2 = masks[:,:,:,2]
    IRM0 = IRM_tensor[:,0,:,:]
    IRM1 = IRM_tensor[:,1,:,:]
    IRM2 = IRM_tensor[:,2,:,:]

    print(sound_path)

    index = 0
    for angle in angles_list:

        print(angle)
        image = index*len(IRM0)/n_doas
        print(image)

        path = os.path.join(sound_path, str(angle[0]) + 'deg_' + str(angle[1]) + 'deg_' + str(angle[2]) + 'deg'  )

        fig = plt.figure(figsize=(10,10))
        fig.suptitle('Estimated masks vs IRMs: angle=' + str(angle[0]) + 'deg_' + str(angle[1]) + 'deg_' + str(angle[2]) + 'deg')
        plt.title('Estimated masks vs IRMs: angle=' + str(angle[0]) + 'deg_' + str(angle[1]) + 'deg_' + str(angle[2]) + 'deg', fontsize=40)

        plt.subplot(3, 2, 1)
        plt.ylabel('Frequency', fontsize=10)
        title('Estimated mask: source 0', fontsize=13)
        plt.imshow(mask_target0[image,:,:], cmap=plt.cm.spectral, aspect='auto', interpolation='none')
        plt.gca().invert_yaxis()

        plt.subplot(3, 2, 2)
        title('IRM: source 0', fontsize=13)
        plt.imshow(IRM0[image,:,:], cmap=plt.cm.spectral, aspect='auto', interpolation='none')
        plt.gca().invert_yaxis()

        plt.subplot(3, 2, 3)
        plt.ylabel('Frequency', fontsize=10)
        title('Estimated mask: source 1', fontsize=13)
        plt.imshow(mask_target1[image,:,:], cmap=plt.cm.spectral, aspect='auto', interpolation='none')
        plt.gca().invert_yaxis()

        plt.subplot(3, 2, 4)
        title('IRM: source 1', fontsize=13)
        plt.imshow(IRM1[image,:,:], cmap=plt.cm.spectral, aspect='auto', interpolation='none')
        plt.gca().invert_yaxis()

        plt.subplot(3, 2, 5)
        title('Estimated mask: source 2', fontsize=13)
        plt.xlabel('Time', fontsize=10)
        plt.ylabel('Frequency', fontsize=10)
        plt.imshow(mask_target2[image,:,:], cmap=plt.cm.spectral, aspect='auto', interpolation='none')
        plt.gca().invert_yaxis()

        plt.subplot(3, 2, 6)
        title('IRM: source 2', fontsize=13)
        plt.xlabel('Time', fontsize=10)
        plt.ylabel('Frequency', fontsize=10)
        plt.imshow(IRM2[image,:,:], cmap=plt.cm.spectral, aspect='auto', interpolation='none')
        plt.gca().invert_yaxis()

        plt.savefig(os.path.join(path, 'masks_'+ str(angle[0]) + 'deg_' + str(angle[1]) + 'deg_' + str(angle[2]) + 'deg'))

        index += 1

def draw_masks4(masks, IRM_tensor, figures_path):

    title_size = 41 #25
    xy_size = 38 #20
    tick_size = 31 #17
    
    # define masks
    mask_target0 = masks[:,:,:,0]
    mask_target1 = masks[:,:,:,1]
    mask_target2 = masks[:,:,:,2]
    mask_target3 = masks[:,:,:,3]

    # define IRMs
    IRM0 = IRM_tensor[:,:,:,0]
    IRM1 = IRM_tensor[:,:,:,1]
    IRM2 = IRM_tensor[:,:,:,2]
    IRM3 = IRM_tensor[:,:,:,3]

    index = 0
    for angle in angles_list:
        
        print(angle)
        image = index*int(len(IRM0)/n_doas)
        print(image)

        #path = os.path.join(figures_path, str(angle[0]) + 'deg_' + str(angle[1]) + 'deg_' + str(angle[2]) + 'deg_' + str(angle[3]) + 'deg'  )

        fig = plt.figure(figsize=(22,17))
        subplots_adjust(hspace=0.2)
        fig.suptitle('Estimated masks vs IRMs: angles = ' + str(angle[0]) + r'$^\circ$, ' + str(angle[1]) + r'$^\circ$, ' + str(angle[2]) + r'$^\circ$, ' + str(angle[3]) + r'$^\circ$', fontsize=46)
        
        plt.rcParams['xtick.labelsize'] = tick_size
        plt.rcParams['ytick.labelsize'] = tick_size
        plt.title('Estimated masks vs IRMs: angles = ' + str(angle[0]) + r'$^\circ$, ' + str(angle[1]) + r'$^\circ$, ' + str(angle[2]) + r'$^\circ$, ' + str(angle[3]) + r'$^\circ$', fontsize=46)      

        plt.subplot(4, 2, 1, aspect=1, adjustable='box')
        plt.xlabel('Time', fontsize=xy_size)
        plt.ylabel('Frequency', fontsize=xy_size)
        plt.yticks(np.arange(0, 256*4+1, 256))
        plt.axis('on')
        title('Estimated mask: source 0', fontsize=title_size)
        plt.imshow(mask_target0[image,:,:], cmap='jet', aspect='auto', interpolation='none')
        plt.gca().invert_yaxis()

        plt.subplot(4, 2, 2, aspect=1, adjustable='box')
        title('IRM: source 0', fontsize=title_size)
        plt.xlabel('Time', fontsize=xy_size)
        plt.ylabel('Frequency', fontsize=xy_size)
        plt.yticks(np.arange(0, 256*4+1, 256))
        plt.imshow(IRM0[image,:,:], cmap='jet', aspect='auto', interpolation='none')
        plt.gca().invert_yaxis()

        plt.subplot(4, 2, 3, aspect=1, adjustable='box')
        plt.xlabel('Time', fontsize=xy_size)
        plt.ylabel('Frequency', fontsize=xy_size)
        plt.yticks(np.arange(0, 256*4+1, 256))
        title('Estimated mask: source 1', fontsize=title_size)
        plt.imshow(mask_target1[image,:,:], cmap='jet', aspect='auto', interpolation='none')
        plt.gca().invert_yaxis()

        plt.subplot(4, 2, 4, aspect=1, adjustable='box')
        title('IRM: source 1', fontsize=title_size)
        plt.xlabel('Time', fontsize=xy_size)
        plt.ylabel('Frequency', fontsize=xy_size)
        plt.yticks(np.arange(0, 256*4+1, 256))
        plt.imshow(IRM1[image,:,:], cmap='jet', aspect='auto', interpolation='none')
        plt.gca().invert_yaxis()

        plt.subplot(4, 2, 5, aspect=1, adjustable='box')
        title('Estimated mask: source 2', fontsize=title_size)
        plt.xlabel('Time', fontsize=xy_size)
        plt.ylabel('Frequency', fontsize=xy_size)
        plt.yticks(np.arange(0, 256*4+1, 256))
        plt.imshow(mask_target2[image,:,:], cmap='jet', aspect='auto', interpolation='none')
        plt.gca().invert_yaxis()

        plt.subplot(4, 2, 6, aspect=1, adjustable='box')
        title('IRM: source 2', fontsize=title_size)
        plt.xlabel('Time', fontsize=xy_size)
        plt.ylabel('Frequency', fontsize=xy_size)
        plt.yticks(np.arange(0, 256*4+1, 256))
        plt.imshow(IRM2[image,:,:], cmap='jet', aspect='auto', interpolation='none')
        plt.gca().invert_yaxis()

        plt.subplot(4, 2, 7, aspect=1, adjustable='box')
        title('Estimated mask: source 3', fontsize=title_size)
        plt.xlabel('Time', fontsize=xy_size)
        plt.ylabel('Frequency', fontsize=xy_size)
        plt.yticks(np.arange(0, 256*4+1, 256))
        plt.imshow(mask_target3[image,:,:], cmap='jet', aspect='auto', interpolation='none')
        plt.gca().invert_yaxis()

        plt.subplot(4, 2, 8, aspect=1, adjustable='box')
        title('IRM: source 3', fontsize=title_size)
        plt.xlabel('Time', fontsize=xy_size)
        plt.ylabel('Frequency', fontsize=xy_size)
        plt.yticks(np.arange(0, 256*4+1, 256))
        plt.imshow(IRM3[image,:,:], cmap='jet', aspect='auto', interpolation='none')
        plt.gca().invert_yaxis()

        fig.tight_layout()

        plt.savefig(os.path.join(figures_path, 'masks_'+ str(angle[0]) + 'deg_' + str(angle[1]) + 'deg_' + str(angle[2]) + 'deg_' + str(angle[3]) + 'deg'),bbox_inches='tight', dpi=300 )

        index += 1
