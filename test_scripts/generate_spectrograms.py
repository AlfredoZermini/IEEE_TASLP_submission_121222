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

# audio parameters
fs = 16000
Wlength = 2048
n_channel = 128
n_softmax = 19
window = 'hann'
window_size = Wlength#1024
hop_size = 512 #512
overlap = Wlength*3/4
fft_size = Wlength

angles_list = [ [0,10,20,30], [0,20,40,60], [0,30,60,90], [0,40,80,120], [0,50,100,150],  [0,60,120,180], [0,70,140,210], [0,80,160,240], [0,90,180,270]  ] 
n_doas = 9


# avoid clipping
MAX_WAV_VALUE = 32768.0
def waveform_norm(output_audio):
    output_audio = output_audio / MAX_WAV_VALUE
    g_norm_out = np.max(np.abs(output_audio))
    return output_audio * 0.95 / g_norm_out


# write audio
def write_audio(save_path, audio_type, angles_string, image, audio_tensor, source, channel, angles, fs):

    [_, waveform] = signal.istft(audio_tensor[image, :, :, source, channel], fs, 'hann', Wlength, overlap, fft_size)
    if channel == 0:
        channel_string = 'L'
    elif channel == 1:
        channel_string = 'R'

    # normalization
    waveform = waveform_norm(waveform)
    
    if audio_type == 'target_separated':
        name = os.path.join(save_path, 'target_separated_' + channel_string +  str(source) + '_' + str(angles[source]) + 'deg.wav')

    elif audio_type == 'spectrogram':
        name = os.path.join(save_path, 'spectrogram_' + channel_string + '_' + angles_string + '.wav')

    wavfile.write(name, fs, waveform)


# generate spectrograms from masks
def generate_spectrograms(X_or, masks, sound_path):

    [n_samples, n_freq, n_time, n_sources, _] = X_or.shape

    # add dimension
    print(masks.ndim)
    print(masks.shape)
    
    if masks.ndim == 4:
        masks = np.expand_dims(masks, axis=4)
    
    # create mixture spectrograms
    X_mix = np.expand_dims(np.sum(X_or[:, :, :, :, 1:3], axis=3), axis= 3)


    print("Getting the separated spectrograms", time.ctime())

    target_separated = np.empty([n_samples, n_freq, n_time, n_sources, 2], np.complex64)
    print(target_separated.shape, X_mix.shape, X_or.shape, masks.shape)
    
    
    # get the spectrograms from the masks
    for image in range(n_samples):
        target_separated[image, :, :, :, :] = np.multiply(X_mix[image, :, :, :, :], masks[image, :, :, :, :])
    
    print(target_separated.shape)
    
    '''
    # assess correlation
    for image in range(0,n_samples):
        print(image)
        
        p11 = pearsonr(np.abs(target_original0[image,:,:,1].ravel()), np.abs(target_separated_L0[image,:,:].ravel()))[0]
        p12 = pearsonr(np.abs(target_original1[image,:,:,1].ravel()), np.abs(target_separated_L0[image,:,:].ravel()))[0]
        p13 = pearsonr(np.abs(target_original2[image,:,:,1].ravel()), np.abs(target_separated_L0[image,:,:].ravel()))[0]        
        p21 = pearsonr(np.abs(target_original0[image,:,:,1].ravel()), np.abs(target_separated_L1[image,:,:].ravel()))[0]
        p22 = pearsonr(np.abs(target_original1[image,:,:,1].ravel()), np.abs(target_separated_L1[image,:,:].ravel()))[0]
        p23 = pearsonr(np.abs(target_original2[image,:,:,1].ravel()), np.abs(target_separated_L1[image,:,:].ravel()))[0]
        p31 = pearsonr(np.abs(target_original0[image,:,:,1].ravel()), np.abs(target_separated_L2[image,:,:].ravel()))[0]
        p32 = pearsonr(np.abs(target_original1[image,:,:,1].ravel()), np.abs(target_separated_L2[image,:,:].ravel()))[0]
        p33 = pearsonr(np.abs(target_original2[image,:,:,1].ravel()), np.abs(target_separated_L2[image,:,:].ravel()))[0]
        
        # left channel
        if p11 >= p12 and p11 >= p13:
            target_separated_L0[image,:,:] = target_separated_L0[image,:,:]
        elif p12 >= p11 and p12 >= p13 :
            target_separated_L1[image,:,:] = target_separated_L0[image,:,:]
        elif p13 >= p11 and p13 >= p12 :
            target_separated_L2[image,:,:] = target_separated_L0[image,:,:]
        
        if p21 >= p22 and p11 >= p23:
            target_separated_L0[image,:,:] = target_separated_L1[image,:,:]
        elif p22 >= p21 and p22 >= p23 :
            target_separated_L1[image,:,:] = target_separated_L1[image,:,:]
        elif p23 >= p21 and p23 >= p22 :
            target_separated_L2[image,:,:] = target_separated_L1[image,:,:]
            
        if p31 >= p32 and p31 >= p33:
            target_separated_L0[image,:,:] = target_separated_L2[image,:,:]
        elif p32 >= p31 and p32 >= p33 :
            target_separated_L1[image,:,:] = target_separated_L2[image,:,:]
        elif p33 >= p31 and p33 >= p32 :
            target_separated_L2[image,:,:] = target_separated_L2[image,:,:]
        
        
        p11 = pearsonr(np.abs(target_original0[image,:,:,2].ravel()), np.abs(target_separated_R0[image,:,:].ravel()))[0]
        p12 = pearsonr(np.abs(target_original1[image,:,:,2].ravel()), np.abs(target_separated_R0[image,:,:].ravel()))[0]
        p13 = pearsonr(np.abs(target_original2[image,:,:,2].ravel()), np.abs(target_separated_R0[image,:,:].ravel()))[0]        
        p21 = pearsonr(np.abs(target_original0[image,:,:,2].ravel()), np.abs(target_separated_R1[image,:,:].ravel()))[0]
        p22 = pearsonr(np.abs(target_original1[image,:,:,2].ravel()), np.abs(target_separated_R1[image,:,:].ravel()))[0]
        p23 = pearsonr(np.abs(target_original2[image,:,:,2].ravel()), np.abs(target_separated_R1[image,:,:].ravel()))[0]
        p31 = pearsonr(np.abs(target_original0[image,:,:,2].ravel()), np.abs(target_separated_R2[image,:,:].ravel()))[0]
        p32 = pearsonr(np.abs(target_original1[image,:,:,2].ravel()), np.abs(target_separated_R2[image,:,:].ravel()))[0]
        p33 = pearsonr(np.abs(target_original2[image,:,:,2].ravel()), np.abs(target_separated_R2[image,:,:].ravel()))[0]
        
        # right channel
        if p11 >= p12 and p11 >= p13:
            target_separated_R0[image,:,:] = target_separated_R0[image,:,:]
        elif p12 >= p11 and p12 >= p13 :
            target_separated_R1[image,:,:] = target_separated_R0[image,:,:]
        elif p13 >= p11 and p13 >= p12 :
            target_separated_R2[image,:,:] = target_separated_R0[image,:,:]
        
        if p21 >= p22 and p11 >= p23:
            target_separated_R0[image,:,:] = target_separated_R1[image,:,:]
        elif p22 >= p21 and p22 >= p23 :
            target_separated_R1[image,:,:] = target_separated_R1[image,:,:]
        elif p23 >= p21 and p23 >= p22 :
            target_separated_R2[image,:,:] = target_separated_R1[image,:,:]
            
        if p31 >= p32 and p31 >= p33:
            target_separated_R0[image,:,:] = target_separated_R2[image,:,:]
        elif p32 >= p31 and p32 >= p33 :
            target_separated_R1[image,:,:] = target_separated_R2[image,:,:]
        elif p33 >= p31 and p33 >= p32 :
            target_separated_R2[image,:,:] = target_separated_R2[image,:,:]
    '''
    image = 0
    samples_per_angles = int(len(target_separated)/n_doas)
    print(samples_per_angles)
    
    for angles in angles_list:
        
        angles = angles[:n_sources]
        print(angles)

        for index in range(samples_per_angles):
        
            print(index, image)

            angles_string_list = [str(angles[source]) + 'deg' for source in range(len(angles))]
            angles_string = '_'.join(angles_string_list)

            save_path = os.path.join(sound_path, angles_string)

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # write spectrogrma paths
            write_audio(save_path, 'spectrogram', angles_string, image, X_mix, 0, 0, angles, fs)
            write_audio(save_path, 'spectrogram', angles_string, image, X_mix, 0, 1, angles, fs)
            
            # write separate paths
            write_audio(save_path, 'target_separated', angles_string, image, target_separated, 0, 0, angles, fs)
            write_audio(save_path, 'target_separated', angles_string, image, target_separated, 1, 0, angles, fs)
            write_audio(save_path, 'target_separated', angles_string, image, target_separated, 2, 0, angles, fs)
            write_audio(save_path, 'target_separated', angles_string, image, target_separated, 0, 1, angles, fs)
            write_audio(save_path, 'target_separated', angles_string, image, target_separated, 1, 1, angles, fs)
            write_audio(save_path, 'target_separated', angles_string, image, target_separated, 2, 1, angles, fs)

  
            '''
            [_, target_separated_L0_sound] = signal.istft(target_separated[image,:,:,0,0], fs, 'hann', Wlength, overlap, fft_size)
            name = path + '/target_separated_L0_'+ str(angle[0]) + 'deg.wav'
            wavfile.write(name,fs,target_separated_L0_sound)
        
            [_, target_separated_L1_sound] = signal.istft(target_separated[image,:,:,1,0], fs, 'hann', Wlength, overlap, fft_size)
            name = path + '/target_separated_L1_'+ str(angle[1]) + 'deg.wav'
            wavfile.write(name,fs,target_separated_L1_sound)
        
            [_, target_separated_L2_sound] = signal.istft(target_separated[image,:,:,2,0], fs, 'hann', Wlength, overlap, fft_size)
            name = path + '/target_separated_L2_'+ str(angle[2]) + 'deg.wav'
            wavfile.write(name,fs,target_separated_L2_sound)

            [_, target_separated_R0_sound] = signal.istft(target_separated[image,:,:,0,1], fs, 'hann', Wlength, overlap, fft_size)
            name = path + '/target_separated_R0_'+ str(angle[0]) + 'deg.wav'
            wavfile.write(name,fs,target_separated_R0_sound)

            [_, target_separated_R1_sound] = signal.istft(target_separated[image,:,:,1,1], fs, 'hann', Wlength, overlap, fft_size)
            name = path + '/target_separated_R1_'+ str(angle[1]) + 'deg.wav'
            wavfile.write(name,fs,target_separated_R1_sound)

            [_, target_separated_R2_sound] = signal.istft(target_separated[image,:,:,2,1], fs, 'hann', Wlength, overlap, fft_size)
            name = path + '/target_separated_R2_'+ str(angle[2]) + 'deg.wav'
            wavfile.write(name,fs,target_separated_R2_sound)
            '''

            if X_or.shape[3] == 4:

                write_audio(save_path, 'target_separated', angles_string, image, target_separated, 3, 0, angles, fs)
                write_audio(save_path, 'target_separated', angles_string, image, target_separated, 3, 1, angles, fs)


            image += 1
        
    return target_separated


def generate_spectrograms4(spectrograms_left, spectrograms_right, target_original0, target_original1, target_original2, target_original3, mask_target0, mask_target1, mask_target2, mask_target3, sound_path ):

    [n_samples, n_freq, n_time, n_source] = target_original0.shape 
        
    print("Getting the separated spectrograms", time.ctime())
    
    #spectrograms_left = np.ones([n_samples, n_freq, n_time])
    #spectrograms_right = np.ones([n_samples, n_freq, n_time])
    
    # initialize complex spectrograms
    target_separated_L0 = np.empty_like(spectrograms_left)
    target_separated_R0 = np.empty_like(spectrograms_right)
    target_separated_L1 = np.empty_like(spectrograms_left)
    target_separated_R1 = np.empty_like(spectrograms_right)
    target_separated_L2 = np.empty_like(spectrograms_left)
    target_separated_R2 = np.empty_like(spectrograms_right)
    target_separated_L3 = np.empty_like(spectrograms_left)
    target_separated_R3 = np.empty_like(spectrograms_right)
    
    
    # get the spectrograms from the masks
    target_separated_L0 = np.multiply(mask_target0, spectrograms_left)
    target_separated_R0 = np.multiply(mask_target0, spectrograms_right)
    target_separated_L1 = np.multiply(mask_target1, spectrograms_left)
    target_separated_R1 = np.multiply(mask_target1, spectrograms_right)
    target_separated_L2 = np.multiply(mask_target2, spectrograms_left)
    target_separated_R2 = np.multiply(mask_target2, spectrograms_right)
    target_separated_L3 = np.multiply(mask_target3, spectrograms_left)
    target_separated_R3 = np.multiply(mask_target3, spectrograms_right)
    
    '''
    
    # assess correlation
    for image in range(0,n_samples):
        print image
        
        p11 = pearsonr(np.abs(target_original0[image,:,:,1].ravel()), np.abs(target_separated_L0[image,:,:].ravel()))[0]
        p12 = pearsonr(np.abs(target_original1[image,:,:,1].ravel()), np.abs(target_separated_L0[image,:,:].ravel()))[0]
        p13 = pearsonr(np.abs(target_original2[image,:,:,1].ravel()), np.abs(target_separated_L0[image,:,:].ravel()))[0]
        p14 = pearsonr(np.abs(target_original3[image,:,:,1].ravel()), np.abs(target_separated_L0[image,:,:].ravel()))[0]       
        p21 = pearsonr(np.abs(target_original0[image,:,:,1].ravel()), np.abs(target_separated_L1[image,:,:].ravel()))[0]
        p22 = pearsonr(np.abs(target_original1[image,:,:,1].ravel()), np.abs(target_separated_L1[image,:,:].ravel()))[0]
        p23 = pearsonr(np.abs(target_original2[image,:,:,1].ravel()), np.abs(target_separated_L1[image,:,:].ravel()))[0]
        p24 = pearsonr(np.abs(target_original3[image,:,:,1].ravel()), np.abs(target_separated_L1[image,:,:].ravel()))[0]
        p31 = pearsonr(np.abs(target_original0[image,:,:,1].ravel()), np.abs(target_separated_L2[image,:,:].ravel()))[0]
        p32 = pearsonr(np.abs(target_original1[image,:,:,1].ravel()), np.abs(target_separated_L2[image,:,:].ravel()))[0]
        p33 = pearsonr(np.abs(target_original2[image,:,:,1].ravel()), np.abs(target_separated_L2[image,:,:].ravel()))[0]
        p34 = pearsonr(np.abs(target_original3[image,:,:,1].ravel()), np.abs(target_separated_L2[image,:,:].ravel()))[0]
        p41 = pearsonr(np.abs(target_original0[image,:,:,1].ravel()), np.abs(target_separated_L3[image,:,:].ravel()))[0]
        p42 = pearsonr(np.abs(target_original1[image,:,:,1].ravel()), np.abs(target_separated_L3[image,:,:].ravel()))[0]
        p43 = pearsonr(np.abs(target_original2[image,:,:,1].ravel()), np.abs(target_separated_L3[image,:,:].ravel()))[0]
        p44 = pearsonr(np.abs(target_original3[image,:,:,1].ravel()), np.abs(target_separated_L3[image,:,:].ravel()))[0]
        
        # left channel
        if p11 >= p12 and p11 >= p13 and p11 >= p14:
            target_separated_L0[image,:,:] = target_separated_L0[image,:,:]
        elif p12 >= p11 and p12 >= p13 and p12 >= p14:
            target_separated_L1[image,:,:] = target_separated_L0[image,:,:]
        elif p13 >= p11 and p13 >= p12 and p13 >= p14 :
            target_separated_L2[image,:,:] = target_separated_L0[image,:,:]
        elif p14 >= p11 and p14 >= p12 and p14 >= p13 :
            target_separated_L3[image,:,:] = target_separated_L0[image,:,:]
            
        if p21 >= p22 and p21 >= p23 and p21 >= p24:
            target_separated_L0[image,:,:] = target_separated_L1[image,:,:]
        elif p22 >= p21 and p22 >= p23 and p22 >= p24:
            target_separated_L1[image,:,:] = target_separated_L1[image,:,:]
        elif p23 >= p21 and p23 >= p22 and p23 >= p24 :
            target_separated_L2[image,:,:] = target_separated_L1[image,:,:]
        elif p24 >= p21 and p24 >= p22 and p24 >= p23 :
            target_separated_L3[image,:,:] = target_separated_L1[image,:,:]
            
        if p31 >= p32 and p31 >= p33 and p31 >= p34:
            target_separated_L0[image,:,:] = target_separated_L2[image,:,:]
        elif p32 >= p31 and p32 >= p33 and p32 >= p34:
            target_separated_L1[image,:,:] = target_separated_L2[image,:,:]
        elif p33 >= p31 and p33 >= p32 and p33 >= p34 :
            target_separated_L2[image,:,:] = target_separated_L2[image,:,:]
        elif p34 >= p31 and p34 >= p32 and p34 >= p33 :
            target_separated_L3[image,:,:] = target_separated_L2[image,:,:]
            
        if p41 >= p42 and p41 >= p43 and p41 >= p44:
            target_separated_L0[image,:,:] = target_separated_L3[image,:,:]
        elif p42 >= p41 and p42 >= p43 and p42 >= p44:
            target_separated_L1[image,:,:] = target_separated_L3[image,:,:]
        elif p43 >= p41 and p43 >= p42 and p43 >= p44 :
            target_separated_L2[image,:,:] = target_separated_L3[image,:,:]
        elif p44 >= p41 and p44 >= p42 and p44 >= p43 :
            target_separated_L3[image,:,:] = target_separated_L3[image,:,:]
        
        
        p11 = pearsonr(np.abs(target_original0[image,:,:,2].ravel()), np.abs(target_separated_R0[image,:,:].ravel()))[0]
        p12 = pearsonr(np.abs(target_original1[image,:,:,2].ravel()), np.abs(target_separated_R0[image,:,:].ravel()))[0]
        p13 = pearsonr(np.abs(target_original2[image,:,:,2].ravel()), np.abs(target_separated_R0[image,:,:].ravel()))[0]
        p14 = pearsonr(np.abs(target_original3[image,:,:,2].ravel()), np.abs(target_separated_R0[image,:,:].ravel()))[0]       
        p21 = pearsonr(np.abs(target_original0[image,:,:,2].ravel()), np.abs(target_separated_R1[image,:,:].ravel()))[0]
        p22 = pearsonr(np.abs(target_original1[image,:,:,2].ravel()), np.abs(target_separated_R1[image,:,:].ravel()))[0]
        p23 = pearsonr(np.abs(target_original2[image,:,:,2].ravel()), np.abs(target_separated_R1[image,:,:].ravel()))[0]
        p24 = pearsonr(np.abs(target_original3[image,:,:,2].ravel()), np.abs(target_separated_R1[image,:,:].ravel()))[0]
        p31 = pearsonr(np.abs(target_original0[image,:,:,2].ravel()), np.abs(target_separated_R2[image,:,:].ravel()))[0]
        p32 = pearsonr(np.abs(target_original1[image,:,:,2].ravel()), np.abs(target_separated_R2[image,:,:].ravel()))[0]
        p33 = pearsonr(np.abs(target_original2[image,:,:,2].ravel()), np.abs(target_separated_R2[image,:,:].ravel()))[0]
        p34 = pearsonr(np.abs(target_original3[image,:,:,2].ravel()), np.abs(target_separated_R2[image,:,:].ravel()))[0]
        p41 = pearsonr(np.abs(target_original0[image,:,:,2].ravel()), np.abs(target_separated_R3[image,:,:].ravel()))[0]
        p42 = pearsonr(np.abs(target_original1[image,:,:,2].ravel()), np.abs(target_separated_R3[image,:,:].ravel()))[0]
        p43 = pearsonr(np.abs(target_original2[image,:,:,2].ravel()), np.abs(target_separated_R3[image,:,:].ravel()))[0]
        p44 = pearsonr(np.abs(target_original3[image,:,:,2].ravel()), np.abs(target_separated_R3[image,:,:].ravel()))[0]
        
        # right channel
        if p11 >= p12 and p11 >= p13 and p11 >= p14:
            target_separated_R0[image,:,:] = target_separated_R0[image,:,:]
        elif p12 >= p11 and p12 >= p13 and p12 >= p14:
            target_separated_R1[image,:,:] = target_separated_R0[image,:,:]
        elif p13 >= p11 and p13 >= p12 and p13 >= p14 :
            target_separated_R2[image,:,:] = target_separated_R0[image,:,:]
        elif p14 >= p11 and p14 >= p12 and p14 >= p13 :
            target_separated_R3[image,:,:] = target_separated_R0[image,:,:]
            
        if p21 >= p22 and p21 >= p23 and p21 >= p24:
            target_separated_R0[image,:,:] = target_separated_R1[image,:,:]
        elif p22 >= p21 and p22 >= p23 and p22 >= p24:
            target_separated_R1[image,:,:] = target_separated_R1[image,:,:]
        elif p23 >= p21 and p23 >= p22 and p23 >= p24 :
            target_separated_R2[image,:,:] = target_separated_R1[image,:,:]
        elif p24 >= p21 and p24 >= p22 and p24 >= p23 :
            target_separated_R3[image,:,:] = target_separated_R1[image,:,:]
            
        if p31 >= p32 and p31 >= p33 and p31 >= p34:
            target_separated_R0[image,:,:] = target_separated_R2[image,:,:]
        elif p32 >= p31 and p32 >= p33 and p32 >= p34:
            target_separated_R1[image,:,:] = target_separated_R2[image,:,:]
        elif p33 >= p31 and p33 >= p32 and p33 >= p34 :
            target_separated_R2[image,:,:] = target_separated_R2[image,:,:]
        elif p34 >= p31 and p34 >= p32 and p34 >= p33 :
            target_separated_R3[image,:,:] = target_separated_R2[image,:,:]
            
        if p41 >= p42 and p41 >= p43 and p41 >= p44:
            target_separated_R0[image,:,:] = target_separated_R3[image,:,:]
        elif p42 >= p41 and p42 >= p43 and p42 >= p44:
            target_separated_R1[image,:,:] = target_separated_R3[image,:,:]
        elif p43 >= p41 and p43 >= p42 and p43 >= p44 :
            target_separated_R2[image,:,:] = target_separated_R3[image,:,:]
        elif p44 >= p41 and p44 >= p42 and p44 >= p43 :
            target_separated_R3[image,:,:] = target_separated_R3[image,:,:]
    '''
    index = 0
    for angle in angles_list:
        
        image = index*int(len(target_separated_L0)/n_doas)
        print(index)
        print(image)

        angle0 = angle[0]
        angle1 = angle[1]
        angle2 = angle[2]
        angle3 = angle[3]
        print(angle)
        print(angle0)
        
        path = os.path.join(sound_path, str(angle0) + 'deg_' + str(angle1) + 'deg_' + str(angle2) + 'deg_' + str(angle3) + 'deg'  )
        if not os.path.exists(path):
            os.makedirs(path)
        
        [_, target_separated_L0_sound] = signal.istft(target_separated_L0[image,:,:], fs, 'hann', Wlength, overlap, fft_size)
        name = path + '/target_separated_L0_'+ str(angle0) + 'deg.wav'
        wavfile.write(name,fs,target_separated_L0_sound)
    
        [_, target_separated_L1_sound] = signal.istft(target_separated_L1[image,:,:], fs, 'hann', Wlength, overlap, fft_size)
        name = path + '/target_separated_L1_'+ str(angle1) + 'deg.wav'
        wavfile.write(name,fs,target_separated_L1_sound)
    
        [_, target_separated_L2_sound] = signal.istft(target_separated_L2[image,:,:], fs, 'hann', Wlength, overlap, fft_size)
        name = path + '/target_separated_L2_'+ str(angle2) + 'deg.wav'
        wavfile.write(name,fs,target_separated_L2_sound)
        
        [_, target_separated_L3_sound] = signal.istft(target_separated_L3[image,:,:], fs, 'hann', Wlength, overlap, fft_size)
        name = path + '/target_separated_L3_'+ str(angle3) + 'deg.wav'
        wavfile.write(name,fs,target_separated_L3_sound)

        [_, spectrogram_L_sound] = signal.istft(spectrograms_left[image,:,:], fs, 'hann', Wlength, overlap, fft_size)
        name = path + '/spectrogram_L_'+ str(angle0) + 'deg_' + str(angle1) + 'deg_' + str(angle2) + 'deg_' + str(angle3) + 'deg.wav'
        wavfile.write(name,fs,spectrogram_L_sound)

        index += 1
    

    return target_separated_L0, target_separated_R0, target_separated_L1, target_separated_R1, target_separated_L2, target_separated_R2, target_separated_L3, target_separated_R3
