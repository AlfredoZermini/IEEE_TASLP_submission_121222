import os
import sys
import time
import scipy.io
from scipy.io import wavfile
from scipy import signal
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from stats import RunningStats
import h5py
import librosa
import random
from random import randint
from operator import sub
from generate_list import *


def convolve(rir, speech):

    return signal.convolve(rir, speech)

def assign_rir_to_index(rirs_resample, angles, index):

    rir_res = rirs_resample[int(angles[index]/10), :, :] ### EDITED int()
    return rir_res

    
def prepare_data_Bformat(args):
    
    # generate list of TIMIT speakers
    list_save = []
    file_path = os.path.join(args.WORK_PATH, 'txt', args.task + '.txt')
    print(file_path)
    text_file = open(file_path, "r")
  
    # reading the file
    data = text_file.read()
    list = data.split("\n")
    text_file.close()
    #list = list[0:6] #!!!

    # clone of the original list when all the elements are deleted
    list_save = list[:]
    
    # initialize indices
    #mix_index = 0
    #index_utt = 0
    
    # initialize lists 
    angles_list = [ [0,10,20,30], [0,20,40,60], [0,30,60,90], [0,40,80,120], [0,50,100,150],  [0,60,120,180], [0,70,140,210], [0,80,160,240], [0,90,180,270]  ] 
    
    # get time lenghts
    rirs = scipy.io.loadmat(os.path.join(args.rir_path, 'B_format_RIRs_12BB01_Alfredo_S3A') )['rirs_final']
    rirs_len = len(librosa.core.resample( y=rirs[0,0,:] , orig_sr=48000, target_sr=args.fs))
    
    # resample RIRs to 16kHz
    rirs_resample = np.zeros([args.n_doas, args.n_channels, rirs_len])
    for ch in range(0, args.n_channels):
        rirs_resample[:,ch,:] =  librosa.core.resample( y=rirs[:,ch,:] , orig_sr=48000, target_sr=args.fs)
    
    del rirs

    # initialize list
    data_list = []
    
    # loop over angles
    for angles in angles_list:

        print(angles[0], angles[1], angles[2], angles[3])
        print(args.n_sources_string)

        # initialize
        if args.n_sources_string == 'Three':
            angles_strings_list = [str(angles[0]), str(angles[1]), str(angles[2])]
        elif args.n_sources_string == 'Four':
            angles_strings_list = [str(angles[0]), str(angles[1]), str(angles[2]), str(angles[3])]
        index_list = 0

           
        if args.task == 'test':
            p0_mat = np.zeros([args.n_time, 1, args.n_utt, args.n_sources])
            vel_mat = np.zeros([args.n_time, 2, args.n_utt, args.n_sources])
            mixind_mat = np.zeros([args.n_sources, args.n_utt])    
            utt_mat = np.zeros([args.n_time, args.n_utt*args.n_sources])
            
           
        ### create mixtures
        # iterate over utterances mixtures
        for mix_index in range(0, args.n_utt):
            
            # define list containing the selected n_sources speech utterances
            selected_speech_paths_list = []

            # initialize min_length to a large value
            min_length = 10000000
                    
            # loop on sources
            for index in range(0, args.n_sources):

                # restore full list if empty
                if list == []:
                    list = list_save[:]
                        
                # select random speech and remove it from list
                speech_path = random.choice(list)
                speech_sample =  sf.read(speech_path)[0] #[0:min_length]
                
                # add to list
                selected_speech_paths_list.append(speech_path)

                # remove sample from list
                list.remove(speech_path)

                # assign pressures
                rir_res = assign_rir_to_index(rirs_resample, angles, index)

                # define pressures
                p0 = convolve(rir_res[0,:], speech_sample)

                # get min_lengt of each mixture
                if len(p0) < min_length:
                    min_length = len(p0)


            ### loop again, but this time cut to previously calculated min_length
            for index in range(0, args.n_sources):
                
                # avoid repeating initialization
                if index == 0:
                    # define tensors
                    vel_tensor = np.zeros([args.n_sources, min_length, 2])
                    p0_tensor = np.zeros([args.n_sources, min_length])
                    label_tensor = np.zeros([args.n_sources])

                # need to reload speech_sample
                speech_sample =  sf.read(selected_speech_paths_list[index])[0]

                # get RIRs for each source index
                rir_res = assign_rir_to_index(rirs_resample, angles, index)

                # define pressures
                p0 = convolve(rir_res[0,:], speech_sample)
                vel_0 = convolve(rir_res[1,:], speech_sample)
                vel_1 = convolve(rir_res[2,:], speech_sample)

                # cut to min_length
                p0 = p0[:min_length]
                vel_0 = vel_0[:min_length]
                vel_1 = vel_1[:min_length]

                # fill tensors
                p0_tensor[index, :] = p0/np.max(np.abs(p0))
                vel_tensor[index,:, 0] = vel_0/np.max(np.abs(p0))
                vel_tensor[index,:, 1] = vel_1/np.max(np.abs(p0))
                label_tensor[index] = int(angles_strings_list[index])

                # pack pressures into data_list
                if index == args.n_sources-1:
                    
                    data_list.append([p0_tensor, vel_tensor, label_tensor])
                    del p0_tensor, vel_tensor, label_tensor

                
                ### DATA CREATION FOR MATLAB CODE (baseline)
                # if args.task == 'test':
                    
                #     # B-format channels
                #     p0_mat[:, 0, mix_index, index] = p0/np.max(np.abs(p0))
                #     vel_mat[:, 0, mix_index, index] = vel_0/np.max(np.abs(p0))
                #     vel_mat[:, 1, mix_index, index] = vel_1/np.max(np.abs(p0))
                    
                #     # mixing indices
                #     mixind_mat[index,mix_index] = args.n_sources*mix_index+index+1
                    
                #     # original utterances
                #     utt_mat[:,index_list] = p0/np.max(np.abs(p0))
                        
                #     index_list += 1

            #mix_index = mix_index+1
            
    print(mix_index, len(data_list), len(data_list[0]))

    return data_list
    
