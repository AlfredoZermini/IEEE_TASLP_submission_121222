import numpy as np
import os
import time
import sys
from separation import bss_eval_sources
from scipy import signal
import time
import h5py
from scipy.io import wavfile
#import wavfile
import matplotlib.pyplot as plt
#import mir_eval
#import vqmetrics
import subprocess
import soundfile as sf
import librosa
import matlab.engine
import soundfile
from generate_spectrograms import *


# STFT params
Wlength = 2048
window = 'hann'
window_size = Wlength#1024
hop_size = 512 #512
overlap = window_size - hop_size
fft_size = Wlength
fs = 16000
n_metrics = 4
n_mixtures=20 #!!!!
write_audio = True

# angles list
angles_list = [ [0,10,20,30], [0,20,40,60], [0,30,60,90], [0,40,80,120], [0,50,100,150],  [0,60,120,180], [0,70,140,210], [0,80,160,240], [0,90,180,270]  ]
my_methods = ['DNN', 'CNN', 'IRM', 'IRM_mono', 'IRM_mono_square', 'IRM_stereo', 'IRM_stereo_square']

# load matlab engine
eng = matlab.engine.start_matlab()
eng.addpath(r'/media/alfredo/storage/B_format/matlab_code',nargout=0)


def evaluate_performance(matlab_path, p0, target_separated, baseline, n_sources_string):

    # baseline path
    method_path = os.path.join(matlab_path, n_sources_string + 'Sources', baseline)

    if not os.path.exists(method_path):
        os.makedirs(method_path)

    # sum all sources to get audio mixture
    p0_mix = np.sum(p0, axis=3)

    # get parameters
    [n_samples, n_freq, n_time, n_sources, n_channels] = p0.shape

    [_, original] = signal.istft(p0_mix[0,:,:,:], fs, 'hann', Wlength, overlap, fft_size)
    n_time_audio = len(original)
    tmax = 34816 # to make it comparable to the baseline results

    # define sdrs tensor shape
    metrics_tensor = np.zeros([n_mixtures, int(n_samples/n_mixtures), n_metrics, n_sources ]  )
    metrics_mix_tensor = np.zeros([n_mixtures, int(n_samples/n_mixtures), n_metrics, n_sources ] )

    sound_set = doa = 0

    # my method path
    #my_path = os.path.join(matlab_path, n_sources_string + 'Sources', 'Zermini')


    for sample in range(0, n_samples):

        if sample % n_mixtures == 0 and sample != 0: # and sample != n_samples:

            sound_set = 0
            doa = doa+1

            angle0 = angles_list[doa][0]
            angle1 = angles_list[doa][1]
            angle2 = angles_list[doa][2]
            angle3 = angles_list[doa][3]

            if n_sources == 3:
                angle_list = [str(angle0), str(angle1), str(angle2)]
            elif n_sources == 4:
                angle_list = [str(angle0), str(angle1), str(angle2), str(angle3)]

            angles_folder = os.path.join('_'.join(angle_list)  )
            print(baseline, angle_list)

        elif sample == 0:

            angle0 = angles_list[doa][0]
            angle1 = angles_list[doa][1]
            angle2 = angles_list[doa][2]
            angle3 = angles_list[doa][3]

            if n_sources == 3:
                angle_list = [str(angle0), str(angle1), str(angle2)]
            elif n_sources == 4:
                angle_list = [str(angle0), str(angle1), str(angle2), str(angle3)]
            

            angles_folder = os.path.join('_'.join(angle_list)  )

            print(angle_list)

        # angles_path
        angles_path = os.path.join(method_path, angles_folder)

        # stereo path
        stereo_path = os.path.join(angles_path, 'stereo')

        # mkdirs
        if not os.path.exists(angles_path):
            os.makedirs(angles_path)
        if not os.path.exists(stereo_path):
            os.makedirs(stereo_path)

        # stereo paths
        target0_path = os.path.join(stereo_path, 'Est_s'+ str(0) + '_' + str(sample%n_mixtures) + '.wav')
        target1_path = os.path.join(stereo_path, 'Est_s'+ str(1) + '_' + str(sample%n_mixtures) + '.wav')
        target2_path = os.path.join(stereo_path, 'Est_s'+ str(2) + '_' + str(sample%n_mixtures) + '.wav')
        target3_path = os.path.join(stereo_path, 'Est_s'+ str(3) + '_' + str(sample%n_mixtures) + '.wav')

        if baseline not in my_methods:
            
            # separated target
            target0 ,_ = librosa.load(target0_path, sr=fs, mono=False)
            target1 ,_ = librosa.load(target1_path, sr=fs, mono=False)
            target2 ,_ = librosa.load(target2_path, sr=fs, mono=False)

            if n_sources == 4:
                target3 ,_ = librosa.load(target3_path, sr=fs, mono=False)
        
        else:
            for source in range(n_sources):

                target_path = os.path.join(stereo_path, 'Est_s'+ str(source) + '_' + str(sample%n_mixtures) + '.wav')
                
                if write_audio == True:
                    [_, target_L] = signal.istft( target_separated[sample,:,:,source,0], fs, 'hann', Wlength, overlap, fft_size)
                    [_, target_R] = signal.istft( target_separated[sample,:,:,source,1], fs, 'hann', Wlength, overlap, fft_size)
                    target = np.vstack((target_L, target_R)).T    
                    
                    # save normalized version for stereo only
                    #target = waveform_norm(target)
                    soundfile.write(target_path, target, fs)
                
        
        # mean of LR channels
        target0 ,_ = librosa.load(target0_path, sr=fs, mono=False)
        target0=target0.T
        target0_mean_path = os.path.join(angles_path,  'Est_s' + str(0) + '_' + str(sample % n_mixtures) + '.wav')
        #target0_mean = waveform_norm(np.mean(target0[0:tmax,:], axis=1))
        target0_mean = np.mean(target0[0:tmax,:], axis=1)
        soundfile.write(target0_mean_path, target0_mean, fs)

        target1 ,_ = librosa.load(target1_path, sr=fs, mono=False)
        target1=target1.T
        target1_mean_path = os.path.join(angles_path,  'Est_s' + str(1) + '_' + str(sample % n_mixtures) + '.wav')
        #target1_mean = waveform_norm(np.mean(target1[0:tmax,:], axis=1))
        target1_mean = np.mean(target1[0:tmax,:], axis=1)
        soundfile.write(target1_mean_path, target1_mean, fs)

        target2 ,_ = librosa.load(target2_path, sr=fs, mono=False)
        target2=target2.T
        target2_mean_path =  os.path.join(angles_path,  'Est_s' + str(2) + '_' + str(sample % n_mixtures) + '.wav')
        #target2_mean = waveform_norm(np.mean(target2[0:tmax,:], axis=1))
        target2_mean = np.mean(target2[0:tmax,:], axis=1)
        soundfile.write(target2_mean_path, target2_mean, fs)

        if n_sources == 4:
            target3 ,_ = librosa.load(target3_path, sr=fs, mono=False)
            target3=target3.T
            target3_mean_path = os.path.join(angles_path,  'Est_s' + str(3) + '_' + str(sample % n_mixtures) + '.wav')
            #target3_mean = waveform_norm(np.mean(target3[0:tmax,:], axis=1))
            target3_mean = np.mean(target3[0:tmax,:], axis=1)
            soundfile.write(target3_mean_path, target3_mean, fs)

        # original and mixtures path
        mixtures_path = os.path.join(matlab_path, n_sources_string + 'Sources', 'Mixtures', angles_folder)
        originall_path = os.path.join(matlab_path, n_sources_string + 'Sources', 'Original', angles_folder)


        # create paths if do not exist
        if not os.path.exists(mixtures_path):
            os.makedirs(mixtures_path)

        if not os.path.exists(originall_path):
            os.makedirs(originall_path)

        # audio paths
        mixture_path = os.path.join(mixtures_path,'mixture' + str(sample % n_mixtures) + '.wav')
        original0_path = os.path.join(originall_path,'original' + str(0) + str(sample % n_mixtures) + '.wav')
        original1_path = os.path.join(originall_path,'original' + str(1) + str(sample % n_mixtures) + '.wav')
        original2_path = os.path.join(originall_path,'original' + str(2) + str(sample % n_mixtures) + '.wav')
        original3_path = os.path.join(originall_path,'original' + str(3) + str(sample % n_mixtures) + '.wav')

        ### METRICS
        
        # define lists
        sdr_list = []
        sir_list = []
        sar_list = []
        pesq_list = []
        sdr_mix_list = []
        sir_mix_list = []
        sar_mix_list = []
        pesq_list_mix = []

        # loop on channels
        for ch in range(2):

            # write audio if do not exist
            [_, mixture] = signal.istft(p0_mix[sample,:,:,ch], fs, 'hann', Wlength, overlap, fft_size)
            #soundfile.write(mixture_path, waveform_norm(mixture[0:tmax]), fs )
            soundfile.write(mixture_path, mixture[0:tmax], fs )

            # source 0
            [_, original0] = signal.istft(p0[sample,:,:,0,ch], fs, 'hann', Wlength, overlap, fft_size)
            #soundfile.write(original0_path, waveform_norm(original0[0:tmax]), fs )
            soundfile.write(original0_path, original0[0:tmax], fs )

            # source 1
            [_, original1] = signal.istft(p0[sample,:,:,1,ch], fs, 'hann', Wlength, overlap, fft_size)
            #soundfile.write(original1_path, waveform_norm(original1[0:tmax]), fs )
            soundfile.write(original1_path, original1[0:tmax], fs )

            # source 2
            [_, original2] = signal.istft(p0[sample,:,:,2,ch], fs, 'hann', Wlength, overlap, fft_size)
            #soundfile.write(original2_path, waveform_norm(original2[0:tmax]), fs )
            soundfile.write(original2_path, original2[0:tmax], fs )


            if n_sources == 4:  # source 3
                [_, original3] = signal.istft(p0[sample,:,:,3,ch], fs, 'hann', Wlength, overlap, fft_size)
                #soundfile.write(original3_path, waveform_norm(original3[0:tmax]), fs )
                soundfile.write(original3_path, original3[0:tmax], fs )

            ### EVALUATE METRICS
            if n_sources == 3:
                sdr, sir, sar = eng.bss_eval3_wrapper(target0_mean_path, target1_mean_path, target2_mean_path, original0_path, original1_path, original2_path, nargout=3)
                sdr_mix, sir_mix, sar_mix = eng.bss_eval3_mix_wrapper(mixture_path, original0_path, original1_path, original2_path, nargout=3)
                #print(sdr)
                
            elif n_sources == 4:
                sdr, sir, sar = eng.bss_eval4_wrapper(target0_mean_path, target1_mean_path, target2_mean_path, target3_mean_path, original0_path, original1_path, original2_path, original3_path, nargout=3)
                sdr_mix, sir_mix, sar_mix = eng.bss_eval4_mix_wrapper(mixture_path, original0_path, original1_path, original2_path, original3_path,  nargout=3)

            sdr_list.append(np.asarray(sdr))
            print(sdr_list)
            sir_list.append(np.asarray(sir))
            sar_list.append(np.asarray(sar))
            sdr_mix_list.append(np.asarray(sdr_mix))
            sir_mix_list.append(np.asarray(sir_mix))
            sar_mix_list.append(np.asarray(sar_mix))

            # # evaluate pesq
            # pesq = []
            # pesq_mix = []

            # pesq0 = eng.pesq(original0_path,target0_mean_path);
            # pesq0_mix = eng.pesq(original0_path,mixture_path);

            # pesq1 = eng.pesq(original1_path,target1_mean_path);
            # pesq1_mix = eng.pesq(original1_path,mixture_path);

            # pesq2 = eng.pesq(original2_path,target2_mean_path);
            # pesq2_mix = eng.pesq(original2_path,mixture_path);

            # if n_sources == 3:
            #     pesq.append(pesq0, pesq1, pesq2)
            #     pesq_mix.append(pesq0_mix, pesq1_mix, pesq2_mix)

            # elif n_sources == 4:
            #     pesq3 = eng.pesq(original3_path,target3_mean_path);
            #     pesq3_mix = eng.pesq(original3_path,mixture_path);
            #     pesq.append(pesq0, pesq1, pesq2, pesq3)
            #     pesq_mix.append(pesq0_mix, pesq1_mix, pesq2_mix, pesq3_mix)
        cocco
        print(sample, sample % n_mixtures, sound_set, doa, 'SDR=', np.mean(sdr),'SIR=', np.mean(sir), 'SAR=', np.mean(sar), 'PESQ=', np.mean(pesq))

        # fill metrics tensor
        for source in range (0, n_sources):
            metrics_tensor[sound_set, doa, 0, source] =  np.asarray(sdr)[source]
            metrics_tensor[sound_set, doa, 1, source] =  np.asarray(sir)[source]
            metrics_tensor[sound_set, doa, 2, source] =  np.asarray(sar)[source]
            metrics_tensor[sound_set, doa, 3, source] =  np.asarray(pesq)[source]

            metrics_mix_tensor[sound_set, doa, 0, source] = np.asarray(sdr_mix)[source]
            metrics_mix_tensor[sound_set, doa, 1, source] = np.asarray(sir_mix)[source]
            metrics_mix_tensor[sound_set, doa, 2, source] = np.asarray(sar_mix)[source]
            metrics_mix_tensor[sound_set, doa, 3, source] = np.asarray(pesq_mix)[source]


        # increase angle index
        sound_set = sound_set + 1

    return metrics_tensor, metrics_mix_tensor