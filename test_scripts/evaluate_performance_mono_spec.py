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
angles_list = [ [0,10,20,30], [0,20,40,60], [0,30,60,90], [0,40,80,120], [0,50,100,150], [0,60,120,180], [0,70,140,210], [0,80,160,240], [0,90,180,270]  ]
my_methods = ['DNN', 'CNN', 'IRM', 'IRM_mono', 'IRM_mono_square', 'IRM_stereo', 'IRM_stereo_square']

# load matlab engine
eng = matlab.engine.start_matlab()
eng.addpath(r'/vol/vssp/mightywings/B_format/matlab_code',nargout=0)


def mean_spec(mono_path, target, tmax):

    [_, _, S_L] = signal.stft(target[0,:], fs, 'hann', Wlength, overlap, fft_size)
    [_, _, S_R] = signal.stft(target[1,:], fs, 'hann', Wlength, overlap, fft_size)

    S = np.stack([S_L, S_R])
    S_mean = np.mean(S, axis=0)

    [_, s] = signal.istft(S_mean, fs, 'hann', Wlength, overlap, fft_size)
    soundfile.write(mono_path, s[:tmax], fs)
    return None


def write_mono(mono_path, stereo_path, S, sample, source, tmax):

    # channel is shiftd by 1 because there are 3 channels
    [_, mixture_L] = signal.istft(S[sample,:,:,source,1], fs, 'hann', Wlength, overlap, fft_size)
    [_, mixture_R] = signal.istft(S[sample,:,:,source,2], fs, 'hann', Wlength, overlap, fft_size)

    # save stereo
    mixture_stereo = np.vstack((mixture_L,mixture_R))
    mixture_stereo_path = os.path.join(stereo_path, os.path.basename(mono_path))
    if not os.path.exists(mixture_stereo_path): 
        soundfile.write(mixture_stereo_path, mixture_stereo[:tmax, :].T, fs)

    S_mean = np.mean(S[sample,:,:,source,1:3], axis=2)
    [_, mixture_mono] = signal.istft(S_mean, fs, 'hann', Wlength, overlap, fft_size)

    # save mono
    if not os.path.exists(mono_path):
        soundfile.write(mono_path, mixture_mono[:tmax], fs)
    return None


def flat_list(input):
    return [float(el) for el in  np.asarray(input)]


def array(input):
    return float(np.array(input))


def output_array(input):
    return np.mean(np.array(input), axis=0)


def evaluate_performance_mono_spec(matlab_path, p0, target_separated, baseline, n_sources_string):

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


        # angles_path
        angles_path = os.path.join(method_path, angles_folder)
        angles_mixture_path =  os.path.join(method_path.replace(baseline, 'Mixtures'), angles_folder)
        angles_original_path =  os.path.join(method_path.replace(baseline, 'Original'), angles_folder)

        # stereo path
        stereo_path = os.path.join(angles_path, 'stereo')
        stereo_mixture_path = os.path.join(angles_mixture_path, 'stereo')
        stereo_original_path = os.path.join(angles_original_path, 'stereo')
        stereo_separated_path = os.path.join(angles_path, 'stereo_separated')

        # mkdirs
        if not os.path.exists(angles_path):
            os.makedirs(angles_path)
        if not os.path.exists(stereo_path):
            os.makedirs(stereo_path)
        if not os.path.exists(stereo_mixture_path):
            os.makedirs(stereo_mixture_path)
        if not os.path.exists(stereo_original_path):
            os.makedirs(stereo_original_path)
        if not os.path.exists(stereo_separated_path):
            os.makedirs(stereo_separated_path)

        # stereo paths
        target0_path = os.path.join(stereo_path, 'Est_s'+ str(0) + '_' + str(sample%n_mixtures) + '.wav')
        target1_path = os.path.join(stereo_path, 'Est_s'+ str(1) + '_' + str(sample%n_mixtures) + '.wav')
        target2_path = os.path.join(stereo_path, 'Est_s'+ str(2) + '_' + str(sample%n_mixtures) + '.wav')
        target3_path = os.path.join(stereo_path, 'Est_s'+ str(3) + '_' + str(sample%n_mixtures) + '.wav')
        
        # mono paths
        target0_mean_path = os.path.join(angles_path, 'Est_s'+ str(0) + '_' + str(sample%n_mixtures) + '_monospec.wav')
        target1_mean_path = os.path.join(angles_path, 'Est_s'+ str(1) + '_' + str(sample%n_mixtures) + '_monospec.wav')
        target2_mean_path = os.path.join(angles_path, 'Est_s'+ str(2) + '_' + str(sample%n_mixtures) + '_monospec.wav')
        target3_mean_path = os.path.join(angles_path, 'Est_s'+ str(3) + '_' + str(sample%n_mixtures) + '_monospec.wav')      
                        

        if baseline not in my_methods:
            
            # separated target
            target0 ,_ = librosa.load(target0_path, sr=fs, mono=False)
            target1 ,_ = librosa.load(target1_path, sr=fs, mono=False)
            target2 ,_ = librosa.load(target2_path, sr=fs, mono=False)

            if n_sources == 4:
                target3 ,_ = librosa.load(target3_path, sr=fs, mono=False)
        
        else:
            for source in range(n_sources):

                target0_path = os.path.join(stereo_path, 'Est_s'+ str(0) + '_' + str(sample%n_mixtures) + '.wav')
                target1_path = os.path.join(stereo_path, 'Est_s'+ str(1) + '_' + str(sample%n_mixtures) + '.wav')
                target2_path = os.path.join(stereo_path, 'Est_s'+ str(2) + '_' + str(sample%n_mixtures) + '.wav')
                target3_path = os.path.join(stereo_path, 'Est_s'+ str(3) + '_' + str(sample%n_mixtures) + '.wav')

                if write_audio == True:
                    [_, target_L0] = signal.istft( target_separated[sample,:,:,0,0], fs, 'hann', Wlength, overlap, fft_size)
                    [_, target_R0] = signal.istft( target_separated[sample,:,:,0,1], fs, 'hann', Wlength, overlap, fft_size)
                    [_, target_L1] = signal.istft( target_separated[sample,:,:,1,0], fs, 'hann', Wlength, overlap, fft_size)
                    [_, target_R1] = signal.istft( target_separated[sample,:,:,1,1], fs, 'hann', Wlength, overlap, fft_size)
                    [_, target_L2] = signal.istft( target_separated[sample,:,:,2,0], fs, 'hann', Wlength, overlap, fft_size)
                    [_, target_R2] = signal.istft( target_separated[sample,:,:,2,1], fs, 'hann', Wlength, overlap, fft_size)
                    
                    target0 = np.vstack((target_L0, target_R0))    
                    target1 = np.vstack((target_L1, target_R1))
                    target2 = np.vstack((target_L2, target_R2))

                    soundfile.write(target0_path, target0[:tmax, :].T, fs)
                    soundfile.write(target1_path, target1[:tmax, :].T, fs)
                    soundfile.write(target2_path, target2[:tmax, :].T, fs)
                    
                    if n_sources == 4:
                        
                        [_, target_L3] = signal.istft( target_separated[sample,:,:,3,0], fs, 'hann', Wlength, overlap, fft_size)
                        [_, target_R3] = signal.istft( target_separated[sample,:,:,3,1], fs, 'hann', Wlength, overlap, fft_size)
                        target3 = np.vstack((target_L3, target_R3))
                        soundfile.write(target3_path, target3[:tmax, :].T, fs)
       

        #write mono audio 
        mean_spec(target0_mean_path, target0, tmax)
        mean_spec(target1_mean_path, target1, tmax)
        mean_spec(target2_mean_path, target2, tmax)
        if n_sources == 4:
            mean_spec(target3_mean_path, target3, tmax)

        # original and mixtures path
        mixtures_path = os.path.join(matlab_path, n_sources_string + 'Sources', 'Mixtures', angles_folder)
        originall_path = os.path.join(matlab_path, n_sources_string + 'Sources', 'Original', angles_folder)

        # create paths if do not exist
        if not os.path.exists(mixtures_path):
            os.makedirs(mixtures_path)

        if not os.path.exists(originall_path):
            os.makedirs(originall_path)

        # audio paths
        mixture_path = os.path.join(mixtures_path,'mixture' + str(sample % n_mixtures) + '_monospec.wav')
        original0_path = os.path.join(originall_path,'original' + str(0) + str(sample % n_mixtures) + '_monospec.wav')
        original1_path = os.path.join(originall_path,'original' + str(1) + str(sample % n_mixtures) + '_monospec.wav')
        original2_path = os.path.join(originall_path,'original' + str(2) + str(sample % n_mixtures) + '_monospec.wav')
        original3_path = os.path.join(originall_path,'original' + str(3) + str(sample % n_mixtures) + '_monospec.wav')


        ### METRICS
        # define lists
        sdr_list = []
        sir_list = []
        sar_list = []
        pesq_list = []
        sdr_mix_list = []
        sir_mix_list = []
        sar_mix_list = []
        pesq_mix_list = []


        # loop on channels
        # targets
        target0_mono = np.mean(target0[:, 0:tmax], axis=0)
        soundfile.write(target0_mean_path, target0_mono, fs )

        target1_mono = np.mean(target1[:, 0:tmax], axis=0)
        soundfile.write(target1_mean_path, target1_mono, fs)

        target2_mono = np.mean(target2[:, 0:tmax], axis=0)
        soundfile.write(target2_mean_path, target2_mono, fs)
        
        # write audio
        write_mono(mixture_path, stereo_mixture_path, np.expand_dims(p0_mix, axis=3), sample, 0, tmax)
        write_mono(original0_path, stereo_original_path, p0, sample, 0, tmax)
        write_mono(original1_path, stereo_original_path, p0, sample, 1, tmax)
        write_mono(original2_path, stereo_original_path, p0, sample, 2, tmax)

        if n_sources == 4:  # source 3

            target3_mono = np.mean(target3[:, 0:tmax], axis=0)
            soundfile.write(target3_mean_path, target3_mono, fs)

            write_mono(original3_path, stereo_original_path, p0, sample, 3, tmax)

        ### EVALUATE METRICS
        if n_sources == 3:
            sdr, sir, sar = eng.bss_eval3_wrapper(target0_mean_path, target1_mean_path, target2_mean_path, original0_path, original1_path, original2_path, nargout=3)
            sdr_mix, sir_mix, sar_mix = eng.bss_eval3_mix_wrapper(mixture_path, original0_path, original1_path, original2_path, nargout=3)
            
        elif n_sources == 4:
            sdr, sir, sar = eng.bss_eval4_wrapper(target0_mean_path, target1_mean_path, target2_mean_path, target3_mean_path, original0_path, original1_path, original2_path, original3_path, nargout=3)
            sdr_mix, sir_mix, sar_mix = eng.bss_eval4_mix_wrapper(mixture_path, original0_path, original1_path, original2_path, original3_path,  nargout=3)

        # evaluate pesq
        pesq = []
        pesq_mix = []

        pesq0 = eng.pesq(original0_path,target0_mean_path);
        pesq0_mix = eng.pesq(original0_path,mixture_path);

        pesq1 = eng.pesq(original1_path,target1_mean_path);
        pesq1_mix = eng.pesq(original1_path,mixture_path);

        pesq2 = eng.pesq(original2_path,target2_mean_path);
        pesq2_mix = eng.pesq(original2_path,mixture_path);

        if n_sources == 3:
            pesq.append([array(pesq0), array(pesq1), array(pesq2)])
            pesq_mix.append([array(pesq0_mix), array(pesq1_mix), array(pesq2_mix)])

        elif n_sources == 4:
            pesq3 = eng.pesq(original3_path,target3_mean_path);
            pesq3_mix = eng.pesq(original3_path,mixture_path);
            pesq.append([array(pesq0), array(pesq1), array(pesq2), array(pesq3)])
            pesq_mix.append([array(pesq0_mix), array(pesq1_mix), array(pesq2_mix),  array(pesq3_mix)])

        # separated
        sdr_list.append(flat_list(sdr))
        sir_list.append(flat_list(sir))
        sar_list.append(flat_list(sar))
        pesq_list.append(pesq[0])

        sdr_output = output_array(sdr_list)
        sir_output = output_array(sir_list)
        sar_output = output_array(sar_list)
        pesq_output = output_array(pesq_list)

        # mixture
        if baseline in my_methods[:2]:
            sdr_mix_list.append(flat_list(sdr_mix))
            sir_mix_list.append(flat_list(sir_mix))
            sar_mix_list.append(flat_list(sar_mix))
            pesq_mix_list.append(pesq_mix[0])

            sdr_mix_output = output_array(sdr_mix_list)
            sir_mix_output = output_array(sir_mix_list)
            sar_mix_output = output_array(sar_mix_list)
            pesq_mix_output = output_array(pesq_mix_list)

        print(baseline, angle_list, sample, sample % n_mixtures, sound_set, doa, 'SDR=', np.mean(sdr),'SIR=', np.mean(sir), 'SAR=', np.mean(sar), 'PESQ=', np.mean(pesq))

        # fill metrics tensor
        for source in range(0, n_sources):
            metrics_tensor[sound_set, doa, 0, source] =  sdr_output[source]
            metrics_tensor[sound_set, doa, 1, source] =  sir_output[source]
            metrics_tensor[sound_set, doa, 2, source] =  sar_output[source]
            metrics_tensor[sound_set, doa, 3, source] =  pesq_output[source]

            if baseline in my_methods[:2]:
                metrics_mix_tensor[sound_set, doa, 0, source] = sdr_mix_output[source]
                metrics_mix_tensor[sound_set, doa, 1, source] = sir_mix_output[source]
                metrics_mix_tensor[sound_set, doa, 2, source] = sar_mix_output[source]
                metrics_mix_tensor[sound_set, doa, 3, source] = pesq_mix_output[source]


        # increase angle index
        sound_set = sound_set + 1

    if baseline in my_methods[:2]:
        return metrics_tensor, metrics_mix_tensor

    else:
        return metrics_tensor