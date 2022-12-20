#python evaluate_sr.py B_format train 12BB01 12BB01 ['theta','MV'] '' ''
import numpy as np
import os
import time
import sys
from separation import bss_eval_sources
from scipy import signal
import time
import h5py
from scipy.io import wavfile
import matplotlib.pyplot as plt
import speech_recognition as sr
import soundfile as sf

# STFT params
Wlength = 2048
window = 'hann'
window_size = Wlength#1024
hop_size = 512 #512
overlap = window_size - hop_size
fft_size = Wlength
fs = 16000
n_mixtures=20 #!!!!
n_sources = 3


# angles list
angles_list = [ [0,10,20,30], [0,20,40,60], [0,30,60,90], [0,40,80,120], [0,50,100,150],  [0,60,120,180], [0,70,140,210], [0,80,160,240], [0,90,180,270]  ]

# baselines list
baseline_list = ['DNN_MV']
#baseline_list = ['DNN', 'CNN', 'IRM_mono', 'XCWW_' + str(n_sources) + 'src', 'Banu_' + str(n_sources) + 'src', 'shujau_' + str(n_sources) + 'src', 'Mixtures', 'Original']

# number of sources
if n_sources == 2:
    n_sources_string = 'Two'

elif n_sources == 3:
    n_sources_string = 'Three'

elif n_sources == 4:
    n_sources_string = 'Four'

# matlab paths
matlab_path = os.path.join('/vol/vssp/mightywings/B_format/matlab_code/mat_files/sepresults', n_sources_string + 'Sources')

def evaluate_sr(baseline):

    # baseline path
    baseline_path = os.path.join(matlab_path, baseline)

    print(baseline_path)

    # if n_sources == 3:
    #     count_folder = os.path.join(my_path, '0_10_20')
    # elif n_sources == 4:
    #     count_folder = os.path.join(my_path, '0_10_20_30')
    # count = len([name for name in os.listdir(count_folder) if os.path.isfile(os.path.join(count_folder, name))])//n_sources
    n_samples = n_mixtures * len(angles_list)

    # define sdrs tensor shape
    metrics_tensor = np.zeros([n_mixtures, int(n_samples/n_mixtures), 1, n_sources ]  )

    sound_set = doa = 0

    for sample in range(0, n_samples):

        print(sample)

        if sample % n_mixtures == 0 and sample != 0 and sample != n_samples:

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


            angles_folder = os.path.join(baseline_path, '_'.join(angle_list)  )
            print(angle_list)

        elif  sample == 0:

            angle0 = angles_list[doa][0]
            angle1 = angles_list[doa][1]
            angle2 = angles_list[doa][2]
            angle3 = angles_list[doa][3]

            if n_sources == 3:
                angle_list = [str(angle0), str(angle1), str(angle2)]
            elif n_sources == 4:
                angle_list = [str(angle0), str(angle1), str(angle2), str(angle3)]


            angles_folder = os.path.join(baseline_path, '_'.join(angle_list)  )
            print(angle_list)


        # angles_path
        angles_path = os.path.join(baseline_path, angles_folder)
        stereo_path = os.path.join(angles_path, 'stereo')

        # separated target

        if baseline != 'Original' and  baseline != 'Mixtures':

            target0_path = os.path.join(angles_path, 'Est_s'+ str(0) + '_' + str(sample%n_mixtures) + '_monospec.wav')
            target1_path = os.path.join(angles_path, 'Est_s'+ str(1) + '_' + str(sample%n_mixtures) + '_monospec.wav')
            target2_path = os.path.join(angles_path, 'Est_s'+ str(2) + '_' + str(sample%n_mixtures) + '_monospec.wav')

            if n_sources == 4:
                target3_path = os.path.join(angles_path, 'Est_s'+ str(3) + '_' + str(sample%n_mixtures) + '_monospec.wav')

        elif baseline == 'Original':
            target0_path = os.path.join(angles_path, 'original'+ str(0) + str(sample%n_mixtures) + '_monospec.wav')
            target1_path = os.path.join(angles_path, 'original'+ str(1) + str(sample%n_mixtures) + '_monospec.wav')
            target2_path = os.path.join(angles_path, 'original'+ str(2) + str(sample%n_mixtures) + '_monospec.wav')

            if n_sources == 4:
                target3_path = os.path.join(angles_path, 'original'+ str(3) + '' + str(sample%n_mixtures) + '_monospec.wav')

        elif baseline == 'Mixtures':
            target0_path = os.path.join(angles_path, 'mixture' + str(sample%n_mixtures) + '_monospec.wav')
            target1_path = os.path.join(angles_path, 'mixture' + str(sample%n_mixtures) + '_monospec.wav')
            target2_path = os.path.join(angles_path, 'mixture' + str(sample%n_mixtures) + '_monospec.wav')

            if n_sources == 4:
                target3_path = os.path.join(angles_path, 'mixture' + str(sample%n_mixtures) + '_monospec.wav')


        ### source 1
        r = sr.Recognizer()
        with sr.WavFile(target0_path) as source:
            audio = r.record(source)

        fs, data = wavfile.read(target0_path)
        speech_recognition_path = os.path.join(matlab_path, 'speech_recognition')
        if not os.path.exists(speech_recognition_path):
            os.makedirs(speech_recognition_path)
        
        target_path = os.path.join(speech_recognition_path, 'speech_' + str(sound_set) )
        print(target_path)
        
        # remove target if exists
        if os.path.exists(target_path):
            os.remove(target_path)

        wavfile.write(target_path,fs,data)
        
        try:
            list = r.recognize_google(audio, None, "en-US", True)
            print(list)
            rate = list['alternative'][0]['confidence']*100
            metrics_tensor[sound_set, doa, 0, 0] = rate
            print(metrics_tensor[sound_set, doa, 0, 0])

        except Exception:
            metrics_tensor[sound_set, doa, 0, 0] = 0.0
            print(metrics_tensor[sound_set, doa, 0, 0])

        ### source 2
        r = sr.Recognizer()
        with sr.WavFile(target1_path) as source:
            audio = r.record(source)
        try:
            list = r.recognize_google(audio, None, "en-US", True)
            rate = list['alternative'][0]['confidence']*100
            metrics_tensor[sound_set, doa, 0, 1] = rate
            print(metrics_tensor[sound_set, doa, 0, 1])

        except Exception:
            metrics_tensor[sound_set, doa, 0, 1] = 0.0
            print(metrics_tensor[sound_set, doa, 0, 1])

        ### source 3
        r = sr.Recognizer()
        with sr.WavFile(target2_path) as source:
            audio = r.record(source)
        try:
            list = r.recognize_google(audio, None, "en-US", True)
            rate = list['alternative'][0]['confidence']*100
            metrics_tensor[sound_set, doa, 0, 2] = rate
            print(metrics_tensor[sound_set, doa, 0, 2])

        except Exception:
            metrics_tensor[sound_set, doa, 0, 2] = 0.0
            print(metrics_tensor[sound_set, doa, 0, 2])

        ### source 4
        if n_sources == 4:
            r = sr.Recognizer()
            with sr.WavFile(target3_path) as source:
                audio = r.record(source)
            try:
                list = r.recognize_google(audio, None, "en-US", True)
                rate = list['alternative'][0]['confidence']*100
                metrics_tensor[sound_set, doa, 0, 3] = rate
                print(metrics_tensor[sound_set, doa, 0, 3])

            except Exception:
                metrics_tensor[sound_set, doa, 0, 3] = 0.0
                print(metrics_tensor[sound_set, doa, 0, 3])

        # increase angle index
        sound_set = sound_set + 1

    return metrics_tensor

if __name__ == '__main__':

    if sys.argv[1] != "" and sys.argv[2].lower() != '' and sys.argv[3] in ['1','12BB01'] and sys.argv[4] in ['1','12BB01'] and set(['theta', 'MV']).issuperset(set(sys.argv[5].replace("[","").replace("]","").split(","))):

        print(sys.argv)
        DNN_name = sys.argv[1]
        task = sys.argv[2].lower()
        Train_Room = sys.argv[3]
        Test_Room = sys.argv[4]
        features_name = sys.argv[5]
        post_suffix = sys.argv[6]
        DNN_suffix = sys.argv[7]

        if post_suffix != '':
            post_suffix = '_' + post_suffix

        if DNN_suffix != '':
            DNN_suffix = '_' + DNN_suffix

        # number of sources
        if n_sources == 2:
            n_sources_string = 'Two'

        elif n_sources == 3:
            n_sources_string = 'Three'

        elif n_sources == 4:
            n_sources_string = 'Four'

        # get dataset name
        suffix =  ( (features_name.replace(",","_") ).replace("[","") ).replace("]","")

        # current folder
        project_path = os.path.join('/vol/vssp/mightywings/' + DNN_name)
        print(project_path)

        # load input path
        load_results_path = os.path.join(project_path, 'Results/Room' + Train_Room)
        save_metrics_folder_path = os.path.join(load_results_path, 'metrics', n_sources_string + 'Sources')


        ### iterate on methods
        for baseline in baseline_list:

            metrics_tensor = evaluate_sr(baseline)

            save_metrics_path = os.path.join(save_metrics_folder_path, baseline + '_speech_recognition_' + suffix + '_' + task + '_'  +'TestRoom' + Test_Room + post_suffix + DNN_suffix + '_monospec.h5')

            with h5py.File(save_metrics_path, 'w') as hf:
                hf.create_dataset('metrics_tensor_sr', data=metrics_tensor)
