#python run_test_mask.py B_format train 12BB01 12BB01 ['theta','MV'] '' ''
import sys
import numpy as np
from scipy import signal
import pickle
import os
import time
import matplotlib.pyplot as plt
import random
import features
import utilities
import timeit
import h5py
from train_utilities import expand_source
import test_NN_mask
from generate_spectrograms import *
# from evaluate_performance_stereo import *
# from evaluate_performance_mono import *
# from evaluate_performance_mono_spec import *
from keras.models import load_model
from draw_masks import draw_masks, draw_masks4
from draw_separated_spectrograms import draw_separated_spectrograms
sys.path.append("..")
from prepare_inputs_Bformat import evaluate_IRM, evaluate_IRM_square
import matplotlib
matplotlib.use('Agg')

# audio parameters
fs = 16000
Wlength = 2048
n_channel = 128
n_softmax = 36
window = 'hann'
window_size = Wlength#1024
hop_size = 512 #512
overlap = Wlength*3/4
fft_size = Wlength

# activate part of code
stereo_mode = 'monospec'

eval_IRMs = True
eval_NNs = True
eval_baselines = True

# custom parameters
DNN_type = 'DNN'
n_sources = 4

# lists
IRMs_list = ['IRM_mono', 'IRM_mono_square', 'IRM_stereo', 'IRM_stereo_square']
#IRMs_list = ['IRM_mono']
baseline_list = ['XCWW_' + str(n_sources) + 'src', 'Banu_' + str(n_sources) + 'src', 'shujau_' + str(n_sources) + 'src']


# define various IRMs
def IRM_mono(X, mix_index, source_index):
    return np.mean(evaluate_IRM(X[mix_index,:,:,:,1:3], source_index), axis = 2)

def IRM_mono_square(X, mix_index, source_index):
    return np.mean(evaluate_IRM_square(X[mix_index,:,:,:,1:3], source_index), axis = 2)

def IRM_stereo(X, mix_index, source_index):    
    return evaluate_IRM(X[mix_index,:,:,:,1:3], source_index)

def IRM_stereo_square(X, mix_index, source_index):    
    return evaluate_IRM_square(X[mix_index,:,:,:,1:3], source_index)


# calculate
def calculate_IRMs(X, label, n_sources):

    # initialize
    if label == 'IRM_mono' or label == 'IRM_mono_square':
        IRM_tensor = np.zeros([len(X), n_sources, X_mix.shape[1], X_mix.shape[2]])
    
    elif label == 'IRM_stereo' or label == 'IRM_stereo_square': 
        IRM_tensor = np.zeros([len(X), n_sources, X_mix.shape[1], X_mix.shape[2], 2])

    
    # loop
    for mix_index in range(len(X)):

        for source_index in range(n_sources):
            
            if label == 'IRM_mono':
                IRM_tensor[mix_index,source_index,:,:] = IRM_mono(X, mix_index, source_index)

            elif label == 'IRM_mono_square': 
                IRM_tensor[mix_index,source_index,:,:] = IRM_mono_square(X, mix_index, source_index)

            elif label == 'IRM_stereo':
                IRM_tensor[mix_index,source_index,:,:,:] = IRM_stereo(X, mix_index, source_index)
                
            elif label == 'IRM_stereo_square':
                IRM_tensor[mix_index,source_index,:,:,:] = IRM_stereo_square(X, mix_index, source_index)
                
    # swap axes
    if label == 'IRM_mono' or label == 'IRM_mono_square':

        IRM_tensor = np.einsum('jklm->jlmk', IRM_tensor)

    elif label == 'IRM_stereo' or label == 'IRM_stereo_square': 

        IRM_tensor = np.einsum('jklmn->jlmkn', IRM_tensor)

    return IRM_tensor


# run test function    
def run_test():

    # load dataset
    if 'theta' in features_name and 'MV' not in features_name:
        Xtest = x[:,:,:,0]
        Xtest = Xtest.reshape(Xtest.shape[0],Xtest.shape[1],Xtest.shape[2],1)

    elif 'MV' in features_name and 'theta' not in features_name:
        Xtest = np.zeros([x.shape[0],x.shape[1],x.shape[2],2])
        Xtest = x[:,:,:,1:3]
        #Xtest = x[:,:,:,1:3]

    elif 'theta' in features_name and 'MV' in features_name:
        Xtest = x

    # resize spectrograms to a more suitable shape for computation
    X_or_reshape = np.einsum('abcde->acdbe', X_or)

    ### evaluate IRMS
    if eval_IRMs == True:

        for label in IRMs_list:
            sound_IRMs_path = os.path.join(os.getcwd(), 'audio', label, n_sources_string)

            if not os.path.exists(sound_IRMs_path):
                os.makedirs(sound_IRMs_path)

            # create IRM tensor
            IRM_tensor = calculate_IRMs(X_or, label, n_sources)

            # generate spectrograms
            target_separated_IRM = generate_spectrograms(X_or_reshape, IRM_tensor, sound_IRMs_path)

            # evaluate metrics
            if stereo_mode == 'stereo':
                save_metrics_IRM_path = os.path.join(save_metrics_folder_path, label + '_' + suffix + '_' + task + '_'  +'TestRoom' + Test_Room + post_suffix + DNN_suffix + '.h5')
                metrics_tensor_IRM, _ = evaluate_performance_stereo(matlab_path, X_or_reshape, target_separated_IRM, label, n_sources_string)
            elif stereo_mode == 'mono':
                save_metrics_IRM_path = os.path.join(save_metrics_folder_path, label + '_' + suffix + '_' + task + '_'  +'TestRoom' + Test_Room + post_suffix + DNN_suffix + '_mono' + '.h5')
                metrics_tensor_IRM = evaluate_performance_mono(matlab_path, X_or_reshape, target_separated_IRM, label, n_sources_string)
            elif stereo_mode == 'monospec':
                save_metrics_IRM_path = os.path.join(save_metrics_folder_path, label + '_' + suffix + '_' + task + '_'  +'TestRoom' + Test_Room + post_suffix + DNN_suffix + '_monospec' + '.h5')
                metrics_tensor_IRM = evaluate_performance_mono_spec(matlab_path, X_or_reshape, target_separated_IRM, label, n_sources_string)

            with h5py.File(save_metrics_IRM_path, 'w') as hf:
                hf.create_dataset('metrics_tensor', data=metrics_tensor_IRM)
    
    
    ### evaluate NNs
    if eval_NNs == True:
        # paths
        print(load_NN_name, n_sources_string)

        # predict masks
        masks = test_NN_mask.test_NN(Xtest, load_NN_name, features_name)

        # generate spectrograms
        target_separated = generate_spectrograms(X_or_reshape, masks, sound_path)

        save_metrics_path = os.path.join(save_metrics_folder_path, 'Zermini_metrics_' + suffix + '_' + task + '_'  +'TestRoom' + Test_Room + post_suffix + DNN_suffix + '_' + DNN_type + '.h5')

        # evaluate metrics
        if stereo_mode == 'stereo':
            metrics_tensor, metrics_mix_tensor = evaluate_performance_stereo(matlab_path, X_or_reshape, target_separated, DNN_type, n_sources_string)

        elif stereo_mode == 'mono':
            save_metrics_path = save_metrics_path.replace('.h5', '_mono.h5')
            metrics_tensor, metrics_mix_tensor = evaluate_performance_mono(matlab_path, X_or_reshape, target_separated, DNN_type, n_sources_string)
        
        elif stereo_mode == 'monospec':
            save_metrics_path = save_metrics_path.replace('.h5', '_monospec.h5')
            metrics_tensor, metrics_mix_tensor = evaluate_performance_mono_spec(matlab_path, X_or_reshape, target_separated, DNN_type, n_sources_string)

 
        with h5py.File(save_metrics_path, 'w') as hf:
            hf.create_dataset('metrics_tensor', data=metrics_tensor)
            hf.create_dataset('metrics_mix_tensor', data=metrics_mix_tensor)

    
    ### evaluate baselines
    if eval_baselines == True:
        # baselines
        for baseline in baseline_list:
            
            if stereo_mode == 'stereo':
                save_metrics_mat_path = os.path.join(save_metrics_folder_path, baseline + '_metrics_' + suffix + '_' + task + '_'  +'TestRoom' + Test_Room + post_suffix + DNN_suffix + '.h5')
                metrics_mat_tensor = evaluate_performance_stereo(matlab_path, X_or_reshape, None, baseline, n_sources_string)
            elif stereo_mode == 'mono':
                save_metrics_mat_path = os.path.join(save_metrics_folder_path, baseline + '_metrics_' + suffix + '_' + task + '_'  +'TestRoom' + Test_Room + post_suffix + DNN_suffix + '_mono' + '.h5')
                metrics_mat_tensor = evaluate_performance_mono(matlab_path, X_or_reshape, None, baseline, n_sources_string)
            elif stereo_mode == 'monospec':
                save_metrics_mat_path = os.path.join(save_metrics_folder_path, baseline + '_metrics_' + suffix + '_' + task + '_'  +'TestRoom' + Test_Room + post_suffix + DNN_suffix + '_monospec' + '.h5')
                metrics_mat_tensor = evaluate_performance_mono_spec(matlab_path, X_or_reshape, None, baseline, n_sources_string)
            
            with h5py.File(save_metrics_mat_path, 'w') as hf:
                hf.create_dataset('metrics_tensor', data=metrics_mat_tensor)
        
    ### draw masks
    # if n_sources == 4:
    #     draw_masks4(masks, IRM_tensor, figures_IRMs_path)

    

    print("Done!", time.ctime())

if __name__ == "__main__":

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

        # load input path
        load_results_path = os.path.join(project_path, 'Results/Room' + Train_Room)
        load_TestData_path = os.path.join(project_path, 'InputData/TestData')
        load_room_path = os.path.join(load_TestData_path, 'Room' + Test_Room)
        load_data_path = os.path.join(load_room_path, 'test_'  +'TestRoom' + Test_Room + '_' + str(n_sources_string) + 'Sources'+ post_suffix +'.h5')

        # save output path
        save_output_path = os.path.join(load_results_path, 'output_tensor_' + suffix + '_' + 'test' + '_'  +'TestRoom' + Test_Room + '_' + n_sources_string + 'Sources' + post_suffix + DNN_suffix  + '.h5')

        # save masks path
        save_mask_spatial_path = os.path.join(load_results_path, 'mask_spatial_' + suffix + '_' + 'test' + '_'  +'TestRoom' + Test_Room + post_suffix + DNN_suffix + '.h5')

        # save metrics path
        save_metrics_folder_path = os.path.join(load_results_path, 'metrics', n_sources_string + 'Sources')

        # load spatial data
        print("Loading spatial input:\n.", time.ctime(), load_data_path)


        with h5py.File(load_data_path, 'r') as hf:
           x = hf.get('data')
           X_mix = hf.get('spectrograms_mixture')
           X_or = hf.get('spectrograms_original')
           # labels = hf.get('labels')

           x = np.array(x)
           X_mix = np.array(X_mix)
           X_or = np.array(X_or)
           # labels = np.array(labels)



        # matlab path
        matlab_path = '/vol/vssp/mightywings/B_format/matlab_code/mat_files/sepresults'

        # save sound path
        sound_path = os.path.join(os.getcwd(), 'Sounds', suffix + post_suffix + DNN_type,  n_sources_string ) 
        load_NN_name = os.path.join(project_path, 'Results/Room' + Train_Room, DNN_type + '_models_' + suffix + '_' + task + DNN_suffix + '_' + n_sources_string + '.h5')
        figures_path = os.path.join(os.getcwd(), 'Figures', suffix + post_suffix + DNN_suffix, n_sources_string )
        figures_IRMs_path = os.path.join(os.getcwd(), 'Figures', 'IRMs', n_sources_string)

        # create folders if they don't exist yet
        if not os.path.exists(project_path):
            os.makedirs(project_path)
        if not os.path.exists(sound_path):
            os.makedirs(sound_path)
        if not os.path.exists(figures_path):
            os.makedirs(figures_path)
        if not os.path.exists(figures_IRMs_path):
            os.makedirs(figures_IRMs_path)
        if not os.path.exists(save_metrics_folder_path):
            os.makedirs(save_metrics_folder_path)

        run_test()

    else:
        raise Exception("Error, please double check the arguments.")
        sys.exit()
