from multiprocessing.forkserver import connect_to_new_process
import os
import sys
import time
from scipy.io import wavfile
from scipy import signal
from scipy.stats import vonmises
import soundfile
import numpy as np
import matplotlib.pyplot as plt
from stats import RunningStats
import h5py
import librosa
from create_args import *
from prepare_data_Bformat import *


def check_nans(X):
    array_sum = np.sum(X)
    array_has_nan = ~np.isfinite(array_sum)

    if array_has_nan == True:
        # print wrong elements
        #print(np.argwhere(~np.isfinite(X)))
        print(X[~np.isfinite(X)])
        sys.exit(-1)


def mag(x):
   
   x = 20 * np.log10(np.abs(x))
   return x

    
def angles_dist(P0, Gx, Gy):
 
    Y = (np.conj(P0)*Gy).real
    X = (np.conj(P0)*Gx).real
    theta =  np.arctan2( Y,X )
    #theta_deg = (np.degrees(theta)+360)%360
    
    return theta
    

def evaluate_IRM(X,source_index):
    
    IRM = np.abs(X[source_index,:,:,:] ) / np.sum( np.abs(X[:,:,:,:] ),axis=0 ) 
    
    return IRM


def evaluate_IRM_square(X,source_index):
    
    IRM = np.abs(X[source_index,:,:,:] )**2 / np.sum( np.abs(X[:,:,:,:] )**2 ,axis=0 ) 
    
    return IRM

    
def fill_spectrogram_tensor(args, p0, px, py):
    
    # STFT
    [_, _, P0] = signal.stft(p0, args.fs, args.window, args.Wlength, args.overlap, args.fft_size)
    [_, _, Gx] = signal.stft(px, args.fs, args.window, args.Wlength, args.overlap, args.fft_size)
    [_, _, Gy] = signal.stft(py, args.fs, args.window, args.Wlength, args.overlap, args.fft_size)
    
    # fill tensor
    X = np.zeros([P0.shape[0],P0.shape[1],3], np.complex64)

    X[:,:,0] = P0
    X[:,:,1] = Gx
    X[:,:,2] = Gy

    return X


def extract_features(args, X):

    # define angular feature
    theta_deg = angles_dist(X[:,:,0], X[:,:,1], X[:,:,2])

    # Define LPS
    Gx = mag(X[:,:,1]+1e-16) # add 1e-16 to avoid nans
    Gy = mag(X[:,:,2]+1e-16)

    return theta_deg, Gx, Gy

    
def prepare_input_Bformat(args):

    # running mean and std
    if args.task == 'train':
        runstats = RunningStats(args.n_freq, np.float64)
        stats = []

    ### CREATE MIXTURES
    data_list = prepare_data_Bformat(args)


    ### INPUT DATA GENERATION
    print("Filling data tensors", time.ctime())

    for mix_index in range(0, len(data_list)):
        
        print(mix_index)
        [p0, vel, angles_labels] = data_list[mix_index]
        
        # create mixtures
        p0_mix = np.sum(p0,axis=0)
        vel_mix = np.sum(vel,axis=0)

        # pressure tensors mixtures
        X = fill_spectrogram_tensor(args, p0_mix, vel_mix[:,0], vel_mix[:,1])
        if X.shape[1] >= args.min_frames:
            X = X[:, :args.min_frames, :] # cut frames to match paper
        else:
            # copy the fist frames to fill the last time frames
            len_x_frames = X.shape[1]
            X = np.resize(X, (X.shape[0], args.min_frames, X.shape[2]))
            X[:, len_x_frames:, :] = X[:, :(args.min_frames-len_x_frames), :]
                
        # create features
        theta_deg, Gx, Gy = extract_features(args, X)

        # initialize tensor for all sources
        X_all = np.zeros([args.n_sources, args.n_freq, args.min_frames, 2], np.complex64)

        if args.task == 'test':
            X_or = np.zeros([args.n_sources, args.n_freq, args.min_frames, 3], np.complex64)

        for source_index in range(0, args.n_sources):
            
            # sources pressures
            X_source = fill_spectrogram_tensor(args, p0[source_index,:], vel[source_index,:,0], vel[source_index,:,1])
            
            if X_source.shape[1] >= args.min_frames:
                X_source = X_source[:, :args.min_frames, :]
            else:
                len_x_frames = X_source.shape[1]
                X_source = np.resize(X_source, (X_source.shape[0], args.min_frames, X_source.shape[2]))
                X_source[:, len_x_frames:, :] = X_source[:, :(args.min_frames-len_x_frames), :]


            # fill X_all with only two channels
            X_all[source_index,:,:,:] = X_source[:,:,1:3]

            if args.task == 'test':
                #pressure_tensor[mix_index,source_index,:,:,:] = X_source
                X_or[source_index, :, :, :] = X_source
            
            # store audio mixtures
            data = np.array([p0_mix, vel_mix[:,0], vel_mix[:,1]]).T



            if args.task == 'val':
                soundfile.write(os.path.join(args.data_val_path, str(mix_index) + '.wav'), data , args.fs)
                

        # create IRM
        IRM = []
        for source_index in range(0,args.n_sources):
                
            IRM_source = np.mean(evaluate_IRM(X_all, source_index),axis=2)
            IRM.append(IRM_source)
            
            # plt.imshow(IRM, cmap='jet', aspect='auto', interpolation='none')
            # clb = plt.colorbar()
            # plt.tick_params(labelsize=28)
            # plt.xlabel('Time', fontsize=36)
            # plt.ylabel('Frequency', fontsize=36)
            # clb.ax.tick_params(labelsize=28) 
            # plt.title('Spectrogram: ' + '$G_y$', fontsize=42)
            # plt.gca().invert_yaxis()
            # plt.savefig(os.path.join(args.figures_path, 'IRM' + str(mix_index) + '_' + str(source_index)),bbox_inches='tight')
            # plt.clf()
    	    
        ### PLOTS
        # title_size = 57 #42
        # xy_size = 51  #36
        # tick_size = 43 #28

        # fig,ax = plt.subplots(1,figsize=(14,14))
        # theta_flat = theta_deg.flatten()
        # n, bins, patches = plt.hist(theta_flat, bins=72, facecolor='blue', alpha=0.5, edgecolor='black', linewidth=1.2)
        # plt.tick_params(labelsize=tick_size)
        # plt.xticks(np.arange(-3, 3+1, 1.0))
        # plt.xlabel(r'$\Theta$ (rad)', fontsize=xy_size)
        # plt.ylabel(r'$\Theta$ count', fontsize=xy_size)
        # plt.title('Angular feature: $\Theta$', fontsize=title_size)
        # plt.savefig(os.path.join(args.figures_path, 'theta_deg' + str(mix_index)),bbox_inches='tight')
        # plt.clf()
        
        # plt.imshow(theta_deg, cmap='jet', aspect='auto', interpolation='none')
        # clb = plt.colorbar()
        # clb.ax.set_ylabel('rad', rotation=90, fontsize=xy_size)
        # plt.tick_params(labelsize=tick_size)
        # plt.xlabel('Time', fontsize=xy_size)
        # plt.ylabel('Frequency', fontsize=xy_size)
        # clb.ax.tick_params(labelsize=tick_size) 
        # plt.title('Distribution  of angles', fontsize=title_size)
        # plt.gca().invert_yaxis()
        # plt.savefig(os.path.join(args.figures_path, 'theta_tensor' + str(mix_index)  ), bbox_inches='tight')
        # plt.clf()

        
        # plt.imshow(Gx, cmap='jet', aspect='auto', interpolation='none')
        # clb = plt.colorbar()
        # plt.tick_params(labelsize=tick_size)
        # plt.xlabel('Time', fontsize=xy_size)
        # plt.ylabel('Frequency', fontsize=xy_size)
        # clb.ax.tick_params(labelsize=tick_size) 
        # plt.title('Spectrogram: ' + '$G_x$', fontsize=title_size)
        # plt.gca().invert_yaxis()
        # plt.savefig(os.path.join(args.figures_path, 'Gx' + str(mix_index)  ),bbox_inches='tight' )
        # plt.clf()

        
        # plt.imshow(Gy, cmap='jet', aspect='auto', interpolation='none')
        # clb = plt.colorbar()
        # plt.tick_params(labelsize=tick_size)
        # plt.xlabel('Time', fontsize=xy_size)
        # plt.ylabel('Frequency', fontsize=xy_size)
        # clb.ax.tick_params(labelsize=tick_size) 
        # plt.title('Spectrogram: ' + '$G_y$', fontsize=title_size)
        # plt.gca().invert_yaxis()
        # plt.savefig(os.path.join(args.figures_path, 'Gy' + str(mix_index)  ),bbox_inches='tight')
        # plt.clf()

        
        if args.task == 'train':
            # update stats
            runstats.update(np.hstack([theta_deg, Gx, Gy ]).T )

            # stats
            mean = np.zeros([args.n_freq])
            std = np.zeros([args.n_freq])
            
            mean =  runstats.stats['mean']
            std = np.sqrt(runstats.stats['var'])


        ### LOAD STATS FOR VAL
        if (args.task == 'val' or args.task == 'test' and mix_index == 0):
            # use train mean and std
            with h5py.File(args.save_stats_data_path, 'r') as hf:
                stats = hf.get('stats')
                stats = np.array(stats)
                mean = stats[0][0]
                std = stats[0][1]  
        
        # normalization
        for f in range(0,len(theta_deg )):
            theta_deg[f,:] = (theta_deg[f,:]-mean[f])/std[f]
            Gx[f,:] = (Gx[f,:]-mean[f])/std[f]
            Gy[f,:] = (Gy[f,:]-mean[f])/std[f]

        features_all = [theta_deg, Gx, Gy]


        ### CHECK AFTER NORMALIZATION
        # theta_list = theta_deg.ravel()
        # nbins = 72
        # step=0.1
        # start = np.floor(min(theta_list) / step) * step
        # stop = max(theta_list) + step

        # bin_edges = np.arange(start, stop, step=step)

        # fig,ax = plt.subplots(1,figsize=(14,14))
        # #N, bins, patches = ax.hist(data,bins=[0.3, 0.8,1.3,1.8,2.3,3.5],edgecolor='white')
        # n, bins, patches = ax.hist(theta_list, bin_edges) #N, bins, patches = ax.hist(data,bins=[0.3, 0.8,1.3,1.8,2.3,3.5],edgecolor='white')
        # plt.tick_params(labelsize=28)
        # #   plt.xticks(bin_edges, rotation=90)
        # plt.xlabel(r'$\theta$ (degrees)', fontsize=36)
        # plt.ylabel(r'$\theta$ count', fontsize=36)
        # plt.title('Distribution of angles', fontsize=42)
        # plt.savefig(os.path.join(args.figures_path, 'test_theta_deg' + str(mix_index)),bbox_inches='tight')
        # plt.clf()


        # plt.imshow(theta_deg, cmap='jet', aspect='auto', interpolation='none')
        # clb = plt.colorbar()
        # plt.tick_params(labelsize=28)
        # plt.xlabel('Time', fontsize=36)
        # plt.ylabel('Frequency', fontsize=36)
        # clb.ax.tick_params(labelsize=28) 
        # plt.title('Distribution  of angles', fontsize=42)
        # plt.savefig(os.path.join(args.figures_path, 'test_theta' + str(mix_index)), bbox_inches='tight')
        # plt.clf()
        # print(os.path.join(args.figures_path, 'test_theta' ))

        # plt.imshow(Gx, cmap='jet', aspect='auto', interpolation='none')
        # clb = plt.colorbar()
        # plt.tick_params(labelsize=28)
        # plt.xlabel('Time', fontsize=36)
        # plt.ylabel('Frequency', fontsize=36)
        # clb.ax.tick_params(labelsize=28) 
        # plt.title('Distribution  of angles', fontsize=42)
        # plt.savefig(os.path.join(args.figures_path, 'test_Gx' + str(mix_index)), bbox_inches='tight')
        # plt.clf()
        

        # save data
        if args.task == 'train':
            save_path = args.save_train_data_path
        
        elif args.task == 'val':
            save_path = args.save_val_data_path

        elif args.task == 'test':
            save_path = args.save_test_data_path

        # save data
        if args.task == 'train' or args.task == 'val':

            with h5py.File(os.path.join(save_path, str(mix_index) + '.h5'), 'w') as hf:
                hf.create_dataset('data', data=features_all)
                hf.create_dataset('IRM', data=IRM)
        
        elif args.task == 'test':

            with h5py.File(save_path, 'w') as hf:
                hf.create_dataset('data', data=features_all)
                hf.create_dataset('spectrograms_mixture', data=X[:,:,1:3])
                hf.create_dataset('spectrograms_original', data=X_or)
                # hf.create_dataset('labels', data=angles_labels)

        mix_index += 1

        #if mix_index == 5:
        #    break

    if args.task == 'train':
        stats.append([mean,std])
        print(mean, std)

        # save data
        with h5py.File(args.save_stats_data_path, 'w') as hf:
            hf.create_dataset('stats', data=stats)

        
    print("Done!.", time.ctime())
   
   
if __name__ == '__main__':
    
    args, config = create_args()

    # main paths
    if args.task == 'train':
        args.figures_path = os.path.join(args.WORK_PATH, 'figures/train')
    elif args.task == 'val':
        args.figures_path = os.path.join(args.WORK_PATH, 'figures/val')
    elif args.task == 'test':
        args.figures_path = os.path.join(args.WORK_PATH, 'figures/test')

    # create path if does not exist
    if not os.path.exists(args.figures_path):
        os.makedirs(args.figures_path)

    print(args.save_train_data_path)
    if not os.path.exists(args.save_train_data_path):
        os.makedirs(args.save_train_data_path)
    if not os.path.exists(args.save_val_data_path):
        os.makedirs(args.save_val_data_path)
    if not os.path.exists(args.save_test_data_path):
        os.makedirs(args.save_test_data_path)

    # audio mixtures paths
    if not os.path.exists(args.data_val_path):
        os.makedirs(args.data_val_path)

    if not os.path.exists(args.data_test_path):
        os.makedirs(args.data_test_path)

    prepare_input_Bformat(args)
