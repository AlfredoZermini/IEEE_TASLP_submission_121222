#python plot_metrics_comparison.py B_format train 12BB01 12BB01 ['theta','MV'] '' ''
import sys
import numpy as np
import os
import time
import h5py
import math
import matplotlib.pyplot as plt
import matplotlib

# parameters
n_sources = 3
DNN_type = 'DNN'
stereo_mode = 'monospec'


### PLOT CURRENT METHOD WITH BASELINES
def plot_metrics_baselines(name,min,max):

    fig,ax = plt.subplots(1,figsize=(10,7))
    x0 = np.arange(0,metrics_all.shape[2])
    y0 = np.mean(np.mean(metrics_all[0,:,:,:], axis=0), axis =1)
    e0 = np.std(np.std(metrics_all[0,:,:,:], axis=0), axis =1)
    g0 = ax.errorbar(x0, y0, e0, linestyle='None', marker='o', color='r', elinewidth=0.0001, markersize='6.0', label="Spectrograms")
    ax.plot(x0[0:], y0[0:], '-o', color='r',  linewidth=1, markersize='6.0')
    ax.plot(x0[0:], y0[0:]+e0[0:], '--', linewidth=1, color='r')
    ax.plot(x0[0:], y0[0:]-e0[0:], '--', linewidth=1, color='r')

    x1 = np.arange(0,metrics_all.shape[2])
    y1 = np.mean(np.mean(metrics_all[1,:,:,:], axis=0), axis =1)
    e1 = np.std(np.std(metrics_all[1,:,:,:], axis=0), axis =1)
    g1 = ax.errorbar(x1, y1, e1, linestyle='None', marker='o', color='k', elinewidth=0.0001, markersize='6.0', label="Angles distributions")
    ax.plot(x1[0:], y1[0:], '-o', color='k',  linewidth=1, markersize='6.0')
    ax.plot(x1[0:], y1[0:]+e1[0:], '--', linewidth=1, color='k')
    ax.plot(x1[0:], y1[0:]-e1[0:], '--', linewidth=1, color='k')

    x2 = np.arange(0,metrics_all.shape[2])
    y2 = np.mean(np.mean(metrics_all[2,:,:,:], axis=0), axis =1)
    e2 = np.std(np.std(metrics_all[2,:,:,:], axis=0), axis =1)
    g2 = ax.errorbar(x2, y2, e2, linestyle='None', marker='o', color='#2bafff', elinewidth=0.0001, markersize='6.0', label="Spectrograms and angles distributions")
    ax.plot(x2[0:], y2[0:], '-o', color='#2bafff',  linewidth=1, markersize='6.0')
    ax.plot(x2[0:], y2[0:]+e2[0:], '--', linewidth=1, color='#2bafff')
    ax.plot(x2[0:], y2[0:]-e2[0:], '--', linewidth=1, color='#2bafff')



    handles, labels = ax.get_legend_handles_labels()
    # remove the errorbars
    handles = [h[0] for h in handles]
    legend = ax.legend(handles, labels, loc='upper right',numpoints=1)
    legend.get_title().set_fontsize(20)
    ax.grid(color='#808080', linestyle=':', linewidth=0.5)
    plt.xlim( -1, metrics_all.shape[2])
    plt.ylim( min, max-0.5)
    locs, labels = plt.xticks([0,1,2,3,4,5,6,7,8], ["10$^\circ$", "20$^\circ$","30$^\circ$","40$^\circ$","50$^\circ$","60$^\circ$","70$^\circ$","80$^\circ$","90$^\circ$" ])
    locs, labels = plt.yticks( [i for i in np.arange(min,max,1.)] )
    if name == 'PESQ':
        locs, labels = plt.yticks( [i for i in np.arange(min,max,0.25)] )
    plt.tick_params(labelsize=18)
    plt.xlabel(r'$\Delta\theta$', fontsize=20)
    plt.ylabel(name, fontsize=20)

    # plot and save
    plt.savefig(os.path.join( os.getcwd(), figures_path, 'Comparison_' + name + DNN_suffix) )
    plt.title('%s comparison' %(name), fontsize=30)
    plt.show()


### MAIN
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

        # get dataset name
        suffix =  ( (features_name.replace(",","_") ).replace("[","") ).replace("]","")

        # number of sources
        if n_sources == 2:
            n_sources_string = 'Two'

        elif n_sources == 3:
            n_sources_string = 'Three'

        elif n_sources == 4:
            n_sources_string = 'Four'

        # paths
        project_path = os.path.join('/vol/vssp/mightywings/', DNN_name)
        figures_path = os.path.join(os.getcwd(), 'Figures', suffix + post_suffix + DNN_suffix + '_' +n_sources_string )

        if project_path.startswith('DNN_'):
            project_path = project_path[len('DNN_'):]

        if post_suffix != '':
            post_suffix = '_' + post_suffix

        if DNN_suffix != '':
            DNN_suffix = '_' + DNN_suffix

        # create folders
        if not os.path.exists(figures_path):
            os.makedirs(figures_path)

        # baselines
        features_list = ['MV','theta','theta_MV']

        # paths
        load_results_path = os.path.join(project_path, 'Results/Room' + Train_Room)

        # count features
        feature_index = 0

        # iterate over dnn and baselines
        for features in features_list:
            print(features)
            print(feature_index)

            save_metrics_folder_path = os.path.join(load_results_path, 'metrics', n_sources_string + 'Sources')
            save_metrics_path = os.path.join(save_metrics_folder_path, 'Zermini' + '_metrics_' + features +  '_'  + task + '_'  +'TestRoom' + Test_Room + post_suffix + DNN_suffix + '_' + DNN_type + '.h5')

            # load save metrics
            with h5py.File(save_metrics_path, 'r') as hf:
                metrics = hf.get('metrics_tensor')
                metrics_mix = hf.get('metrics_mix_tensor')
                metrics = np.array(metrics)
                metrics_mix = np.array(metrics_mix)

            if features == 'MV':
                # metrics tensor with dnn results and baselines
                metrics_all_tensor = np.zeros([3, metrics.shape[0], metrics.shape[1], metrics.shape[2], metrics.shape[3] ])

            print(save_metrics_path)

            # fill metrics_all tensor
            metrics_all_tensor[feature_index, :, :, :, :] = metrics

            # increae baseline index
            feature_index += 1


        ### PLOT METRICS FOR ALL BASELINES
        metrics_index = 0
        metrics_list = ['SDR','SIR', 'SAR', 'PESQ']

        for metric_name in metrics_list:

            metrics_all = metrics_all_tensor[:,:, :, metrics_index,:]

            if metric_name == metrics_list[0]:
                min = -6
                max = 15
            elif metric_name == metrics_list[1]:
                min = -3
                max = 24
            elif metric_name == metrics_list[2]:
                min = 0
                max = 15
            elif metric_name == metrics_list[3]:
                min = 0
                max = 4.

            # plot metrics for all baselines
            plot_metrics_baselines(metric_name, min, max)

            # increase metrics index
            metrics_index += 1


    else:
        raise Exception("Error, please double check the arguments.")
        sys.exit()
