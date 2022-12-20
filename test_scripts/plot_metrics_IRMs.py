# -*- coding: utf-8 -*-
#python plot_metrics_allnet.py B_format train 12BB01 12BB01 ['theta','MV'] '' ''
import sys
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt

# parameters
n_sources = 4
stereo_mode = 'monospec'

def plot_metrics_IRMs(name,min,max):

    fig,ax = plt.subplots(1,figsize=(10,7))

    x0 = np.arange(0,metrics_all.shape[2])
    y0 = np.mean(np.mean(metrics_all[0,:,:,:], axis=0), axis =1)
    e0 = np.std(np.std(metrics_all[0,:,:,:], axis=0), axis =1)
    g0 = ax.errorbar(x0, y0, e0, linestyle='None', marker='o', color='orange', elinewidth=0.0001, markersize='8.0', label=IRMs_list[0],zorder=5)
    ax.plot(x0[0:], y0[0:], '-', color='orange',  linewidth=1.5, markersize='6.0',zorder=5)
    ax.plot(x0[0:], y0[0:]+e0[0:], '--', linewidth=0.8, color='orange',zorder=5)
    ax.plot(x0[0:], y0[0:]-e0[0:], '--', linewidth=0.8, color='orange',zorder=5)

    x1 = np.arange(0,metrics_all.shape[2])
    y1 = np.mean(np.mean(metrics_all[1,:,:,:], axis=0), axis =1)
    e1 = np.std(np.std(metrics_all[1,:,:,:], axis=0), axis =1)
    g1 = ax.errorbar(x1, y1, e1, linestyle='None', marker='^', color='r', elinewidth=0.0001, markersize='8.0', label=IRMs_list[1],zorder=4)
    ax.plot(x1[0:], y1[0:], '-', color='r',  linewidth=1.5, markersize='6.0',zorder=4)
    ax.plot(x1[0:], y1[0:]+e1[0:], '--', linewidth=0.8, color='r',zorder=4)
    ax.plot(x1[0:], y1[0:]-e1[0:], '--', linewidth=0.8, color='r',zorder=4)

    x2 = np.arange(0,metrics_all.shape[2])
    y2 = np.mean(np.mean(metrics_all[2,:,:,:], axis=0), axis =1)
    e2 = np.std(np.std(metrics_all[2,:,:,:], axis=0), axis =1)
    g2 = ax.errorbar(x2, y2, e2, linestyle='None', marker='s', color='g', elinewidth=0.0001, markersize='8.0', label=IRMs_list[2],zorder=3)
    ax.plot(x2[0:], y2[0:], '-', color='g',  linewidth=1.5, markersize='6.0',zorder=3)
    ax.plot(x2[0:], y2[0:]+e2[0:], '--', linewidth=0.8, color='g',zorder=3)
    ax.plot(x2[0:], y2[0:]-e2[0:], '--', linewidth=0.8, color='g',zorder=3)

    x3 = np.arange(0,metrics_all.shape[2])
    y3 = np.mean(np.mean(metrics_all[3,:,:,:], axis=0), axis =1)
    e3 = np.std(np.std(metrics_all[3,:,:,:], axis=0), axis =1)
    g3 = ax.errorbar(x3, y3, e3, linestyle='None', marker='P', color='#2bafff', elinewidth=0.0001, markersize='8.0', label=IRMs_list[3],zorder=2)
    ax.plot(x3[0:], y3[0:], '-', color='#2bafff',  linewidth=1.5, markersize='6.0',zorder=2)
    ax.plot(x3[0:], y3[0:]+e3[0:], '--', linewidth=0.8, color='#2bafff',zorder=2)
    ax.plot(x3[0:], y3[0:]-e3[0:], '--', linewidth=0.8, color='#2bafff',zorder=2)

    handles, labels = ax.get_legend_handles_labels()
    # remove the errorbars
    handles = [h[0] for h in handles]
    legend = ax.legend(handles, labels, loc='upper center',numpoints=1, ncol=3, prop={'size': 17})
    legend.get_title().set_fontsize(20)
    ax.grid(color='#808080', linestyle=':', linewidth=0.3)
    plt.xlim( -1, metric0.shape[1])
    plt.ylim( min, max-3)
    locs, labels = plt.xticks([0,1,2,3,4,5,6,7,8], ["10$^\circ$", "20$^\circ$","30$^\circ$","40$^\circ$","50$^\circ$","60$^\circ$","70$^\circ$","80$^\circ$","90$^\circ$" ])
    locs, labels = plt.yticks( [i for i in np.arange(min,max,3)] )
    if name == 'PESQ':
        locs, labels = plt.yticks( [i for i in np.arange(min,max,0.5)] )
    plt.tick_params(labelsize=18)
    plt.xlabel(r'$\Delta\theta$ (degrees)', fontsize=20)
    plt.ylabel('%s (dB)' %name, fontsize=20)
    if name == 'PESQ':
        plt.ylabel(name, fontsize=20)

    # plot and save
    figures_IRMs_path = os.path.join(figures_path, 'IRMs')
    if not os.path.exists(figures_IRMs_path):
        os.makedirs(figures_IRMs_path)
    print(figures_IRMs_path)
    plt.savefig(os.path.join( figures_IRMs_path, name + '_' + suffix + post_suffix + DNN_suffix + '_' + stereo_mode), bbox_inches='tight',dpi=300)
    plt.title('%s comparison' %(name), fontsize=30)
    plt.clf()


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

        if post_suffix != '':
            post_suffix = '_' + post_suffix

        if DNN_suffix != '':
            DNN_suffix = '_' + DNN_suffix

        print('suffix', suffix, 'post_suffix', post_suffix, 'DNN_suffix', DNN_suffix, 'Test_Room', Test_Room)
        

        # number of sources
        if n_sources == 2:
            n_sources_string = 'Two'

        elif n_sources == 3:
            n_sources_string = 'Three'

        elif n_sources == 4:
            n_sources_string = 'Four'

        # paths
        project_path = os.path.join('/media/alfredo/storage/', DNN_name)
        figures_path = os.path.join(os.getcwd(), 'Figures', suffix + post_suffix + DNN_suffix, n_sources_string )

        if project_path.startswith('DNN_'):
            project_path = project_path[len('DNN_'):]


        # create folders
        if not os.path.exists(figures_path):
            os.makedirs(figures_path)

        i = 0
        # baselines
        #baseline_list = ['', 'XCWW_' + str(n_sources) + 'src', 'Banu_' + str(n_sources) + 'src', 'shujau_' + str(n_sources) + 'src']
        IRMs_list = ['IRM_mono', 'IRM_mono_square', 'IRM_stereo', 'IRM_stereo_square']

        # paths
        load_results_path = os.path.join(project_path, 'Results/Room' + Train_Room)

        # count baseline
        baseline_index = 0

        # iterate over dnn and baselines
        for baseline in IRMs_list:
            print(baseline)

            save_metrics_folder_path = os.path.join(load_results_path, 'metrics', n_sources_string + 'Sources')
            #save_metrics_path = os.path.join(save_metrics_folder_path, baseline + '_metrics_' + suffix + '_' + task + '_'  +'TestRoom' + Test_Room + post_suffix + DNN_suffix + '_' + DNN_type + '.h5')

            # load metrics IRM
            save_metrics_IRM_path = os.path.join(save_metrics_folder_path, baseline + '_' + suffix + '_' + task + '_'  +'TestRoom' + Test_Room + post_suffix + DNN_suffix + '_' + stereo_mode + '.h5')
            with h5py.File(save_metrics_IRM_path, 'r') as hf:
                metrics_IRM = hf.get('metrics_tensor')
                metrics_IRM = np.array(metrics_IRM)

            if baseline == 'IRM_mono' and i == 0:
                # metrics tensor with dnn results and baselines
                metrics_all_tensor = np.zeros([4, metrics_IRM.shape[0], metrics_IRM.shape[1], metrics_IRM.shape[2], metrics_IRM.shape[3] ])
                i = i+1

            # fill metrics_all tensor
            metrics_all_tensor[baseline_index, :, :, :, :] = metrics_IRM

            ### PLOT METRICS FOR EACH METHOD
            metrics_index = 0
            metrics_list = ['SDR','SIR', 'SAR', 'PESQ']
            for metric_name in metrics_list:

                print(metric_name)
                # sources
                metric0 = metrics_IRM[:,:,metrics_index,0]
                metric1 = metrics_IRM[:,:,metrics_index,1]
                metric2 = metrics_IRM[:,:,metrics_index,2]

                if n_sources == 4:
                    metric3 = metrics_IRM[:,:,metrics_index,3]

                #plot_metrics(metric_name,-6,15,baseline)

                '''
                # mixture
                metric0 = metrics_mix[:,:,metrics_index,0]
                metric1 = metrics_mix[:,:,metrics_index,1]
                metric2 = metrics_mix[:,:,metrics_index,2]
                if n_sources == 4:
                    metric3 = metrics_mix[:,:,metrics_index,3]

                plot_metrics(metric_name,-6,15,baseline)

                # delta
                metric0 = np.mean(metrics0[:,:,metrics_index,:]-metrics_mix0[:,:,metrics_index,:],axis=2)
                metric1 = np.mean(metrics1[:,:,metrics_index,:]-metrics_mix1[:,:,metrics_index,:],axis=2)
                metric2 = np.mean(metrics2[:,:,metrics_index,:]-metrics_mix2[:,:,metrics_index,:],axis=2)
                if n_sources == 4:
                    metric3 = np.mean(metrics3[:,:,metrics_index,:]-metrics_mix3[:,:,metrics_index,:],axis=2)

                plot_metrics(metric_name,-6,15,baseline)
                '''
                # increase metrics index
                metrics_index += 1

            # increae baseline index
            baseline_index += 1


        ### PLOT METRICS FOR ALL BASELINES
        metrics_index = 0
        metrics_list = ['SDR','SIR', 'SAR', 'PESQ']

        for metric_name in metrics_list:

            metrics_all = metrics_all_tensor[:,:, :, metrics_index,:]

            if n_sources == 4:

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
                    min = 0.
                    max = 4.

            elif n_sources == 3:

                if metric_name == metrics_list[0]:
                    min = -3
                    max = 18
                elif metric_name == metrics_list[1]:
                    min = 0
                    max = 27
                elif metric_name == metrics_list[2]:
                    min = 3
                    max = 18
                elif metric_name == metrics_list[3]:
                    min = 1.
                    max = 5.

            # plot metrics for all baselines
            plot_metrics_IRMs(metric_name, min, max)

            # increase metrics index
            metrics_index += 1


    else:
        raise Exception("Error, please double check the arguments.")
        sys.exit()
