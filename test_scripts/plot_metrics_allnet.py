# -*- coding: utf-8 -*-
#python plot_metrics_allnet.py B_format train 12BB01 12BB01 ['theta','MV'] '' ''
import sys
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
from create_latex_table import *


# parameters
n_sources = 4

# activate part of code
stereo_mode = 'monospec' # '' for stereo
metrics_values_list = []



### PLOT METHODS INDIVIDUALLY
def plot_metrics(name,min,max,baseline):

    fig,ax = plt.subplots(1,figsize=(10,7))

    if n_sources == 3:

        x0 = np.arange(0,metric0.shape[1])
        y0 = np.mean(metric0, axis=0)
        e0 = np.std(metric0, axis=0)
        g0 = ax.errorbar(x0, y0, e0, linestyle='None', marker='o', color='r', elinewidth=0.0001, markersize='6.0', label="1st target")
        ax.plot(x0[0:], y0[0:], '-o', color='r',  linewidth=1.2, markersize='6.0')
        ax.plot(x0[0:], y0[0:]+e0[0:], '--', linewidth=0.3, color='r')
        ax.plot(x0[0:], y0[0:]-e0[0:], '--', linewidth=0.3, color='r')

        x1 = np.arange(0,metric1.shape[1])
        y1 = np.mean(metric1, axis=0)
        e1 = np.std(metric1, axis=0)
        g1 = ax.errorbar(x1, y1, e1, linestyle='None', marker='o', color='k', elinewidth=0.0001, markersize='6.0', label="2nd target")
        ax.plot(x1[0:], y1[0:], '-o', color='k',  linewidth=1.2, markersize='6.0')
        ax.plot(x1[0:], y1[0:]+e1[0:], '--', linewidth=0.3, color='k')
        ax.plot(x1[0:], y1[0:]-e1[0:], '--', linewidth=0.3, color='k')

        x2 = np.arange(0,metric2.shape[1])
        y2 = np.mean(metric2, axis=0)
        e2 = np.std(metric2, axis=0)
        g2 = ax.errorbar(x2, y2, e2, linestyle='None', marker='o', color='g', elinewidth=0.0001, markersize='6.0', label="3rd target")
        ax.plot(x2[0:], y2[0:], '-o', color='g',  linewidth=1.2, markersize='6.0')
        ax.plot(x2[0:], y2[0:]+e2[0:], '--', linewidth=0.3, color='g')
        ax.plot(x2[0:], y2[0:]-e2[0:], '--', linewidth=0.3, color='g')

    if n_sources == 4:

        x0 = np.arange(0,metric0.shape[1])
        y0 = np.mean(metric0, axis=0)
        e0 = np.std(metric0, axis=0)
        g0 = ax.errorbar(x0, y0, e0, linestyle='None', marker='o', color='r', elinewidth=0.0001, markersize='6.0', label="1st target")
        ax.plot(x0[0:], y0[0:], '-o', color='r',  linewidth=1.2, markersize='6.0')
        ax.plot(x0[0:], y0[0:]+e0[0:], '--', linewidth=0.3, color='r')
        ax.plot(x0[0:], y0[0:]-e0[0:], '--', linewidth=0.3, color='r')

        x1 = np.arange(0,metric1.shape[1])
        y1 = np.mean(metric1, axis=0)
        e1 = np.std(metric1, axis=0)
        g1 = ax.errorbar(x1, y1, e1, linestyle='None', marker='o', color='k', elinewidth=0.0001, markersize='6.0', label="2nd target")
        ax.plot(x1[0:], y1[0:], '-o', color='k',  linewidth=1.2, markersize='6.0')
        ax.plot(x1[0:], y1[0:]+e1[0:], '--', linewidth=0.3, color='k')
        ax.plot(x1[0:], y1[0:]-e1[0:], '--', linewidth=0.3, color='k')

        x2 = np.arange(0,metric2.shape[1])
        y2 = np.mean(metric2, axis=0)
        e2 = np.std(metric2, axis=0)
        g2 = ax.errorbar(x2, y2, e2, linestyle='None', marker='o', color='g', elinewidth=0.0001, markersize='6.0', label="3rd target")
        ax.plot(x2[0:], y2[0:], '-o', color='g',  linewidth=1.2, markersize='6.0')
        ax.plot(x2[0:], y2[0:]+e2[0:], '--', linewidth=0.3, color='g')
        ax.plot(x2[0:], y2[0:]-e2[0:], '--', linewidth=0.3, color='g')

        x3 = np.arange(0,metric3.shape[1])
        y3 = np.mean(metric3, axis=0)
        e3 = np.std(metric3, axis=0)
        g3 = ax.errorbar(x3, y3, e3, linestyle='None', marker='o', color='#2bafff', elinewidth=0.0001, markersize='6.0', label="4th target")
        ax.plot(x3[0:], y3[0:], '-o', color='#2bafff',  linewidth=1.2, markersize='6.0')
        ax.plot(x3[0:], y3[0:]+e3[0:], '--', linewidth=0.3, color='#2bafff')
        ax.plot(x3[0:], y3[0:]-e3[0:], '--', linewidth=0.3, color='#2bafff')


    handles, labels = ax.get_legend_handles_labels()
    # remove the errorbars
    handles = [h[0] for h in handles]
    legend = ax.legend(handles, labels, loc='upper right',numpoints=1)
    legend.get_title().set_fontsize(20)
    ax.grid(color='#808080', linestyle=':', linewidth=0.3)
    plt.xlim( -1, metric0.shape[1])
    plt.ylim( min, max-3)
    locs, labels = plt.xticks([0,1,2,3,4,5,6,7,8], ["10$^\circ$", "20$^\circ$","30$^\circ$","40$^\circ$","50$^\circ$","60$^\circ$","70$^\circ$","80$^\circ$","90$^\circ$" ])
    locs, labels = plt.yticks( [i for i in np.arange(min,max,3)] )

    plt.tick_params(labelsize=18)
    plt.xlabel(r'$\Delta\theta$ (degrees)', fontsize=20)
    plt.ylabel('%s (dB)' %name, fontsize=20)
    if name == 'PESQ':
        plt.ylabel(name, fontsize=20)

    # plot and save
    if baseline == '':

        method_path = os.path.join(figures_path,DNN_type)
        if not os.path.exists(method_path):
            os.makedirs(method_path)

        plt.savefig(os.path.join( os.getcwd(), method_path, name + '_' + suffix + post_suffix + DNN_suffix + '_' + DNN_type), bbox_inches='tight',dpi=300)
        plt.title('%s: - %s' %(DNN_type, name), fontsize=30)
        plt.clf()

    else:

        method_path = os.path.join(figures_path,baseline)
        if not os.path.exists(method_path):
            os.makedirs(method_path)

        plt.savefig(os.path.join( os.getcwd(), method_path, name + '_' + suffix + post_suffix + DNN_suffix + '_' + baseline), bbox_inches='tight',dpi=300)
        plt.title('%s: - %s' %(baseline, name), fontsize=30)
        plt.clf()


### PLOT CURRENT METHOD WITH BASELINES
def plot_metrics_baselines(name,min,max):

    fig,ax = plt.subplots(1,figsize=(10,7))

    x0 = np.arange(0,metrics_all.shape[2])
    y0 = np.mean(np.mean(metrics_all[0,:,:,:], axis=0), axis =1)
    e0 = np.std(np.std(metrics_all[0,:,:,:], axis=0), axis =1)
    g0 = ax.errorbar(x0, y0, e0, linestyle='None', marker='o', color='orange', elinewidth=0.0001, markersize='8.0', label="Proposed (MLP)",zorder=5)
    ax.plot(x0[0:], y0[0:], '-', color='orange',  linewidth=1.5, markersize='6.0',zorder=5)
    ax.plot(x0[0:], y0[0:]+e0[0:], '--', linewidth=0.8, color='orange',zorder=5)
    ax.plot(x0[0:], y0[0:]-e0[0:], '--', linewidth=0.8, color='orange',zorder=5)

    x1 = np.arange(0,metrics_all.shape[2])
    y1 = np.mean(np.mean(metrics_all[1,:,:,:], axis=0), axis =1)
    e1 = np.std(np.std(metrics_all[1,:,:,:], axis=0), axis =1)
    g1 = ax.errorbar(x1, y1, e1, linestyle='None', marker='^', color='r', elinewidth=0.0001, markersize='8.0', label="Proposed (CNN)",zorder=4)
    ax.plot(x1[0:], y1[0:], '-', color='r',  linewidth=1.5, markersize='6.0',zorder=4)
    ax.plot(x1[0:], y1[0:]+e1[0:], '--', linewidth=0.8, color='r',zorder=4)
    ax.plot(x1[0:], y1[0:]-e1[0:], '--', linewidth=0.8, color='r',zorder=4)

    x2 = np.arange(0,metrics_all.shape[2])
    y2 = np.mean(np.mean(metrics_all[2,:,:,:], axis=0), axis =1)
    e2 = np.std(np.std(metrics_all[2,:,:,:], axis=0), axis =1)
    g2 = ax.errorbar(x2, y2, e2, linestyle='None', marker='s', color='g', elinewidth=0.0001, markersize='8.0', label="Chen et al.",zorder=3)
    ax.plot(x2[0:], y2[0:], '-', color='g',  linewidth=1.5, markersize='6.0',zorder=3)
    ax.plot(x2[0:], y2[0:]+e2[0:], '--', linewidth=0.8, color='g',zorder=3)
    ax.plot(x2[0:], y2[0:]-e2[0:], '--', linewidth=0.8, color='g',zorder=3)

    x3 = np.arange(0,metrics_all.shape[2])
    y3 = np.mean(np.mean(metrics_all[3,:,:,:], axis=0), axis =1)
    e3 = np.std(np.std(metrics_all[3,:,:,:], axis=0), axis =1)
    g3 = ax.errorbar(x3, y3, e3, linestyle='None', marker='P', color='#2bafff', elinewidth=0.0001, markersize='8.0', label=u'Günel et al.',zorder=2)
    ax.plot(x3[0:], y3[0:], '-', color='#2bafff',  linewidth=1.5, markersize='6.0',zorder=2)
    ax.plot(x3[0:], y3[0:]+e3[0:], '--', linewidth=0.8, color='#2bafff',zorder=2)
    ax.plot(x3[0:], y3[0:]-e3[0:], '--', linewidth=0.8, color='#2bafff',zorder=2)

    x4 = np.arange(0,metrics_all.shape[2])
    y4 = np.mean(np.mean(metrics_all[4,:,:,:], axis=0), axis =1)
    e4 = np.std(np.std(metrics_all[4,:,:,:], axis=0), axis =1)
    g4 = ax.errorbar(x4, y4, e4, linestyle='None', marker='*', color='gray', elinewidth=0.0001, markersize='12.0', label="Shujau et al.",zorder=1)
    ax.plot(x4[0:], y4[0:], '-', color='gray',  linewidth=1.5, markersize='6.0',zorder=1)
    ax.plot(x4[0:], y4[0:]+e4[0:], '--', linewidth=0.8, color='gray',zorder=1)
    ax.plot(x4[0:], y4[0:]-e4[0:], '--', linewidth=0.8, color='gray',zorder=1)

    if name != 'SAR':
        x5 = np.arange(0,metrics_all.shape[2])
        y5 = np.mean(np.mean(metrics_all[5,:,:,:], axis=0), axis =1)
        e5 = np.std(np.std(metrics_all[5,:,:,:], axis=0), axis =1)
        g5 = ax.errorbar(x5, y5, e5, linestyle='None', marker='D', color='k', elinewidth=0.0001, markersize='7.0', label="Mixture",zorder=0)
        ax.plot(x5[0:], y5[0:], '-', color='k',  linewidth=1.5, markersize='6.0',zorder=0)
        ax.plot(x5[0:], y5[0:]+e5[0:], '--', linewidth=0.8, color='k',zorder=0)
        ax.plot(x5[0:], y5[0:]-e5[0:], '--', linewidth=0.8, color='k',zorder=0)

    x6 = np.arange(0,metrics_all.shape[2])
    y6 = np.mean(np.mean(metrics_all[6,:,:,:], axis=0), axis =1)
    e6 = np.std(np.std(metrics_all[6,:,:,:], axis=0), axis =1)
    g6 = ax.errorbar(x6, y6, e6, linestyle='None', marker='X', color='blue', elinewidth=0.0001, markersize='12.0', label="IRMs",zorder=1)
    ax.plot(x6[0:], y6[0:], '-', color='blue',  linewidth=1.5, markersize='6.0',zorder=1)
    ax.plot(x6[0:], y6[0:]+e6[0:], '--', linewidth=0.8, color='blue',zorder=1)
    ax.plot(x6[0:], y6[0:]-e6[0:], '--', linewidth=0.8, color='blue',zorder=1)


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
    ###plt.savefig(os.path.join( os.getcwd(), figures_path, name + '_' + suffix + post_suffix + DNN_suffix + '_' + stereo_mode), bbox_inches='tight',dpi=300)
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
        project_path = os.path.join('/vol/vssp/mightywings', DNN_name)
        figures_path = os.path.join(os.getcwd(), 'Figures', suffix + post_suffix + DNN_suffix, n_sources_string )

        if project_path.startswith('DNN_'):
            project_path = project_path[len('DNN_'):]


        # create folders
        if not os.path.exists(figures_path):
            os.makedirs(figures_path)

        i = 0
        # baselines
        #baseline_list = ['', 'XCWW_' + str(n_sources) + 'src', 'Banu_' + str(n_sources) + 'src', 'shujau_' + str(n_sources) + 'src']
        baseline_list = ['Zermini', 'Zermini', 'XCWW_' + str(n_sources) + 'src', 'Banu_' + str(n_sources) + 'src', 'shujau_' + str(n_sources) + 'src']

        # paths
        load_results_path = os.path.join(project_path, 'Results/Room' + Train_Room)

        # count baseline
        baseline_index = 0

        # iterate over dnn and baselines
        for baseline in baseline_list:
            print(baseline)

            save_metrics_folder_path = os.path.join(load_results_path, 'metrics', n_sources_string + 'Sources')
            #save_metrics_path = os.path.join(save_metrics_folder_path, baseline + '_metrics_' + suffix + '_' + task + '_'  +'TestRoom' + Test_Room + post_suffix + DNN_suffix + '_' + DNN_type + '.h5')
            print(save_metrics_folder_path)

            if baseline == 'Zermini':
                if i == 0:
                    save_metrics_path = os.path.join(save_metrics_folder_path, baseline + '_metrics_' + suffix + '_' + task + '_'  +'TestRoom' + Test_Room + post_suffix + DNN_suffix + '_' + 'DNN' + '_' + stereo_mode + '.h5')


                elif i == 1:
                    save_metrics_path = os.path.join(save_metrics_folder_path, baseline + '_metrics_' + suffix + '_' + task + '_'  +'TestRoom' + Test_Room + post_suffix + DNN_suffix + '_' + 'CNN' + '_' + stereo_mode +  '.h5')

            else:
                # previous
                #save_metrics_path = os.path.join(save_metrics_folder_path, baseline + '_metrics_' + suffix + '_' + task + '_'  +'TestRoom' + Test_Room + post_suffix + '.h5')
                save_metrics_path = os.path.join(save_metrics_folder_path, baseline + '_metrics_' + suffix + '_' + task + '_'  +'TestRoom' + Test_Room + post_suffix + DNN_suffix + '_' + stereo_mode + '.h5')     
            
            # load save metrics
            with h5py.File(save_metrics_path, 'r') as hf:
                metrics = hf.get('metrics_tensor')
                metrics = np.array(metrics)

            if baseline == 'Zermini' and i == 0:
                # metrics tensor with dnn results and baselines
                metrics_all_tensor = np.zeros([7, metrics.shape[0], metrics.shape[1], metrics.shape[2], metrics.shape[3] ])
                i = i+1

                # load mix metrics
                with h5py.File(save_metrics_path, 'r') as hf:
                    metrics_mix = hf.get('metrics_mix_tensor')
                    metrics_mix = np.array(metrics_mix)

            # fill metrics_all tensor
            metrics_all_tensor[baseline_index, :, :, :, :] = metrics
            #print(metrics)
            metrics_all_tensor[5, :, :, :, :] =  metrics_mix

            # load metrics IRM
            save_metrics_IRM_path = os.path.join(save_metrics_folder_path, 'IRM_mono_' + suffix + '_' + task + '_'  +'TestRoom' + Test_Room + post_suffix + DNN_suffix +  '_' + stereo_mode + '.h5')
            with h5py.File(save_metrics_IRM_path, 'r') as hf:
                metrics_IRM = hf.get('metrics_tensor')
                metrics_IRM = np.array(metrics_IRM)
            
            metrics_all_tensor[6, :, :, :, :] =  metrics_IRM
            

            ### PLOT METRICS FOR EACH METHOD
            metrics_index = 0
            metrics_list = ['SDR','SIR', 'SAR', 'PESQ']
            for metric_name in metrics_list:

                print(metric_name)
                # sources
                metric0 = metrics[:,:,metrics_index,0]
                metric1 = metrics[:,:,metrics_index,1]
                metric2 = metrics[:,:,metrics_index,2]

                if n_sources == 4:
                    metric3 = metrics[:,:,metrics_index,3]

                ###plot_metrics(metric_name,-6,15,baseline)

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

            if n_sources == 3:
                if metric_name == metrics_list[0]:
                    min = -6
                    max = 18
                elif metric_name == metrics_list[1]:
                    min = -6
                    max = 24
                elif metric_name == metrics_list[2]:
                    min = 0
                    max = 18
                elif metric_name == metrics_list[3]:
                    min = 0.
                    max = 5.

            elif n_sources == 4:
                if metric_name == metrics_list[0]:
                    min = -6
                    max = 18
                elif metric_name == metrics_list[1]:
                    min = -6
                    max = 24
                elif metric_name == metrics_list[2]:
                    min = 0
                    max = 18
                elif metric_name == metrics_list[3]:
                    min = 0.
                    max = 5.

            print(metrics_all.shape)
            # plot metrics for all baselines
            plot_metrics_baselines(metric_name, min, max)
            
            # make latex tables
            create_latex_table(metrics_all, metric_name)
            

            # increase metrics index
            metrics_index += 1

        #with open("file.txt", "w") as output:
        #    output.write(str(metrics_values_list))


    else:
        raise Exception("Error, please double check the arguments.")
        sys.exit()
