#python plot_metrics_comparison_bars.py B_format train 12BB01 12BB01 ['theta','MV'] '' 'newnorm'
import matplotlib
matplotlib.use('Agg')
import sys
import numpy as np
import os
import time
import h5py
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# parameters
n_sources = 3
DNN_type = 'DNN'
stereo_mode = 'monospec'

lwc = 1.8
lw = 0.6

def plot_bars_percentage(metrics_bars, metrics_list):
    x = np.arange(2)
    width = 0.15

    # Plot 1
    #diff0 = metrics_bars[2,:]-metrics_bars[0,:]
    #diff1 = metrics_bars[2,:]-metrics_bars[1,:]

    diff0 = (metrics_bars[2, :] - metrics_bars[0, :])/metrics_bars[2, :]*100
    diff1 = (metrics_bars[2, :] - metrics_bars[1, :]) / metrics_bars[2, :] * 100

    #print(diff0, diff1)
    sdr_bar = [diff0[0], diff1[0]]
    sir_bar = [diff0[2], diff1[2]]
    sar_bar = [diff0[1], diff1[1]]
    pesq_bar = [diff0[3], diff1[3]]
    wer_bar = [diff0[4], diff1[4]]


    plt.figure(figsize=(20, 11))
    ax = plt.subplot(111)


    rects0 = ax.bar(x - 2*width , sdr_bar , width=width , color='tab:blue', align='center') #yerr
    rects1 = ax.bar(x - width, sir_bar, width=width, color='tab:green', align='center')
    rects2 = ax.bar(x, sar_bar, width=width, color='tab:orange', align='center')
    rects3 = ax.bar(x + width, pesq_bar, width=width, color='grey', align='center')
    rects4 = ax.bar(x + 2*width, wer_bar, width=width, color='tab:pink', align='center')

    rects_all = [rects0, rects1, rects2, rects3, rects4]

    #plt.yticks(np.arange(-.5, 4, .5))
    plt.yticks(np.arange(-20,75, 10))

    # Call the function above. All the magic happens there.
    #add_value_labels_percentage(ax)

    labels = [r'$\left(\Theta, G_x^{LPS},G_y^{LPS}\right)$ vs $\left(G_x^{LPS},G_y^{LPS}\right)$',
            r'$\left(\Theta, G_x^{LPS},G_y^{LPS}\right)$ vs $\left(\Theta\right)$']
    ax.set_xticklabels(['','', labels[0], '','','', labels[1], ''],fontsize=60)

    tau = ['SDR', 'SAR', 'SIR', 'PESQ', 'Word recognition']
    plt.legend(tau, loc='upper left', prop={'size': 30})

    # print rects and add percentages
    for rects in rects_all:
        for rect in rects:

            height = rect.get_height()

            if height > 0:
                va = 'bottom'
                form = '+' + '%.2f'
            else:
                va = 'top'
                form = '%.2f'

            ax.text(rect.get_x() + rect.get_width() / 2., 0.99 * height,
                        form % float(height) + "%", ha='center', va=va, size=28, rotation='45')


    # plt.tight_layout()
    matplotlib.rcParams.update({'font.size': 30})
    plt.subplots_adjust(bottom=0.15)
    plt.subplots_adjust(left=0.1)

    formatter = FuncFormatter(lambda y, pos: "%d%%" % (y))
    ax.yaxis.set_major_formatter(formatter)

    plt.axhline(0, color='k')
    plt.minorticks_on()

    plt.xlabel('Input features', fontsize=35)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.grid(which='major',axis='y',linestyle='-')
    plt.grid(which='minor', axis='y', linestyle=':', linewidth='0.25', color='black')
    plt.ylabel('Average improvement', fontsize=35)
    plt.title('Average improvement', fontsize=45)
    plt.savefig(os.path.join(os.getcwd(), 'histo_percentage'))



def plot_bars(metrics_bars, metrics_list):
    x = np.arange(2)
    width = 0.15

    # Plot 1
    diff0 = metrics_bars[2,:]-metrics_bars[0,:]
    diff1 = metrics_bars[2,:]-metrics_bars[1,:]


    #print(diff0, diff1)
    sdr_bar = [diff0[0], diff1[0]]
    sir_bar = [diff0[2], diff1[2]]
    sar_bar = [diff0[1], diff1[1]]
    pesq_bar = [diff0[3], diff1[3]]
    wer_bar = [diff0[4], diff1[4]]



    plt.figure(figsize=(20, 11))
    ax = plt.subplot(111)

    rects0 = ax.bar(x - 2*width, sdr_bar, width=width, color='tab:blue', align='center') #yerr
    rects1 = ax.bar(x - width, sir_bar, width=width, color='tab:green', align='center')
    rects2 = ax.bar(x, sar_bar, width=width, color='tab:orange', align='center')
    rects3 = ax.bar(x + width, pesq_bar, width=width, color='grey', align='center')
    rects4 = ax.bar(x + 2*width, wer_bar, width=width, color='tab:pink', align='center')

    rects_all = [rects0, rects1, rects2, rects3]

    #plt.yticks(np.arange(-.5, 4, .5))
    plt.yticks(np.arange(-1., 4.5, .5))

    # Call the function above. All the magic happens there.
    #add_value_labels(ax)

    labels = [r'$\left(\Theta, G_x^{LPS},G_y^{LPS}\right)$ vs $\left(G_x^{LPS},G_y^{LPS}\right)$',
            r'$\left(\Theta, G_x^{LPS},G_y^{LPS}\right)$ vs $\left(\Theta\right)$']
    ax.set_xticklabels(['','', labels[0], '','','', labels[1], ''],fontsize=60)

    tau = ['SDR', 'SAR', 'SIR', 'PESQ', 'Word recognition']
    plt.legend(tau, loc='upper left', prop={'size': 30})

    # print rects and add percentages
    for rects in rects_all:
        for rect in rects:
            height = rect.get_height()

            if height > 0:
                va = 'bottom'
                form = '+' + '%.2f'
            else:
                va = 'top'
                form = '%.2f'

            ax.text(rect.get_x() + rect.get_width() / 2., 0.99 * height,
                form % float(height) + " dB", ha='center', va=va, size=25, rotation='45')

    for rect in rects4:
        height = rect.get_height()

        if height > 0:
            va = 'bottom'
            form = '+' + '%.2f'
        else:
            va = 'top'
            form = '%.2f'

        ax.text(rect.get_x() + rect.get_width() / 2., 0.99 * height,
                form % float(height) + "%", ha='center', va=va, size=25, rotation='45')

    plt.axhline(0, color='k')
    # plt.tight_layout()
    matplotlib.rcParams.update({'font.size': 25})
    plt.subplots_adjust(bottom=0.15)
    plt.subplots_adjust(left=0.1)

    plt.minorticks_on()

    plt.xlabel('Input features', fontsize=35)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.grid(which='major',axis='y',linestyle='-')
    plt.grid(which='minor', axis='y', linestyle=':', linewidth='0.25', color='black')
    plt.ylabel('Average improvement', fontsize=30)
    plt.title('Average improvement', fontsize=40)
    plt.savefig(os.path.join(os.getcwd(), 'histo_0'))


def add_value_labels(ax, spacing=10):

    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        if rect.get_y() < 0:
            y_value = rect.get_y()
        else:
            y_value = rect.get_height()

        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'

        print(y_value)

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            print(space)
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place
        label = "{:.2f}".format(y_value)
        #if y_value > 0:
        #    label = '+' + "{:.2f}".format(y_value) + '%'

        #elif y_value < 0:
        #    label = "{:.2f}".format(y_value) + '%'

        # Create annotation
        ax.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(0, space),          # Vertically shift label by `space`
            textcoords="offset points", # Interpret `xytext` as offset in points
            ha='center',                # Horizontally center label
            va=va,
            size=23)                      # Vertically align label differently for
                                        # positive and negative values.


def add_value_labels_percentage(ax, spacing=10):

    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        if rect.get_y() < 0:
            y_value = rect.get_y()
        else:
            y_value = rect.get_height()

        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'

        print(y_value)

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            print(space)
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place
        #label = "{:.2f}".format(y_value) + '%'

        if y_value > 0:
            label = '+' + "{:.2f}".format(y_value) + '%'

        elif y_value < 0:
            label = "{:.2f}".format(y_value) + '%'

        # Create annotation
        ax.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(0, space),          # Vertically shift label by `space`
            textcoords="offset points", # Interpret `xytext` as offset in points
            ha='center',                # Horizontally center label
            va=va,
            size=23)                      # Vertically align label differently for
                                        # positive and negative values.


### PLOT CURRENT METHOD WITH BASELINES
def plot_metrics_baselines(name,min,max):

    fig,ax = plt.subplots(1,figsize=(10,7))

    x1 = np.arange(0,metrics_all.shape[2])
    y1 = np.mean(np.mean(metrics_all[1,:,:,:], axis=0), axis =1)
    e1 = np.std(np.std(metrics_all[1,:,:,:], axis=0), axis =1)
    g1 = ax.errorbar(x1, y1, e1, linestyle='None', marker='o', color='brown', elinewidth=0.0001, markersize='6.0', label=r'MLP, input = $\left(\Theta\right)$')
    ax.plot(x1[0:], y1[0:], '-o', color='brown',  linewidth=lwc, markersize='6.0',zorder=5)
    ax.plot(x1[0:], y1[0:]+e1[0:], '--', linewidth=lw, color='brown',zorder=5)
    ax.plot(x1[0:], y1[0:]-e1[0:], '--', linewidth=lw, color='brown',zorder=5)

    x0 = np.arange(0,metrics_all.shape[2])
    y0 = np.mean(np.mean(metrics_all[0,:,:,:], axis=0), axis =1)
    e0 = np.std(np.std(metrics_all[0,:,:,:], axis=0), axis =1)
    g0 = ax.errorbar(x0, y0, e0, linestyle='None', marker='o', color='b', elinewidth=0.0001, markersize='6.0', label=r'MLP, input = $\left(G_x^{LPS},G_y^{LPS}\right)$')
    ax.plot(x0[0:], y0[0:], '-o', color='b',  linewidth=lwc, markersize='6.0',zorder=5)
    ax.plot(x0[0:], y0[0:]+e0[0:], '--', linewidth=lw, color='b',zorder=5)
    ax.plot(x0[0:], y0[0:]-e0[0:], '--', linewidth=lw, color='b',zorder=5)

    x2 = np.arange(0,metrics_all.shape[2])
    y2 = np.mean(np.mean(metrics_all[2,:,:,:], axis=0), axis =1)
    e2 = np.std(np.std(metrics_all[2,:,:,:], axis=0), axis =1)
    g2 = ax.errorbar(x2, y2, e2, linestyle='None', marker='o', color='darkorange', elinewidth=0.0001, markersize='6.0', label=r'MLP, input = $\left(\Theta,G_x^{LPS},G_y^{LPS}\right)$')
    ax.plot(x2[0:], y2[0:], '-o', color='darkorange',  linewidth=lwc, markersize='6.0',zorder=5)
    ax.plot(x2[0:], y2[0:]+e2[0:], '--', linewidth=lw, color='darkorange',zorder=5)
    ax.plot(x2[0:], y2[0:]-e2[0:], '--', linewidth=lw, color='darkorange',zorder=5)


    handles, labels = ax.get_legend_handles_labels()
    # remove the errorbars
    handles = [h[0] for h in handles]
    legend = ax.legend(handles, labels, loc='upper right',numpoints=1)
    legend.get_title().set_fontsize(20)
    ax.grid(color='#808080', linestyle=':', linewidth=0.3)
    plt.xlim( -1, metrics_all.shape[2])
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

    if name == 'Google_speech_API':
        plt.ylabel('Accuracy (%))', fontsize=20)
        plt.ylim( min, max-10)
        locs, labels = plt.yticks( [i for i in np.arange(min,max,10)] )

    save_path = os.path.join( os.getcwd(), figures_path, 'Comparison_' + name + DNN_suffix)

    # plot and save
    plt.savefig(save_path,bbox_inches='tight' )
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
        figures_path = os.path.join(os.getcwd(), 'Figures', suffix  + '_' + post_suffix + DNN_suffix + '_' +n_sources_string )

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
        load_results_path = os.path.join(project_path, 'Results/Room' + Train_Room, 'metrics', n_sources_string + 'Sources')

        # count features
        feature_index = 0

        # iterate over dnn and baselines
        for features in features_list:
            
            print(features)
            
            print(feature_index)

            save_metrics_path = os.path.join(load_results_path, 'Zermini' + '_metrics_' + features +  '_'  + task + '_'  +'TestRoom' + Test_Room + post_suffix + DNN_suffix + '_' + DNN_type + '_' + stereo_mode + '.h5')
            print(save_metrics_path)
            
            save_sr_path = os.path.join(load_results_path, 'Zermini_speech_recognition_' + features + '_' + task + '_'  +'TestRoom' + Test_Room  + post_suffix + DNN_suffix + '_' + DNN_type + '_' + stereo_mode + '.h5')

            # load save metrics
            with h5py.File(save_metrics_path, 'r') as hf:
                metrics = hf.get('metrics_tensor')
                metrics_mix = hf.get('metrics_mix_tensor')
                metrics = np.array(metrics)
                metrics_mix = np.array(metrics_mix)
            
            print(save_sr_path)

            # load save sr
            with h5py.File(save_sr_path, 'r') as hf:
                sr = hf.get('metrics_tensor_sr')
                sr = np.array(sr)

            if features == 'MV':
                metrics_all_tensor = np.zeros([3, metrics.shape[0], metrics.shape[1], metrics.shape[2]+1, metrics.shape[3] ]) # define only once

            # fill metrics_all tensor
            metrics_all_tensor[feature_index, :, :, 0:4, :] = metrics
            print(metrics.shape)
            metrics_all_tensor[feature_index, :, :, 4, :] = sr.reshape(sr.shape[0],sr.shape[1],sr.shape[3])
            print(metrics_all_tensor.shape)


            # increase baseline index
            feature_index += 1



        ### PLOT METRICS FOR ALL BASELINES
        metrics_index = 0
        metrics_list = ['SDR','SIR', 'SAR', 'PESQ', 'Google_speech_API']

        metrics_bars = np.zeros([3, 5])

        for metric_name in metrics_list:

            metrics_all = metrics_all_tensor[:,:, :, metrics_index,:]
            metrics_mean = np.mean(np.mean(np.mean(metrics_all, axis=1), axis=1), axis=1)
            metrics_std = np.std(np.std(np.std(metrics_all, axis=1), axis=1), axis=1)

            print(metrics_index, metrics_mean, 'std', metrics_std)


            metrics_bars[:, metrics_index] = metrics_mean

            # increase metrics index
            metrics_index += 1
        
        plot_bars(metrics_bars, metrics_list)
        plot_bars_percentage(metrics_bars, metrics_list)


    else:
        raise Exception("Error, please double check the arguments.")
        sys.exit()
