from texttable import Texttable
import latextable
import numpy as np


def calculate_mean(metrics_all, index):
    
    return np.mean(np.mean(metrics_all[index,:,:,:], axis=0), axis =1)


def create_latex_table(metrics_all, metric_name):

    # metrics_mean =  np.mean(np.mean(metrics_all, axis=1 ), axis=2)
    # print(metrics_mean[0,:])


    baseline_table_list = ['Proposed (MLP)', 'Proposed (CNN)', "Chen et al.",\
         u'Günel et al.', "Shujau et al.", 'Mixture', "IRMs"]

    metrics_mean_list = []

    
    # convert numpy array to list
    for j in range(len(metrics_all)):
        metrics_mean_list.append(list(   np.round(calculate_mean(metrics_all, j),decimals=2)  ) )

    # add baseline name to list
    for i in range(len(metrics_mean_list)):
        metrics_mean_list[i].insert(0, baseline_table_list[i])


    table_2 = Texttable()
    table_2.set_deco(Texttable.HEADER)
    table_2.set_cols_dtype(['t', 'f', 'f', 'f', 'f','f', 'f', 'f', 'f','f'])
    table_2.set_cols_align(["c", "c", "c", "c", "c","c", "c", "c", "c","c"])
    table_2.add_rows([["Method/r$\Delta\theta$", '$10^\circ$', '$20^\circ$', '$30^\circ$', \
        '$40^\circ$', '$50^\circ$', '$60^\circ$', '$70^\circ$', '$80^\circ$', '$90^\circ$'],
        metrics_mean_list[0],
        metrics_mean_list[1],
        metrics_mean_list[2],
        metrics_mean_list[3],
        metrics_mean_list[4],
        metrics_mean_list[5],
        metrics_mean_list[6]])
    print('\nTexttable Output:')
    print(table_2.draw())
    print('\nLatextable Output:')
    print(latextable.draw_latex(table_2, caption=metric_name, label=metric_name))