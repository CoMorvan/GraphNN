from utils import plots
import pandas as pd
import numpy as np
from utils import *



dataset_para = {'QM7': {'metric': 'MAE', 'C-SGEL': 2, 'k': 4, 'col': 'magenta'},
                'QM8': {'metric': 'MAE', 'C-SGEL': 2, 'k': 4, 'col': 'darkviolet'},
                'QM9': {'metric': 'MAE', 'C-SGEL': 2, 'k': 4, 'col': 'purple'},
                'ESOL': {'metric': 'MSE', 'C-SGEL': 6, 'k': 4, 'col': 'darkorange'},
                'GA': {'metric': 'MSE', 'C-SGEL': 6, 'k': 4, 'col': 'gold'},
                'FreeSolv': {'metric': 'MSE', 'C-SGEL': 2, 'k': 4, 'col': 'brown'},
                'Lipophilicity': {'metric': 'MSE', 'C-SGEL': 2, 'k': 4, 'col': 'tomato'},
                'PDBbind_core': {'metric': 'MSE', 'C-SGEL': 2, 'k': 16, 'col': 'cyan'},
                'PDBbind_refined': {'metric': 'MSE', 'C-SGEL': 1, 'k': 16, 'col': 'royalblue'},
                'PDBbind_full': {'metric': 'MSE', 'C-SGEL': 5, 'k': 16, 'col': 'navy'}}


for dataset in ['Lipophilicity', 'FreeSolv', 'ESOL', 'GA', 'PDBbind_core', 'PDBbind_refined', 'QM7']:

    col = dataset_para[dataset]['col']
    dir = 'data/'+dataset+'/' + 'chemprop/'
    stats = pd.read_csv('data/'+dataset+'/'+'stats.csv')
    size = stats.iloc[0]['Size']
    std = stats.iloc[0]['STD']
    mean = stats.iloc[0]['Mean']
    metrics = []
    true_data = pd.read_csv('data/'+dataset+'/'+ dataset+ '.csv')
    true = np.array(true_data.iloc[:,-1].to_list())
    true_norm = (true-mean)/std
    for i in range(5):
        metrics.append([])
        preds_data = pd.read_csv(dir+dataset+str(i)+'_preds.csv')
        preds_norm = np.array(preds_data.iloc[:,-1].to_list())
        preds = preds_norm*std + mean

        metrics[i].append(rms_score(preds, true))
        metrics[i].append(mae(true, preds))
        metrics[i].append(pearson(preds, true))
        metrics[i].append(spearman(preds, true))
        metrics[i].append(ci(preds, true))
        metrics[i].append(mse(true_norm, preds_norm))
        plots(true, preds, metrics[i], dir + dataset + str(i) + '_chemprop.png', dataset, col)

    col = dataset_para[dataset]['col']

    mean_metrics = np.mean(metrics, axis=0)
    std_metrics = np.std(metrics, axis = 0)

    with open(dir + 'best_metrics.csv', 'w') as o:
            o.write('Model,dataset,Size,MSE,RMSE,Concordance Index,Spearman,Pearson\n')
            o.write('Chemprop' + ',' + dataset + ',' + str(size) + ',' + str(mean_metrics[5]) + '+/-' + str(std_metrics[5]) +
                    ',' + str(mean_metrics[0]) + '+/-' + str(std_metrics[0]) + ',' + str(mean_metrics[4]) + '+/-' +
                    str(std_metrics[4]) + ',' + str(mean_metrics[3]) + '+/-' + str(std_metrics[3]) + ',' +
                     str(mean_metrics[2]) + '+/-' + str(std_metrics[2]))