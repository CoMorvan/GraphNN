import os
import pickle
import shutil
import sys
import timeit

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from C_SGEN import mydataset, C_SGEN, Trainer
from torch.utils.data import Dataset, DataLoader
from utils import *
from smiles_embedding import embed_smiles

training_number = 5
iteration = 33
batch = 32
lr = 5e-4
window = 5

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


def model_train(dataset):
    metric = dataset_para[dataset]['metric']
    C_SGEN_layers = dataset_para[dataset]['C-SGEL']
    k = dataset_para[dataset]['k']
    col = dataset_para[dataset]['col']

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    dir = 'data/' + dataset + '/'
    if os.path.exists(dir + 'plots_' + dataset + '/'):
        shutil.rmtree(dir + 'plots_' + dataset + '/')

    os.makedirs(dir + 'plots_' + dataset + '/', exist_ok=True)
    stats = pd.read_csv(dir + 'stats.csv')
    size = int(stats.iloc[0]['Size'])
    std = stats.iloc[0]['STD']
    mean = stats.iloc[0]['Mean']

    train_dataset = mydataset('train_data', dir)
    test_dataset = mydataset('test_data', dir)

    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=True, drop_last=True)

    setting = 'Dataset: ' + dataset + '\n' + \
              'Size: ' + str(size) + '\n' + \
              'Batch: ' + str(batch) + '\n' + \
              'k: ' + str(k) + '\n' + \
              'lr: ' + str(lr) + '\n' + \
              'C-SGEN layers: ' + str(C_SGEN_layers) + '\n' + \
              'Epochs: ' + str(iteration)

    print(setting)

    model = C_SGEN(k, C_SGEN_layers, batch, window, lr).to(torch.device('cuda'))

    trainer = Trainer(model.train(), std, mean, C_SGEN_layers, lr)
    tester = Trainer(model.eval(), std, mean, C_SGEN_layers, lr)
    metric_list = []

    data = pd.read_csv(dir + dataset + '.csv')
    smiles, scores = data.iloc[:,0].to_list(), data.iloc[:, -1].to_list()
    smiles = embed_smiles(smiles)

    for a in range(training_number):

        best_RMSE = float('inf')
        best_loss = float('inf')
        best_MAE = float('inf')
        best_epoch = 0
        for epoch in range(1, (iteration + 1)):
            start = timeit.default_timer()

            train_loss, pred_train, true_train = trainer.train(train_loader)
            test_loss, RMSE_test, predicted_test, true_test = tester.test(test_loader)
            MAE = mae(true_test, predicted_test)

            end = timeit.default_timer()
            time = end - start

            print('epoch: {:d} - train loss: {:.3f}, test loss: {:.3f}, RMSE: {:.3f}, MAE: {:.3f}, time: {:.3f}'.format(
                epoch, train_loss, test_loss, RMSE_test, MAE, time))


            if (RMSE_test < best_RMSE and metric=='MSE') or (MAE<best_MAE and metric=='MAE'):
                best_RMSE = RMSE_test
                best_loss = test_loss
                best_MAE = MAE
                best_epoch = epoch
                if metric=='MSE':
                    print('RMSE improved')
                elif metric=='MAE':
                    print('MAE improved')
                ci_value = ci(predicted_test, true_test)
                spear = spearman(predicted_test, true_test)
                pear = pearson(predicted_test, true_test)
                print('CI = {:.3}, Spearman = {:.3}, Pearson = {:.3}'.format(ci_value, spear, pear))
                plots(true_test, predicted_test, (best_RMSE, best_MAE, pear, spear, ci_value, test_loss),
                    dir + 'plots_' + dataset + '/' + dataset + '_' + str(best_RMSE) + '.png', dataset, col)
                #torch.save(model, dir + 'trained_model_' + str(a))

            if epoch == iteration:
                print('Best results:')
                print('RMSE = {}, CI = {:.3}, Spearman = {:.3}, Pearson = {:.3}'.format(best_RMSE, ci_value, spear, pear))
                metric_list.append([best_loss, best_RMSE, ci_value, spear, pear])

    mean_metrics = np.mean(metric_list, axis=0)
    std_metrics = np.std(metric_list, axis=0)
    with open(dir + 'plots_' + dataset + '/best_metrics.csv', 'w') as o:
            o.write('Model,dataset,Size,MSE,RMSE,Concordance Index,Spearman,Pearson\n')
            o.write('C-SGEN' + ',' + dataset + ',' + str(size) + ',' + str(mean_metrics[0]) + '+/-' + str(std_metrics[0]) + ',' + str(mean_metrics[1]) + '+/-' + str(std_metrics[1]) + ',' + str(mean_metrics[2]) + '+/-' + str(std_metrics[2]) + ',' + str(mean_metrics[3]) + '+/-' + str(std_metrics[3]) + ',' + str(mean_metrics[4]) + '+/-' + str(std_metrics[4]))


if __name__=="__main__":
    model_train(sys.argv[1])