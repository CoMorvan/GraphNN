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
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset, DataLoader
from utils import *


training_number = 5
decay_interval = 10
ch_num = 4
iteration = 10
window = 5
layer_cnn = 3
lr_decay = 0.5
batch = 32
k = 16
lr = 5e-3
C_SGEN_layers = 2

if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    for dataset in ['kiba', 'davis']:
        dir = 'data/' + dataset + '/'
        if os.path.exists(dir + 'plots_' + dataset + '/'):
            shutil.rmtree(dir + 'plots_' + dataset + '/')

        os.makedirs(dir + 'plots_' + dataset + '/', exist_ok=True)

        train_dataset = mydataset('train', dir)
        test_dataset = mydataset('test', dir)

        train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=True, drop_last=True)

        setting = 'dir: ' + str(dir) + '\n' + \
                'batch: ' + str(batch) + '\n' + \
                'k: ' + str(k) + '\n' + \
                'lr: ' + str(lr) + '\n' + \
                'C-SGEN layers: ' + str(C_SGEN_layers) + '\n' + \
                'epochs: ' + str(iteration)

        print(setting)

        model = C_SGEN().to(torch.device('cuda'))

        trainer = Trainer(model.train(), C_SGEN_layers)
        tester = Trainer(model.eval(), C_SGEN_layers)

        metric_list = []

        for _ in range(training_number):

            best_RMSE = float('inf')
            best_loss = float('inf')
            best_epoch = 0

            for epoch in range(1, (iteration + 1)):
                start = timeit.default_timer()

                train_loss, pred_train, true_train = trainer.train(train_loader)
                test_loss, RMSE_test, predicted_test, true_test = tester.test(test_loader)

                end = timeit.default_timer()
                time = end - start

                print('epoch:{:d} - train loss: {:.3f}, test loss: {:.3f}, test rmse: {:.3f}, time: {:.3f}'.format(
                    epoch, train_loss, test_loss, RMSE_test, time))
                if RMSE_test < best_RMSE:
                    best_RMSE = RMSE_test
                    best_loss = test_loss
                    best_epoch = epoch
                    best_predicted, best_true = predicted_test, true_test
                    print('RMSE improved')

                if epoch == iteration:

                    ci_value = ci(best_predicted, best_true)
                    spear = spearman(best_predicted, best_true)
                    pear = pearson(best_predicted, best_true)
                    print('CI = {:.3f}, Spearman = {:.3f}, Pearson = {:.3f}'.format(ci_value, spear, pear))
                    plots(best_true, best_predicted,
                        dir + 'plots_' + dataset + '/' + dataset + '_' + str(best_RMSE) + '.png', save=True)
                    print('Best results:')
                    print('RMSE = {}, CI = {:.3f}, Spearman = {:.3f}, Pearson = {:.3f}'.format(best_RMSE, ci_value, spear, pear))
                    metric_list.append([best_loss, best_RMSE, ci_value, spear, pear])

        mean_metrics = np.mean(metric_list, axis=0)
        std_metrics = np.std(metric_list, axis=0)
        with open(dir + 'plots_' + dataset + '/best_metrics.csv', 'w') as o:
                o.write('Model,MSE,RMSE,Concordance Index,Spearman,Pearson\n')
                o.write('C-SGEN' + ',' + str(mean_metrics[0]) + '+/-' + str(std_metrics[0]) + ',' + str(mean_metrics[1]) + '+/-' + str(std_metrics[1]) + ',' + str(mean_metrics[2]) + '+/-' + str(std_metrics[2]) + ',' + str(mean_metrics[3]) + '+/-' + str(std_metrics[3]) + ',' + str(mean_metrics[4]) + '+/-' + str(std_metrics[4]))

