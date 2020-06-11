import os
import shutil
import sys
import timeit

import numpy as np
import pandas as pd
import torch
from C_SGEN import mydataset, C_SGEN, Trainer
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

    for dataset in ['davis', 'kiba']:
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

        smiles_train = list(np.load(dir + 'train/smiles.npy'))
        length = (len(smiles_train)//batch)*batch
        smiles_train = smiles_train[:length]
        proteins_seq_train = list(np.load(dir + 'train/proteins_seq.npy'))
        proteins_seq_train = proteins_seq_train[:length]
        smiles_test = list(np.load(dir + 'test/smiles.npy'))
        length_test = (len(smiles_test)//batch)*batch
        smiles_test = smiles_test[:length_test]
        proteins_seq_test = list(np.load(dir + 'test/proteins_seq.npy'))
        proteins_seq_test = proteins_seq_test[:length_test]
        error_drug = {}
        error_prot = {}

        metric_list = []

        for train_num in range(training_number):
            mean_error_drug = {}
            mean_error_prot = {}

            best_RMSE = float('inf')
            best_loss = float('inf')
            best_epoch = 0

            for epoch in range(1, (iteration + 1)):
                start = timeit.default_timer()

                train_loss, pred_train, true_train = trainer.train(train_loader)
                test_loss, RMSE_test, predicted_test, true_test = tester.test(test_loader)
                print(pred_train[0])

                end = timeit.default_timer()
                time = end - start

                print('epoch:{:d} - train loss: {:.3f}, test loss: {:.3f}, test rmse: {:.3f}, time: {:.3f}'.format(
                    epoch, train_loss, test_loss, RMSE_test, time))
                if RMSE_test < best_RMSE:
                    best_RMSE = RMSE_test
                    best_loss = test_loss
                    best_epoch = epoch
                    best_predicted, best_true = predicted_test, true_test
                    best_train_predicted, best_train_true = pred_train, true_train
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


                    best_train_predicted, best_train_true = list(best_train_predicted), list(best_train_true)
                    best_predicted, best_true = list(best_predicted), list(best_true)


                    for i in range(len(smiles_train)):

                        try:
                            mean_error_drug[smiles_train[i]] += [abs(best_train_predicted[i] - best_train_true[i])]
                        except KeyError:
                            mean_error_drug[smiles_train[i]] = [abs(best_train_predicted[i] - best_train_true[i])]
                        try:
                            mean_error_prot[proteins_seq_train[i]] += [abs(best_train_predicted[i] - best_train_true[i])]
                        except KeyError:
                            mean_error_prot[proteins_seq_train[i]] = [abs(best_train_predicted[i] - best_train_true[i])]

                    for i in range(len(smiles_test)):
                        try:
                            mean_error_drug[smiles_test[i]] += [abs(best_predicted[i] - best_true[i])]
                        except KeyError:
                            mean_error_drug[smiles_test[i]] = [abs(best_predicted[i] - best_true[i])]
                        try:
                            mean_error_prot[proteins_seq_test[i]] += [abs(best_predicted[i] - best_true[i])]
                        except KeyError:
                            mean_error_prot[proteins_seq_test[i]] = [abs(best_predicted[i] - best_true[i])]

                    if train_num == 0:
                        for key in mean_error_drug.keys():
                            error_drug[key] = [np.mean(mean_error_drug[key])]

                        for key in mean_error_prot.keys():
                            error_prot[key] = [np.mean(mean_error_prot[key])]
                    else:
                        for key in mean_error_drug.keys():
                            error_drug[key] += [np.mean(mean_error_drug[key])]

                        for key in mean_error_prot.keys():
                            error_prot[key] += [np.mean(mean_error_prot[key])]

        error_drug_list = []
        error_prot_list = []

        for key in error_drug.keys():
            mean = np.mean(error_drug[key])
            std = np.std([error_drug[key]])
            error_drug_list.append([mean, std, key])

        for key in error_prot.keys():
            mean = np.mean(error_prot[key])
            std = np.std([error_prot[key]])
            error_prot_list.append([mean, std, key])

        error_drug_list.sort(key = lambda x: x[0])
        error_prot_list.sort(key = lambda x: x[0])

        plot_error(error_drug_list, error_prot_list, dataset)
        with open(dir + 'plots_' + dataset + '/max_error.tsv', 'w') as f:
            f.write('Drug'+'\t'+'Mean error +/- std\n')
            for i in range(10, 0, -1):
                f.write(str(error_drug_list[-i][2]) + '\t' + str(error_drug_list[-i][0]) + '+/-' + str(error_drug_list[-i][1]) + '\n')
            f.write('Proteins'+'\t'+'Mean error +/- std\n')
            for i in range(10, 0, -1):
                f.write(str(error_prot_list[-i][2]) + '\t' + str(error_prot_list[-i][0]) + '+/-' + str(error_prot_list[-i][1]) + '\n')



        mean_metrics = np.mean(metric_list, axis=0)
        std_metrics = np.std(metric_list, axis=0)
        with open(dir + 'plots_' + dataset + '/best_metrics.csv', 'w') as o:
            o.write('Model,MSE,RMSE,Concordance Index,Spearman,Pearson\n')
            o.write('C-SGEN' + ',' + str(mean_metrics[0]) + '+/-' + str(std_metrics[0]) + ',' + str(mean_metrics[1]) + '+/-' + str(std_metrics[1]) + ',' + str(mean_metrics[2]) + '+/-' + str(std_metrics[2]) + ',' + str(mean_metrics[3]) + '+/-' + str(std_metrics[3]) + ',' + str(mean_metrics[4]) + '+/-' + str(std_metrics[4]))

