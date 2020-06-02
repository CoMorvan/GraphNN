import os
import pickle
import timeit
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from import_data import load_data
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset, DataLoader

from utils import *
from model import C_SGEN, Trainer, T


iteration = 100
batch = 8
C_SGEN_layers = 6
dataset_name = 'davis'

if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    dir = "./data/"

    train_dataset = mydataset('train', dir+dataset_name+'/')
    test_dataset = mydataset('test', dir+dataset_name+'/')

    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False, drop_last=True)

    setting = 'dataset:' + str(dataset_name) + '\n' + \
            'dir:' + str(dir) + '\n' +  \
            'batch:' + str(batch) + '\n' +  \
            'C-SGEN layers:' + str(C_SGEN_layers) + '\n' +  \
            'epochs:' + str(iteration)

    print(setting,'\n')

    model = C_SGEN().to(torch.device('cuda'))

    trainer = Trainer(model.train(), C_SGEN_layers)
    tester = T(model.eval(), C_SGEN_layers)

    Best_MSE = 100

    for epoch in range(1, (iteration + 1)):
        train_loss = trainer.train(train_loader)
        test_loss, RMSE_test, predicted_test, true_test = tester.test(test_loader)
        print('Epoch:', epoch, 'MSE:', test_loss)

        if test_loss<Best_MSE:
            Best_MSE = test_loss
            Best_epoch = epoch
            T_val, P_val = np.array(true_test), np.array(predicted_test)
            pear = pearson(T_val,P_val)
            spear = spearman(T_val,P_val)
            ci_value = ci(T_val, P_val)
            print('MSE improved to', Best_MSE, 'Pearson:', pear, 'Spearman:', spear, 'CI:', ci_value, '\n')
            plots(T_val, P_val, label=dataset_name+' '+str(test_loss), save=True)
        else:
            print('MSE did not improved since epoch', Best_epoch, '\n', 'MSE:', Best_MSE, 'CI:', ci_value)




