import os
import pickle
import shutil
import sys
import timeit
import pandas as pd

import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data import DataLoader
from torch_geometric.transforms import AddSelfLoops
from utils import *

training_number = 5
batch = 32
iteration = 33
lr = 0.01
device = torch.device('cuda')


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


def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy')]


class TestDataset(InMemoryDataset):
    def __init__(self, data_list):
        super(TestDataset, self).__init__('/tmp/TestDataset')
        self.data, self.slices = self.collate(data_list)

    def _download(self):
        pass

    def _process(self):
        pass


def load_dataset(dataset, name):
    with open('./data/' + name + '/' + dataset + '/full_feature', 'rb') as node_features:
        x_train = pickle.load(node_features)
    with open('./data/' + name + '/' + dataset + '/edge', 'rb') as f:
        edge_index_train = pickle.load(f)
    y_train = load_tensor('./data/' + name + '/' + dataset + '/Interactions', torch.FloatTensor)

    d = []
    for i in range(len(y_train)):
        data = Data(x=x_train[i], edge_index=edge_index_train[i], y=y_train[i])
        data = AddSelfLoops()(data)
        data.atom_num = x_train[i].shape[0]
        d.append(data)
    set = TestDataset(d)
    return set


class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    def train(self, train_loader, std, mean):

        loss_total = 0
        num = 0
        for data in train_loader:
            num += 1
            data = data.to(device)
            self.optimizer.zero_grad()
            loss, _, _ = self.model(data, std, mean)
            loss.backward()
            self.optimizer.step()
            loss_total += loss.to('cpu').data.numpy()

        loss_mean = loss_total / num
        return loss_mean

    def test(self, test_loader, std, mean):

        loss_total = 0
        all_p = []
        all_t = []
        num = 0
        for data in test_loader:
            num += 1
            data = data.to(device)
            loss, predicted, true = self.model(data, std, mean)

            for i in predicted:
                all_p.append(float(i))
            for i in true:
                all_t.append(float(i))
            loss_total += loss.to('cpu').data.numpy()

        RMSE = rms_score(all_t, all_p)
        loss_mean = loss_total / num
        return loss_mean, RMSE, all_t, all_p


def pyg_train(dataset):

    dir = 'data/' + dataset + '/'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device('cuda')
    metric = dataset_para[dataset]['metric']
    col = dataset_para[dataset]['col']

    stats = pd.read_csv(dir + 'stats.csv')
    size = stats.iloc[0]['Size']
    std = stats.iloc[0]['STD']
    mean = stats.iloc[0]['Mean']

    train_dataset = load_dataset('train_data', dataset)
    test_dataset = load_dataset('test_data', dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=True)

    for i in [GAT, SGC, ARMA, AGNN]:

        model = i().to(device)
        name = model.name

        if os.path.exists(dir + 'plots_' + dataset + '_' + name + '/'):
            shutil.rmtree(dir + 'plots_' + dataset + '_' + name + '/')

        os.makedirs(dir + 'plots_' + dataset + '_' + name + '/', exist_ok=True)

        trainer = Trainer(model.train())
        Tester = Trainer(model.eval())

        metric_list = []

        for _ in range(training_number):

            best_RMSE = float('inf')
            best_loss = float('inf')
            best_MAE = float('inf')
            best_epoch = 0

            for epoch in range(1, (iteration + 1)):

                start = timeit.default_timer()
                train_loss = trainer.train(train_loader, std, mean)
                test_loss, RMSE_test, true_test, predicted_test = Tester.test(test_loader, std, mean)
                MAE = mae(true_test, predicted_test)
                end = timeit.default_timer()
                time = end - start

                print('{} - epoch: {:d} - train loss: {:.3f}, test loss: {:.3f}, RMSE: {:.3f}, MAE: {:.3f}, time: {:.3f}'.format(name, epoch, train_loss, test_loss, RMSE_test, MAE, time))

                if (RMSE_test < best_RMSE and metric=='MSE') or (MAE<best_MAE and metric=='MAE'):
                    best_RMSE = RMSE_test
                    best_loss = test_loss
                    best_MAE = MAE
                    best_epoch = epoch
                    print('RMSE improved')
                    ci_value = ci(predicted_test, true_test)
                    spear = spearman(predicted_test, true_test)
                    pear = pearson(predicted_test, true_test)
                    print('CI = {:.3f}, Spearman = {:.3f}, Pearson = {:.3f}'.format(ci_value, spear, pear))
                    plots(true_test, predicted_test, (best_RMSE, best_MAE, pear, spear, ci_value, test_loss),
                        dir + 'plots_' + dataset + '_' + name + '/' + dataset + '_' + name + '_' + str(best_RMSE) + '.png', dataset, col)

                if epoch == iteration:
                    print('Best results:')
                    print('RMSE = {:f}, CI = {:.3f}, Spearman = {:.3f}, Pearson = {:.3f}'.format(best_RMSE, ci_value, spear, pear))
                    metric_list.append([best_loss, best_RMSE, ci_value, spear, pear])

        mean_metrics = np.mean(metric_list, axis=0)
        std_metrics = np.std(metric_list, axis=0)
        with open(dir + 'plots_' + dataset + '_' + name + '/best_metrics.csv', 'w') as o:
                o.write('Model,dataset,Size,MSE,RMSE,Concordance Index,Spearman,Pearson\n')
                o.write(name + ',' + dataset + ',' + str(size) + ',' + str(mean_metrics[0]) + '+/-' + str(std_metrics[0]) + ',' + str(mean_metrics[1]) + '+/-' + str(std_metrics[1]) + ',' + str(mean_metrics[2]) + '+/-' + str(std_metrics[2]) + ',' + str(mean_metrics[3]) + '+/-' + str(std_metrics[3]) + ',' + str(mean_metrics[4]) + '+/-' + str(std_metrics[4]))


if __name__ == "__main__":
    pyg_train(sys.argv[1])