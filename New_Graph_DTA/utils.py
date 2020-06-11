import math
import random

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import linregress, spearmanr
from torch_geometric.nn import GATConv, SGConv, AGNNConv, ARMAConv
from torch_geometric.nn import global_add_pool


def rms_score(real_value, predicted_value):
    squared_error = 0
    size = len(real_value)
    for i in range(size):
        squared_error += (predicted_value[i] - real_value[i]) ** 2
    mean_squared_error = squared_error / size
    return math.sqrt(mean_squared_error)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    row = []
    for i in range(adj.shape[0]):
        sum = adj[i].sum()
        row.append(sum)
    rowsum = np.array(row)
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    a = d_mat_inv_sqrt.dot(adj)
    return a


def preprocess_adj(adj, norm=True, sparse=False):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj = adj + np.eye(len(adj))
    if norm:
        adj = normalize_adj(adj)
    return adj


def plots(x_values, y_values, filepath='./data/', save=False):
    a, b, r, p_value, std = linregress(x_values, y_values)
    print('r = {:.2}, p-value = {}, Standard error = {:.5}'.format(r, p_value, std))

    mini = min(min(x_values), min(y_values))
    maxi = max(max(x_values), max(y_values))
    margin = (maxi - mini) / 10

    fig = plt.figure(1)
    gridspec.GridSpec(2, 3)

    ax1 = plt.subplot2grid((2, 3), (0, 2))
    ax1.set_title('Real distribution')
    ax1.hist(x_values, bins=10)
    ax1.set_xlim(mini, maxi)

    ax2 = plt.subplot2grid((2, 3), (1, 2))
    ax2.set_title('Prediction distrubution')
    ax2.hist(y_values, bins=10)
    ax2.set_xlim(mini, maxi)

    ax3 = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2)
    ax3.set_title('confusion graph')
    ax3.scatter(x_values, y_values)
    ax3.set_xlabel('Real values')
    ax3.set_ylabel('Predicted values')
    ax3.set_ylim(mini - margin, maxi + margin)
    ax3.set_xlim(mini - margin, maxi + margin)
    ax3.plot([mini - margin, maxi + margin], [mini - margin, maxi + margin], 'r', label='y=x')
    ax3.plot([mini - margin, maxi + margin], [a * (mini - margin) + b, a * (maxi + margin) + b],
             label='Linear regression')
    ax3.legend()

    fig.tight_layout()
    fig.set_size_inches(w=9, h=6)
    plt.show()

    if save:
        plt.savefig(filepath)


def ci(y_raw, f_raw):
    ind = np.argsort(y_raw)
    y, f = [], []
    for increment in range(len(ind)):
        y.append(y_raw[ind[increment]])
        f.append(f_raw[ind[increment]])
    i = len(y) - 1
    j = i - 1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z + 1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i - 1
    ci = S / z
    return ci


def pearson(y, f):
    rp = np.corrcoef(y, f)[0, 1]
    return rp


def spearman(y, f):
    rs = spearmanr(y, f)[0]
    return rs


class GAT(torch.nn.Module):
    """
    Graph Attention Networks
    <https://arxiv.org/abs/1710.10903>
    """

    def __init__(self):
        super(GAT, self).__init__()
        self.conv1 = GATConv(75, 8, heads=8, dropout=0.6)
        self.conv2 = GATConv(
            8 * 8, 128, heads=1, concat=True, dropout=0.6)

        self.gather_layer = nn.Linear(128, 1)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x1 = F.dropout(x, p=0.6, training=self.training)
        x2 = F.elu(self.conv1(x1, edge_index))
        x3 = F.dropout(x2, p=0.6, training=self.training)
        x4 = self.conv2(x3, edge_index)

        y_molecules = global_add_pool(x4, batch)
        z_molecules = self.gather_layer(y_molecules)
        return z_molecules

    def __call__(self, data):
        target = torch.unsqueeze(data.y, 1)
        out = self.forward(data)
        loss = F.mse_loss(out, target)
        z = out.to('cpu').data.numpy()
        t = target.to('cpu').data.numpy()
        return loss, z, t


class SGC(torch.nn.Module):
    """
    Simplifying Graph Convolutional Networks"
    <https://arxiv.org/abs/1902.07153>
    """

    def __init__(self):
        super(SGC, self).__init__()
        self.conv1 = SGConv(
            75, 128, K=2, cached=False)
        self.gather_layer = nn.Linear(128, 1)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.gather_layer.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x1 = self.conv1(x, edge_index)

        y_molecules = global_add_pool(x1, batch)
        z_molecules = self.gather_layer(y_molecules)
        return z_molecules

    def __call__(self, data):
        target = torch.unsqueeze(data.y, 1)
        out = self.forward(data)
        loss = F.mse_loss(out, target)
        z = out.to('cpu').data.numpy()
        t = target.to('cpu').data.numpy()
        return loss, z, t


class AGNN(torch.nn.Module):
    """
    Attention-based Graph Neural Network for Semi-Supervised Learning
    <https://arxiv.org/abs/1803.03735>
    """

    def __init__(self):
        super(AGNN, self).__init__()
        self.lin1 = torch.nn.Linear(75, 16)
        self.prop1 = AGNNConv(requires_grad=False)
        self.prop2 = AGNNConv(requires_grad=True)
        self.lin2 = torch.nn.Linear(16, 64)

        self.gather_layer = nn.Linear(64, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.dropout(x, training=self.training)
        x = F.relu(self.lin1(x))
        x = self.prop1(x, edge_index)
        x = self.prop2(x, edge_index)
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)

        y_molecules = global_add_pool(x, batch)
        z_molecules = self.gather_layer(y_molecules)
        return z_molecules

    def __call__(self, data):
        target = torch.unsqueeze(data.y, 1)
        out = self.forward(data)
        loss = F.mse_loss(out, target)
        z = out.to('cpu').data.numpy()
        t = target.to('cpu').data.numpy()
        return loss, z, t


class ARMA(torch.nn.Module):
    """
    Graph Neural Networks with Convolutional ARMA Filters
    <https://arxiv.org/abs/1901.01343>
    """

    def __init__(self):
        super(ARMA, self).__init__()

        self.conv1 = ARMAConv(
            75,
            16,
            num_stacks=3,
            num_layers=2,
            shared_weights=True,
            dropout=0.25)

        self.conv2 = ARMAConv(
            16,
            64,
            num_stacks=3,
            num_layers=2,
            shared_weights=True,
            dropout=0.25,
            act=None)

        self.gather_layer = nn.Linear(64, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        y_molecules = global_add_pool(x, batch)
        z_molecules = self.gather_layer(y_molecules)
        return z_molecules

    def __call__(self, data):
        target = torch.unsqueeze(data.y, 1)
        out = self.forward(data)
        loss = F.mse_loss(out, target)
        z = out.to('cpu').data.numpy()
        t = target.to('cpu').data.numpy()
        return loss, z, t


# Print iterations progress
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def metric(RMSE_k_test):
    RMSE_mean_test = np.mean(np.array(RMSE_k_test))
    RMSE_std_test = np.std(np.array(RMSE_k_test))

    return RMSE_mean_test, RMSE_std_test

def plot_error(list_drug, list_prot, dataset):

    fig1 = plt.subplot(211)
    x = [i for i in range(len(list_drug))]
    y = [list_drug[i][0] for i in range(len(list_drug))]
    err = [list_drug[i][1] for i in range(len(list_drug))]
    labels = [list_drug[i][2] for i in range(len(list_drug))]
    fig1.errorbar(x, y, yerr = err, c='g', ls=None)
    fig1.set_xlabel('Drugs')
    fig1.set_ylabel('Mean error')

    fig2 = plt.subplot(212)
    x_prot = [i for i in range(len(list_prot))]
    y_prot = [list_prot[i][0] for i in range(len(list_prot))]
    err_prot = [list_prot[i][1] for i in range(len(list_prot))]
    labels_prot = [list_prot[i][2] for i in range(len(list_prot))]
    fig2.errorbar(x_prot, y_prot, yerr = err_prot, c='b', ls=None)
    fig2.set_xlabel('Protein')
    fig2.set_ylabel('Mean error')

    plt.savefig('data/' + dataset + '/plots_' + dataset + '/errors.png')