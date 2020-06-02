import math
import os
import pickle
import timeit

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from utils import printProgressBar, rms_score

decay_interval = 10
ch_num = 4
window = 5
batch = 32
k = 16
lr = 5e-3
C_SGEN_layers = 2
fingerprint_size = 198
maxNumAtoms = 64
prot_len = 100


def load_tensor(file_name, dtype):
    return [dtype(d).to(torch.device('cuda')) for d in np.load((file_name + '.npy'))]


class mydataset(torch.utils.data.Dataset):

    def __init__(self, dataset, dir):
        self.Features = load_tensor(dir + dataset + '/Features', torch.FloatTensor)
        self.Normed_adj = load_tensor(dir + dataset + '/Normed_adj', torch.FloatTensor)
        self.Fringer = load_tensor(dir + dataset + '/fingerprint_stand', torch.FloatTensor)
        self.interactions = load_tensor(dir + dataset + '/Interactions', torch.FloatTensor)
        self.proteins = load_tensor(dir + dataset + '/proteins', torch.FloatTensor)

        self.dataset = list(zip(np.array(self.Features), np.array(self.Normed_adj), np.array(self.Fringer), np.array(self.proteins), np.array(self.interactions)))

    def __getitem__(self, item):
        data_batch = self.dataset[item]

        return data_batch

    def __len__(self):
        return len(self.interactions)


class C_SGEN(nn.Module):
    def __init__(self):
        super(C_SGEN, self).__init__()
        self.layer1 = nn.Linear(75, 4 * ch_num)
        self.dropout_feature = nn.Dropout(p=0.5)
        self.dropout_adj = nn.Dropout(p=0.5)

        self.conv1d_1 = nn.Conv1d(in_channels=4 * ch_num, out_channels=5 * ch_num // 2, kernel_size=(k + 1) // 2 + 1)
        self.conv1d_2 = nn.Conv1d(in_channels=5 * ch_num // 2, out_channels=ch_num, kernel_size=k // 2 + 1)
        self.bn = nn.BatchNorm1d(ch_num)

        self.conv1d_3 = nn.Conv1d(in_channels=5 * ch_num, out_channels=6 * ch_num // 2, kernel_size=(k + 1) // 2 + 1)
        self.conv1d_4 = nn.Conv1d(in_channels=6 * ch_num // 2, out_channels=ch_num, kernel_size=k // 2 + 1)

        self.conv1d_5 = nn.Conv1d(in_channels=6 * ch_num, out_channels=7 * ch_num // 2, kernel_size=(k + 1) // 2 + 1)
        self.conv1d_6 = nn.Conv1d(in_channels=7 * ch_num // 2, out_channels=ch_num, kernel_size=k // 2 + 1)

        self.conv1d_7 = nn.Conv1d(in_channels=7 * ch_num, out_channels=8 * ch_num // 2, kernel_size=(k + 1) // 2 + 1)
        self.conv1d_8 = nn.Conv1d(in_channels=8 * ch_num // 2, out_channels=ch_num, kernel_size=k // 2 + 1)

        self.conv1d_9 = nn.Conv1d(in_channels=8 * ch_num, out_channels=9 * ch_num // 2, kernel_size=(k + 1) // 2 + 1)
        self.conv1d_10 = nn.Conv1d(in_channels=9 * ch_num // 2, out_channels=ch_num, kernel_size=k // 2 + 1)

        self.conv1d_11 = nn.Conv1d(in_channels=9 * ch_num, out_channels=10 * ch_num // 2, kernel_size=(k + 1) // 2 + 1)
        self.conv1d_12 = nn.Conv1d(in_channels=10 * ch_num // 2, out_channels=ch_num, kernel_size=k // 2 + 1)

        self.layer2 = nn.Linear((4 + C_SGEN_layers) * ch_num, ch_num)

        self.predict_property = nn.Linear(ch_num + fingerprint_size + prot_len, 1)
        self.conv1d = nn.Conv1d(batch * maxNumAtoms, k, 3, stride=1, padding=1)

        self.W_cnn = nn.ModuleList([nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=2 * window + 1,
            stride=1, padding=window) for _ in range(3)])

        self.cnn_line = nn.Linear(fingerprint_size, fingerprint_size)

        self.dnn1 = nn.Linear(fingerprint_size, 512)
        self.dnn2 = nn.Linear(512, 1024)
        self.dnn3 = nn.Linear(1024,1024)
        self.dnn4 = nn.Linear(1024,1024)
        self.dnn5 = nn.Linear(1024,1024)
        self.dnn_end = nn.Linear(1024, fingerprint_size)

        self.dnn1_prot = nn.Linear(100, 512)
        self.dnn_protend = nn.Linear(1024, 100)

    def pad(self, matrices, value):
        """Pad adjacency matrices for batch processing."""
        device = torch.device('cuda')
        sizes = [d.shape[0] for d in matrices]
        D = sum(sizes)
        pad_matrices = value + np.zeros((D, D))
        m = 0
        for i, d in enumerate(matrices):
            s_i = sizes[i]
            pad_matrices[m:m + s_i, m:m + s_i] = d.cpu()
            m += s_i
        return torch.FloatTensor(pad_matrices).to(device)

    def sum_axis(self, xs, axis):
        y = list(map(lambda x: torch.sum(x, 0), torch.split(xs, axis)))
        return torch.stack(y)

    def simple_conv1(self, Normed_adj, Features):
        adj_m = self.dropout_adj(Normed_adj)
        outs = self.dropout_feature(Features)
        outs = self.layer1(outs)
        outs = torch.matmul(adj_m, outs)
        return outs

    def DNN(self, x_words):

        x_words = F.relu(self.dnn1(x_words))
        x_words = F.relu(self.dnn2(x_words))
        x_words = self.dnn_end(x_words)

        return x_words

    def protDNN(self, proteins):

        proteins = F.relu(self.dnn1_prot(proteins))
        proteins = F.relu(self.dnn2(proteins))
        proteins = F.relu(self.dnn3(proteins))
        proteins = self.dnn_protend(proteins)

        return proteins

    def cnn_process(self, x, layer):
        """Controlled experiments, CNN Processing Molecular Fingerprints."""
        for i in range(layer):
            hs = self.cnn(x, i)
            x = torch.relu(self.cnn_line(hs))
        return x

    def cnn(self, xs, i):
        xs = torch.unsqueeze(torch.unsqueeze(xs, 0), 0)
        hs = torch.relu(self.W_cnn[i](xs))
        return torch.squeeze(torch.squeeze(hs, 0), 0)

    def conv1d_spatial_graph_matrix(self, adj_m, fea_m):
        """
        After the 1-d convolution processes spatial graph matrix,
        it is concatenated with the initial atomic features.
        """
        adj_m_graph1 = torch.unsqueeze(adj_m, 1)
        fea_m_graph1 = torch.unsqueeze(fea_m, -1)
        feas = torch.mul(adj_m_graph1, fea_m_graph1)
        feas = feas.permute(2, 1, 0)
        features = feas.permute(0, 2, 1)

        spatial_feature = self.conv1d(features)
        spatial_feature = spatial_feature.permute(0, 2, 1)
        outs = torch.cat([fea_m_graph1, spatial_feature], 2)
        outs = outs.permute(0, 2, 1)
        return outs

    def C_SGEL1(self, Normed_adj, input_feature):
        adj_m = self.dropout_adj(Normed_adj)
        outs = self.conv1d_spatial_graph_matrix(adj_m, input_feature)
        outs = self.dropout_feature(outs)
        outs = outs.permute(0, 2, 1)
        outs_conv1d_1 = self.conv1d_1(outs)
        outs_conv1d_1_relu = F.relu(outs_conv1d_1)
        outs_conv1d_1_relu_drought = self.dropout_feature(outs_conv1d_1_relu)
        outs_conv1d_2 = self.conv1d_2(outs_conv1d_1_relu_drought)
        outs_conv1d_2_relu = F.relu(outs_conv1d_2)
        outs_conv1d_2_permute = outs_conv1d_2_relu.permute(0, 2, 1)
        outs_conv1d_2_permute_unbind = torch.unbind(outs_conv1d_2_permute, dim=1)
        outs_conv1d_2_batch_norm = self.bn(outs_conv1d_2_permute_unbind[0])
        return outs_conv1d_2_batch_norm

    def C_SGEL2(self, Normed_adj, input_feature):
        adj_m = self.dropout_adj(Normed_adj)
        outs = self.conv1d_spatial_graph_matrix(adj_m, input_feature)
        outs = self.dropout_feature(outs)
        outs = outs.permute(0, 2, 1)
        outs_conv1d_3 = self.conv1d_3(outs)
        outs_conv1d_3_relu = F.relu(outs_conv1d_3)
        outs_conv1d_3_relu_drought = self.dropout_feature(outs_conv1d_3_relu)

        outs_conv1d_4 = self.conv1d_4(outs_conv1d_3_relu_drought)
        outs_conv1d_4 = F.relu(outs_conv1d_4)
        outs_conv1d_4_permute = outs_conv1d_4.permute(0, 2, 1)
        outs_conv1d_4_permute_unbind = torch.unbind(outs_conv1d_4_permute, dim=1)
        outs_conv1d_4_batch_norm = self.bn(outs_conv1d_4_permute_unbind[0])
        return outs_conv1d_4_batch_norm

    def C_SGEL3(self, Normed_adj, input_feature):
        adj_m = self.dropout_adj(Normed_adj)
        outs = self.conv1d_spatial_graph_matrix(adj_m, input_feature)
        outs = self.dropout_feature(outs)
        outs = outs.permute(0, 2, 1)
        outs_conv1d_5 = self.conv1d_5(outs)
        outs_conv1d_5_relu = F.relu(outs_conv1d_5)
        outs_conv1d_5_relu_drought = self.dropout_feature(outs_conv1d_5_relu)
        outs_conv1d_6 = self.conv1d_6(outs_conv1d_5_relu_drought)
        outs_conv1d_6 = F.relu(outs_conv1d_6)
        outs_conv1d_6_permute = outs_conv1d_6.permute(0, 2, 1)
        outs_conv1d_6_permute_unbind = torch.unbind(outs_conv1d_6_permute, dim=1)
        outs_conv1d_6_batch_norm = self.bn(outs_conv1d_6_permute_unbind[0])
        return outs_conv1d_6_batch_norm

    def C_SGEL4(self, Normed_adj, input_feature):
        adj_m = self.dropout_adj(Normed_adj)
        outs = self.conv1d_spatial_graph_matrix(adj_m, input_feature)
        outs = self.dropout_feature(outs)
        outs = outs.permute(0, 2, 1)
        outs_conv1d_7 = self.conv1d_7(outs)
        outs_conv1d_7_relu = F.relu(outs_conv1d_7)
        outs_conv1d_7_relu_drought = self.dropout_feature(outs_conv1d_7_relu)
        outs_conv1d_8 = self.conv1d_8(outs_conv1d_7_relu_drought)
        outs_conv1d_8 = F.relu(outs_conv1d_8)
        outs_conv1d_8_permute = outs_conv1d_8.permute(0, 2, 1)
        outs_conv1d_8_permute_unbind = torch.unbind(outs_conv1d_8_permute, dim=1)
        outs_conv1d_8_batch_norm = self.bn(outs_conv1d_8_permute_unbind[0])
        return outs_conv1d_8_batch_norm

    def C_SGEL5(self, Normed_adj, input_feature):
        adj_m = self.dropout_adj(Normed_adj)
        outs = self.conv1d_spatial_graph_matrix(adj_m, input_feature)
        outs = self.dropout_feature(outs)
        outs = outs.permute(0, 2, 1)
        outs_conv1d_9 = self.conv1d_9(outs)
        outs_conv1d_9_relu = F.relu(outs_conv1d_9)
        outs_conv1d_9_relu_drought = self.dropout_feature(outs_conv1d_9_relu)
        outs_conv1d_10 = self.conv1d_10(outs_conv1d_9_relu_drought)
        outs_conv1d_10 = F.relu(outs_conv1d_10)
        outs_conv1d_10_permute = outs_conv1d_10.permute(0, 2, 1)
        outs_conv1d_10_permute_unbind = torch.unbind(outs_conv1d_10_permute, dim=1)
        outs_conv1d_10_batch_norm = self.bn(outs_conv1d_10_permute_unbind[0])
        return outs_conv1d_10_batch_norm

    def C_SGEL6(self, Normed_adj, input_feature):
        adj_m = self.dropout_adj(Normed_adj)
        outs = self.conv1d_spatial_graph_matrix(adj_m, input_feature)
        outs = self.dropout_feature(outs)
        outs = outs.permute(0, 2, 1)
        outs_conv1d_11 = self.conv1d_11(outs)
        outs_conv1d_11_relu = F.relu(outs_conv1d_11)
        outs_conv1d_11_relu_drought = self.dropout_feature(outs_conv1d_11_relu)

        outs_conv1d_12 = self.conv1d_12(outs_conv1d_11_relu_drought)

        outs_conv1d_12 = F.relu(outs_conv1d_12)
        outs_conv1d_12_permute = outs_conv1d_12.permute(0, 2, 1)
        outs_conv1d_12_permute_unbind = torch.unbind(outs_conv1d_12_permute, dim=1)
        outs_conv1d_12_batch_norm = self.bn(outs_conv1d_12_permute_unbind[0])
        return outs_conv1d_12_batch_norm

    def simple_conv2(self, Normed_adj, Features):
        adj_m = self.dropout_adj(Normed_adj)
        outs = self.dropout_feature(Features)
        outs = self.layer2(outs)
        outs = torch.matmul(adj_m, outs)
        return outs

    def forward(self, inputs, C_SGEN_layers):

        Features, Normed_adj, Fringer = list(inputs[0]), list(inputs[1]), list(inputs[2])
        proteins = inputs[3]

        axis = list(map(lambda x: len(x), Features))

        Features = torch.cat(Features)
        Normed_adj = self.pad(Normed_adj, 0)

        Fringer = list(Fringer)
        for i in range(len(Fringer)):
            Fringer[i] = torch.unsqueeze(Fringer[i], 0)
        Fringer = torch.cat(Fringer, 0)

        # Graph embedding layer
        outs1 = self.simple_conv1(Normed_adj, Features)
        outs = {}

        # Layer 1 Convolution Spatial Graph Embedding layer
        if C_SGEN_layers > 0:
            cur_outs1 = self.C_SGEL1(Normed_adj, outs1)
            # Skip connection
            outs['outs2'] = torch.cat((outs1, cur_outs1), 1)

        # Layer 2 Convolution Spatial Graph Embedding layer
        if C_SGEN_layers > 1:
            cur_outs2 = self.C_SGEL2(Normed_adj, outs['outs2'])
            # Skip connection
            outs['outs3'] = torch.cat((outs['outs2'], cur_outs2), 1)

        # Layer 3 Convolution Spatial Graph Embedding layer
        if C_SGEN_layers > 2:
            cur_outs3 = self.C_SGEL3(Normed_adj, outs['outs3'])
            # Skip connection
            outs['outs4'] = torch.cat((outs['outs3'], cur_outs3), 1)

        # Layer 4 Convolution Spatial Graph Embedding layer
        if C_SGEN_layers > 3:
            cur_outs4 = self.C_SGEL4(Normed_adj, outs['outs4'])
            # Skip connection
            outs['outs5'] = torch.cat((outs['outs4'], cur_outs4), 1)

        # Layer 5 Convolution Spatial Graph Embedding layer
        if C_SGEN_layers > 4:
            cur_outs5 = self.C_SGEL5(Normed_adj, outs['outs5'])
            # Skip connection
            outs['outs6'] = torch.cat((outs['outs5'], cur_outs5), 1)

        # Layer 6 Convolution Spatial Graph Embedding layer
        if C_SGEN_layers > 5:
            cur_outs6 = self.C_SGEL6(Normed_adj, outs['outs6'])
            # Skip connection
            outs['outs7'] = torch.cat((outs['outs6'], cur_outs6), 1)

        # Graph-gather layer
        out_final = self.simple_conv2(Normed_adj, outs['outs' + str(C_SGEN_layers + 1)])
        y_molecules = self.sum_axis(out_final, axis)

        # Deep neural network for molecular fingerprint
        Fringer = self.DNN(Fringer)

        # Concatenate molecule and fingerprint
        y_molecules = torch.cat((y_molecules, Fringer), 1)

        #proteins = self.protDNN(proteins)

        y_molecules = torch.cat((y_molecules, proteins), 1)

        # Prediction of Molecular Properties by Fully Connected Layer
        z_molecules = self.predict_property(y_molecules)

        return z_molecules

    def __call__(self, data_batch, C_SGEN_layers, train=True):
        inputs = data_batch[:-1]
        t_interaction = torch.squeeze(data_batch[-1])
        z_interaction = self.forward(inputs, C_SGEN_layers)

        t_interaction = torch.unsqueeze(t_interaction, 1)
        loss = F.mse_loss(z_interaction, t_interaction)

        z = z_interaction.to('cpu').data.numpy()
        t = t_interaction.to('cpu').data.numpy()
        return loss, z, t


class Trainer(object):
    def __init__(self, model, C_SGEN_layers):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=5e-4)
        self.C_SGEN_layers = C_SGEN_layers

    def train(self, train_loader):
        loss_total = 0
        num = 0
        print('Training')
        all_p = []
        length = len(train_loader)
        for inc, data in enumerate(train_loader):
            num += 1
            self.optimizer.zero_grad()
            loss, pred, true = self.model(data, C_SGEN_layers=self.C_SGEN_layers)

            all_p.append(list(pred.flatten()))

            loss.backward()
            self.optimizer.step()
            loss_total += loss.to('cpu').data.numpy()

            printProgressBar(inc + 1, length, prefix='Progress', suffix='Complete')

        loss_mean = loss_total / num
        return loss_mean, pred, true

    def test(self, test_loader):

        loss_total = 0
        all_p = []
        all_t = []
        length = len(test_loader)
        print('Testing')
        num = 0
        for inc, data in enumerate(test_loader):
            num += 1
            loss, predicted, true = self.model(data, C_SGEN_layers=self.C_SGEN_layers)
            all_p += list(predicted.flatten())
            all_t += list(true.flatten())
            loss_total += loss.to('cpu').data.numpy()

            printProgressBar(inc + 1, length, prefix='Progress', suffix='Complete')

        loss_mean = loss_total / num
        RMSE = rms_score(all_t, all_p)
        return loss_mean, RMSE, all_p, all_t
