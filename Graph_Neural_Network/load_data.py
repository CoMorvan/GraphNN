import os
import pickle
import sys
import timeit

import deepchem as dc
import numpy as np
import pandas as pd
import torch
from deepchem.data.datasets import NumpyDataset
from load_fingerprints import load_fingerprints
from numpy import flip
from rdkit import Chem
from utils import preprocess_adj, printProgressBar


def create_adjacency(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency, dtype=float)


def save_feature(dir, Features, Normed_adj, Interactions, smiles, edge, full_feature, fingerprints, dataset=None):
    dir_input = (dir + dataset + '/')
    os.makedirs(dir_input, exist_ok=True)
    np.save(dir_input + 'Features', Features)
    np.save(dir_input + 'Normed_adj', Normed_adj)
    np.save(dir_input + 'Interactions', Interactions)
    np.save(dir_input + 'smiles', smiles)
    np.save(dir_input + 'fingerprint_stand', fingerprints)

    with open(dir_input + 'edge', 'wb') as f:
        pickle.dump(edge, f)

    with open(dir_input + 'full_feature', 'wb') as a:
        pickle.dump(full_feature, a)


def fix_input(feature_array, iAdjTmp):
    "Fix number of input molecular atoms"
    maxNumAtoms = 64
    iFeature = np.zeros((maxNumAtoms, 75))
    if len(feature_array) <= maxNumAtoms:
        iFeature[0:len(feature_array), 0:75] = feature_array
    else:
        iFeature = feature_array[0:maxNumAtoms]

    adjacency = np.zeros((maxNumAtoms, maxNumAtoms))

    if len(feature_array) <= maxNumAtoms:
        adjacency[0:len(feature_array), 0:len(feature_array)] = iAdjTmp
    else:
        adjacency = iAdjTmp[0:maxNumAtoms, 0:maxNumAtoms]

    return iFeature, adjacency


def get_feature(dataset):
    Features_decrease, adj_decrease, edge_decrease, full_feature_decrease = [], [], [], []
    Interactions, smiles, fingerprints = [], [], []
    length = len(dataset)
    n = 0
    start_feat = timeit.default_timer()

    for x, label, _, smile in dataset.itersamples():

        # The smile is used to extract molecular fingerprints
        smiles.append(smile)

        interaction = label
        Interactions.append([interaction])

        mol = Chem.MolFromSmiles(smile)

        if not mol:
            raise ValueError("Could not parse SMILES string:", smile)

        # increased order
        feature_increase = x.get_atom_features()
        iAdjTmp_increase = create_adjacency(mol)

        # decreased order
        # Turn the data upside down
        feature_decrease = flip(feature_increase, 0)
        iAdjTmp_decrease = flip(iAdjTmp_increase, 0)

        # Obtaining fixed-size molecular input data
        iFeature_decrease, adjacency_decrease = fix_input(feature_decrease, iAdjTmp_decrease)

        Features_decrease.append(np.array(iFeature_decrease))
        normed_adj_decrease = preprocess_adj(adjacency_decrease)
        adj_decrease.append(normed_adj_decrease)

        # Transforms data into PyTorch Geometrics specific data format.
        index = np.array(np.where(iAdjTmp_decrease == 1))
        edge_index = torch.from_numpy(index).long()
        edge_decrease.append(edge_index)

        feature = torch.from_numpy(feature_decrease.copy()).float()
        full_feature_decrease.append(feature)

        FP = load_fingerprints(smile)
        fingerprints.append(FP)

        n += 1
        stop_feat = timeit.default_timer()
        time = stop_feat - start_feat
        eta = (time / n) * (length - n)
        printProgressBar(n, length, prefix='Progression :', suffix='Time: {}s, ETA: {}s'.format(int(time), int(eta)))

    return Features_decrease, adj_decrease, edge_decrease, full_feature_decrease, Interactions, smiles, fingerprints


def load_dataset(dir, filename):
    # Load GA dataset
    start = timeit.default_timer()
    data = pd.read_csv(filename)
    data = data.sample(frac=1)
    smiles = data.iloc[:, 0].to_list()

    conv = []
    featurizer = dc.feat.graph_features.ConvMolFeaturizer()
    for smile in smiles:
        conv.append(Chem.MolFromSmiles(smile))
    graphs = featurizer.featurize(conv)

    scores = data.iloc[:, -1].to_list()
    mean = np.mean(scores)
    std = np.std(scores)
    scores = (scores - mean)/std
    with open(dir + 'stats.csv', 'w') as o:
        o.write('Size,Mean,STD\n'+str(int(len(scores)))+','+str(mean)+','+str(std))
    step = len(smiles) // 5
    smiles_dict = {}
    smiles_dict['test_data'] = smiles[:step]
    smiles_dict['train_data'] = smiles[step + 1:]
    test_dataset = NumpyDataset(graphs[:step], scores[:step], ids=smiles[:step])
    train_dataset = NumpyDataset(graphs[step + 1:], scores[step + 1:], ids=smiles[step + 1:])

    # Create files of graph information
    Features_decrease1, adj_decrease1, edge_decrease1, full_feature_decrease1, Interactions1, smiles1, fingerprints1 = get_feature(
        train_dataset)
    Features_decrease2, adj_decrease2, edge_decrease2, full_feature_decrease2, Interactions2, smiles2, fingerprints2 = get_feature(
        test_dataset)

    save_feature(dir, Features_decrease1, adj_decrease1, Interactions1, smiles1, edge_decrease1, full_feature_decrease1,
                 fingerprints1, dataset='train_data')
    save_feature(dir, Features_decrease2, adj_decrease2, Interactions2, smiles2, edge_decrease2, full_feature_decrease2,
                 fingerprints2, dataset='test_data')
    end_graph = timeit.default_timer()
    print('Time for data loading: {0:.2f}s'.format(end_graph - start))


def load_data(dataset):
    dir = 'data/' + dataset + '/'
    filepath = dir + dataset + '.csv'
    load_dataset(dir, filepath)


if __name__ == "__main__":
    load_data(sys.argv[1])
