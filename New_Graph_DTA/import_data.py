import json, pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import numpy as np
import os
import re
import timeit

from gensim.models import KeyedVectors
import deepchem as dc
import pandas as pd
import torch
from deepchem.feat.mol_graphs import ConvMol
from numpy import flip
from rdkit import Chem
import scipy.sparse as sp
from utils import *
from load_fingerprints import load_fingerprints


def get_DT_pairs(dataset):
    fpath = 'data/' + dataset + '/'
    train_fold = json.load(open(fpath + "folds/train_fold_setting1.txt"))
    train_fold = [ee for e in train_fold for ee in e ]
    valid_fold = json.load(open(fpath + "folds/test_fold_setting1.txt"))
    ligands = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)
    proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)
    affinity = pickle.load(open(fpath + "Y","rb"), encoding='latin1')
    drugs = []
    prots = []
    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]),isomericSmiles=True)
        drugs.append(lg)
    for t in proteins.keys():
        prots.append(proteins[t])
    if dataset == 'davis':
        affinity = [-np.log10(y/1e9) for y in affinity]
    affinity = np.asarray(affinity)
    opts = ['train','test']
    for opt in opts:
        data = []
        rows, cols = np.where(np.isnan(affinity)==False)
        if opt=='train':
            rows,cols = rows[train_fold], cols[train_fold]
        elif opt=='test':
            rows,cols = rows[valid_fold], cols[valid_fold]
        for pair_ind in range(len(rows)):
            ls = []
            ls.append(drugs[rows[pair_ind]])
            ls.append(prots[cols[pair_ind]])
            ls.append(affinity[rows[pair_ind],cols[pair_ind]])
            data.append(ls)
        if opt=='train':
            train_data = pd.DataFrame(data)
        elif opt=='test':
            test_data = pd.DataFrame(data)
    print('Dataset: {}, Number of drugs: {}, Number of proteins: {}'.format(dataset, len(set(drugs)),len(set(prots))))
    return(train_data, test_data)

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


def create_adjacency(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency, dtype=float)


def save_feature(dir, Features, Normed_adj, Interactions, smiles, edge, full_feature, fingerprints):
    dir_input = dir
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


def get_feature(graphs, smiles_set, smiles_ori, scores):
    Features_decrease, adj_decrease, edge_decrease, full_feature_decrease = [], [], [], []
    Interactions, smiles, fingerprints = [], [], []
    length = len(graphs)
    n = 0
    start_feat = timeit.default_timer()
    feat_dict = {}
    smiles_set = list(smiles_set)

    for inc, x in enumerate(graphs):

        start_single = timeit.default_timer()
        smile = smiles_set[inc]
        #print(smile)
        feat_dict[smile] = []

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

        feat_dict[smile].append(np.array(iFeature_decrease)) #0 = Features_decrease
        normed_adj_decrease = preprocess_adj(adjacency_decrease)
        feat_dict[smile].append(normed_adj_decrease) #1 = adj_decrease

        # Transforms data into PyTorch Geometrics specific data format.
        index = np.array(np.where(iAdjTmp_decrease == 1))
        edge_index = torch.from_numpy(index).long()
        feat_dict[smile].append(edge_index) #2 = edge_decrease

        feature = torch.from_numpy(feature_decrease.copy()).float()
        feat_dict[smile].append(feature) #3 = full_feature_decrease


        FP = load_fingerprints(smile)
        feat_dict[smile].append(FP) #4 = fingerprints

        n += 1
        stop_feat = timeit.default_timer()
        time = stop_feat - start_feat
        time_bis = stop_feat - start_single
        eta = (time / n) * (length - n)
        #print('{} {}/{} Time: {:.3f}'.format(smile, n, length, time_bis))
        printProgressBar(n, length, prefix='Progress :', suffix='Time: {:d}s, ETA: {:d}s'.format(int(time), int(eta)))

    for i, smi in enumerate(smiles_ori):
        Features_decrease.append(feat_dict[smi][0])
        adj_decrease.append(feat_dict[smi][1])
        edge_decrease.append(feat_dict[smi][2])
        full_feature_decrease.append(feat_dict[smi][3])
        Interactions.append([scores[i]])
        smiles.append(smi)
        fingerprints.append(feat_dict[smi][4])

    return Features_decrease, adj_decrease, edge_decrease, full_feature_decrease, Interactions, smiles, fingerprints


def embed_protein(data):
    data = set(list(data))
    vectors = {}
    model = KeyedVectors.load("data/protein_embedding.model", mmap='r')
    for (j, document) in enumerate(data):
        prot_words = [word for word in re.findall(r'.{3}', document)]
        words_embed = []
        for word in prot_words:
            try:
                words_embed.append(list(model[word]))
            except KeyError:
                print(word, "will not be included in", j, "embedding")
        vectors[document] = np.mean(words_embed, axis=0)
    return vectors


def load_data(dir_inp, dataset):
    start = timeit.default_timer()
    opts = ['train','test']
    # Load dataset
    df = {}
    df['train'], df['test'] = get_DT_pairs(dataset)
    end_read = timeit.default_timer()
    print('Data read in {:.3f}s'.format(end_read-start))

    for opt in opts:
        start_fold = timeit.default_timer()
        dir = dir_inp + dataset + '/' + opt + '/'
        data = df[opt].sample(frac=1)

        smiles_raw = data.iloc[:,0].to_list()
        smiles_ori = []
        for smile in smiles_raw:
            if '.' in smile:
                smile, _ = smile.split('.', 1)
            smiles_ori.append(smile)

        smiles = set(smiles_ori)
        conv = [Chem.MolFromSmiles(smile) for smile in smiles]
        featurizer = dc.feat.graph_features.ConvMolFeaturizer()
        graphs = featurizer.featurize(conv)

        scores = data.iloc[:,-1].to_list()

        proteins = data.iloc[:,1].to_list()
        embeddings = embed_protein(proteins)
        embedded_proteins = [embeddings[protein] for protein in proteins]

        os.makedirs(dir, exist_ok=True)
        np.save(dir+'proteins_seq.npy', proteins)
        end_prot= timeit.default_timer()
        print('Proteins embedded in {:.3f}s'.format(end_prot-start_fold))

        # Create files of graph information
        Features_decrease, adj_decrease, edge_decrease, full_feature_decrease, Interactions, smiles, fingerprints = get_feature(graphs, smiles, smiles_ori, scores)
        save_feature(dir, Features_decrease, adj_decrease, Interactions, smiles, edge_decrease, full_feature_decrease, fingerprints)
        end_graph = timeit.default_timer()



        np.save(dir+"proteins_embedded.npy", embedded_proteins)

        print('Data imported for the', opt, 'fold, time taken: {:.2f}s'.format(end_graph - start_fold))


if __name__=="__main__":
    datasets = ['kiba', 'davis']
    for dataset in datasets:
        load_data("data/", dataset)