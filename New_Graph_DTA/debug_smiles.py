from rdkit import Chem
from import_data import *


list = ['CC(C)C(C)(O)C1CN(c2nc(-c3[nH]nc4ncccc34)c(F)cc2Cl)CCN1',
        'Cc1cc(C)cc(NC(=O)Nc2ccc(-c3cccc4onc(N)c34)cc2)c1',
        'CC1=CN2CC(=O)NN=C2C=C1',
        'O=C(Nc1nc2c(O)cccc2s1)Nc1ccccc1-c1ccccc1',
        'CCN(CC)CC#Cc1ccc2c(c1)-c1[nH]nc(-c3ccc(C#N)nc3)c1C2']



for smile in list:
    print(smile)




    mol = Chem.MolFromSmiles(smile)

    if not mol:
        raise ValueError("Could not parse SMILES string:", smile)

    featurizer = dc.feat.graph_features.ConvMolFeaturizer()
    x = featurizer.featurize([mol])[0]
    print(0, 'ok')


    # increased order
    feature_increase = x.get_atom_features()
    iAdjTmp_increase = create_adjacency(mol)
    print(1, 'ok')

    # decreased order
    # Turn the data upside down
    feature_decrease = flip(feature_increase, 0)
    iAdjTmp_decrease = flip(iAdjTmp_increase, 0)
    print(2, 'ok')

    # Obtaining fixed-size molecular input data
    iFeature_decrease, adjacency_decrease = fix_input(feature_decrease, iAdjTmp_decrease)
    print(3, 'ok')

    normed_adj_decrease = preprocess_adj(adjacency_decrease)
    print(4, 'ok')

    # Transforms data into PyTorch Geometrics specific data format.
    index = np.array(np.where(iAdjTmp_decrease == 1))
    edge_index = torch.from_numpy(index).long()
    print(5, 'ok')

    feature = torch.from_numpy(feature_decrease.copy()).float()
    print(6, 'ok')

    FP = load_fingerprints(smile)
    print(7, 'ok')
