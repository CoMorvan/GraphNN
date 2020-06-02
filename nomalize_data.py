import pandas as pd
import numpy as np

for dataset in ['Lipophilicity', 'FreeSolv', 'ESOL', 'GA', 'PDBbind_core', 'PDBbind_refined', 'QM7']:
    dir = 'data/' + dataset + '/'
    data = pd.read_csv(dir+dataset+'.csv')
    stats = pd.read_csv(dir+'stats.csv')
    std = stats.iloc[0]['STD']
    mean = stats.iloc[0]['Mean']
    scores = np.array(data.iloc[:, -1].to_list())
    new = (scores - mean)/std
    new_data = pd.DataFrame()
    new_data['smiles'] = data.iloc[:, 0].to_list()
    new_data['values'] = new
    new_data.to_csv(dir+dataset+'_normalized.csv', index = False)