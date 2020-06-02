from load_data import load_data
from model_train import model_train
from pyg_train import pyg_train
from results_gathering import results_gathering
import timeit

all = ['ESOL', 'FreeSolv', 'GA', 'Lipophilicity', 'PDBbind_core', 'PDBbind_refined', 'QM7']

datasets = all

if __name__=="__main__":
    start = timeit.default_timer()
    for dataset in datasets:
        if dataset!='GA':
            load_data(dataset)
        model_train(dataset)
        pyg_train(dataset)
    results_gathering()
    end = timeit.default_timer()
    time = end-start
    hour, min, sec = int(time//3600), int((time//60)%60), int(time%60)
    print('Time: {}h {}m {}s'.format(hour, min, sec))