import os
import sys

import pandas as pd
from rdkit.Chem.PandasTools import LoadSDF


def get_SMILES_scores(sdf_filename):
    """
    Gets smiles and the highest scorpion score from a scorpion sdf file
    :param sdf_filename: the sdf file to read from
    :return: the found smiles, the found score, and the file if an error occured, or None otherwise
    """
    df = LoadSDF(sdf_filename, smilesName='SMILES')
    file = None
    try:
        score = max([float(x) for x in df["TOTAL"]])
        SMILES = [df["SMILES"]][0][0]
    except KeyError:
        SMILES = None
        score = None
        file = sdf_filename
    return SMILES, score, file


if __name__ == "__main__":
    info = {"SMILES": [], "score": []}
    defect = []
    dir = 'data/' + sys.argv[1] + '/'
    count = 0
    for subdir, dirs, files in os.walk(dir):
        for filename in files:
            filepath = subdir + os.sep + filename
            if filepath.endswith("3d_scorp.sdf"):
                sm, sc, filename = get_SMILES_scores(filepath)
                info["SMILES"].append(sm)
                info["score"].append(sc)
                if sm == None:
                    print(filepath)
                else:
                    count += 1
                    print('\r{} {}'.format('Number of smiles found:', count), end='\r')
    print()
    info = pd.DataFrame(info)
    info = info.dropna()
    info.to_csv(dir + sys.argv[1] + '.csv', index=False)
    print('CSV file created at', dir)
