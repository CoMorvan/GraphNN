from mol2vec.features import mol2alt_sentence, MolSentence, sentences2vec
from gensim.models import word2vec
from rdkit import Chem


def embed_smiles(smiles):
    model = word2vec.Word2Vec.load('data/model_300dim.pkl')
    mols = (Chem.MolFromSmiles(i) for i in smiles)
    sentences = [sentences2vec(MolSentence(mol2alt_sentence(m, 1)), model, unseen='UNK') for m in mols]
    return sentences

def embed_single_smiles(smiles):
    model = word2vec.Word2Vec.load('data/model_300dim.pkl')
    mol = Chem.MolFromSmiles(smiles)
    sentences = sentences2vec(MolSentence(mol2alt_sentence(mol, 1)), model, unseen='UNK')
    return sentences


if __name__ == '__main__':
    smiles_embedded = embed_smiles(["c1ccccc1", "O=C1CCCC2=C1C1(CCS(=O)(=O)C1)N=C(Nc1nc3ccccc3o1)N2",
                                    "CN[C@@H]1C[C@H]2O[C@@](C)([C@@H]1OC)n1c3ccccc3c3c4CNC(=O)c4c4c5ccccc5n2c4c13"])
    print(len(smiles_embedded[0][0]))
