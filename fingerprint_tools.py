import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import rdMHFPFingerprint
from sklearn.decomposition import PCA
from .map4 import MAP4Calculator
import tmap as tm

"""
    When using map4 for machine learning, a custom kernel (or a custom loss function) is needed because the similarity between
    two MinHashed fingerprints cannot be assessed with "standard" Jaccard, Manhattan, or Cosine functions. Using tmap MinHash is
    the same as calculating the similarity: 1 - np.float(np.count_nonzero(np.array(fp1) == np.array(fp2))) / len(np.array(fp1)).

    Other ideas from: https://github.com/MunibaFaiza/tanimoto_similarities/blob/main/tanimoto_similarities.py    
"""

# Lazy initialization with a cache
_cache_fp = {}

def calculate_fp(mol, method='morgan2', nBits=2048, pca=False, save=False):
    if 'morgan' in method:
        n = int(method.split('morgan')[1])
        return AllChem.GetMorganFingerprintAsBitVect(mol, n, nBits)

    elif method == 'maccs':
        return MACCSkeys.GenMACCSKeys(mol)

    elif method == 'rdkit':
        return AllChem.RDKFingerprint(mol)

    elif method == 'map4':
        # if nBits not in _cache_fp:
        _cache_fp[nBits] = MAP4Calculator(nBits)
        map4 = _cache_fp[nBits]
        return map4.calculate(mol)

    elif method == 'mhfp':
        if nBits not in _cache_fp:
            _cache_fp[nBits] = Chem.rdMHFPFingerprint.MHFPEncoder(nBits)
            mhfp = _cache_fp[nBits]
        return mhfp.EncodeMol(mol, isomeric=True)

    elif method == 'torsion':
        # if nBits not in _cache_fp:
        _cache_fp[nBits] = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=2048, countSimulation=False)
        gttg = _cache_fp[nBits]
        return gttg.GetCountFingerprint(mol)

    # elif method == 'hl':
    #     # Return np.array, not list!!
    #     # Calculate Hadipour-Liu fingerprint
    #     res = HLFP.mol2local(mol, onehot=True, pca=True)
    #     # if pca:
    #     atom_pca = np.array(res.f_atoms_pca)
    #     bond_pca = np.array(res.f_bonds_pca)
    #     fp = np.concatenate((atom_pca, bond_pca), axis=1)
    #     print(fp)
        # second_pca = PCA(n_components = 50) 
        # fp = second_pca.fit_transform(fp)
        # # else:
        # #     atom = np.array(res.f_atoms)
        # #     bond = np.array(res.f_bonds)
        # #     fp = atom
        #     # fp = np.concatenate((atom, bond), axis=1)

    # if save is True:
    #     FP64 = FPs_list.ToBase64()
    #     open(method + '.fp64','a').write(FP64 + '\n')
    else:
        return None

# def read_fp64(filename):
#     df = pd.read_csv(filename, sep=';\n', header=None, engine='python', names=['FP64'])
#     list_FP = []
#     for fp in df['FP64']:
#         fp_from_base64 = ExplicitBitVect(nBits)
#         fp_from_base64.FromBase64(fp)
#         list_FP.append(fp_from_base64)
#     return list_FP

def vec2matr(array):
    n = len(array)
    m = int((np.sqrt(1 + 4 * 2 * n) + 1) / 2)
    arr = np.ones([m, m])
    counter=0
    for i in range(m):
        for j in range(i):
            arr[i][j] = array[counter]
            arr[j][i] = array[counter]  # 0 for low-triangular matrix
            counter+=1
    return arr

_cache_sim = {}

def get_similarity(fp1, fp2, method='morgan', nBits=2048):
    """
    Calculate similarity between two fingerprints based on the specified method.
    More info at: https://github.com/cosconatilab/PyRMD/blob/main/PyRMD_v1.03.py
    """

    if method in ['morgan2', 'maccs', 'rdkit']:
        sim = DataStructs.FingerprintSimilarity(fp1, fp2)

    elif method == 'map4':
        # if nBits not in _cache_sim:
        _cache_sim[nBits] = tm.Minhash(nBits)
        enc = _cache_sim[nBits]
        sim = 1 - enc.get_distance(fp1, fp2)

    elif method == 'mhfp':
        # if nBits not in _cache_sim:
        _cache_sim[nBits] = Chem.rdMHFPFingerprint.MHFPEncoder(nBits)
        mhfp = _cache_sim[nBits]
        sim = 1 - mhfp.Distance(fp1, fp2)

    elif method == 'torsion':
        sim = DataStructs.TanimotoSimilarity(fp1, fp2)

    else:
        raise ValueError(f"Unknown method: {method}")
    return round(sim, 4)


def get_simmatrix(fps_list, method='morgan2', nBits=2048):
    n = len(fps_list)
    similarity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):  # Only compute upper triangle
            similarity = get_similarity(fps_list[i], fps_list[j], method=method, nBits=nBits)
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity  # Mirror the value for symmetry

    return similarity_matrix


# Plot heatmap
import plotly.graph_objects as go
import plotly.express as px
def plot_heatmap(similarity_matrix, labels, savename=None):
    fig = px.imshow(similarity_matrix)#, color_continuous_scale='haline')
    fig.update_layout(height=500, width=500,
                    margin=dict(l=65, r=100, b=0, t=0),
                    coloraxis_colorbar=dict(len=0.73, thickness=20),
                    paper_bgcolor="#2d3035",
                    font=dict(color='#8a8d93'))
    fig.update_xaxes(tickvals=np.arange(len(labels)), ticktext=labels,
                     color="#2d3035")
    fig.update_yaxes(tickvals=np.arange(len(labels)), ticktext=labels,
                     color="#2d3035")
    fig.show()
    # fig.write_html('/tmp/tmp.html')
