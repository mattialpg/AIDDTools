import warnings, os, sys
import numpy as np
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem.rdMHFPFingerprint import MHFPEncoder
from rdkit.Chem.AtomPairs import Pairs, Torsions


"""
    When using map4 for machine learning, a custom kernel (or a custom loss function) is needed because the similarity between
    two MinHashed fingerprints cannot be assessed with "standard" Jaccard, Manhattan, or Cosine functions. Using tmap MinHash is
    the same as calculating the similarity: 1 - np.float(np.count_nonzero(np.array(fp1) == np.array(fp2))) / len(np.array(fp1)).

    Other ideas from: https://github.com/MunibaFaiza/tanimoto_similarities/blob/main/tanimoto_similarities.py    
"""

# Cache to store fingerprint generators
_cache_fp = {}

def fp_to_numpy(fp, nBits=None):
    """Convert RDKit fingerprint (ExplicitBitVect or SparseIntVect) to numpy array."""
    if nBits is None:
        nBits = fp.GetNumBits()
    arr = np.zeros((nBits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def calculate_fp(mol, method='morgan2', nBits=2048, pca=False, as_numpy=False):
    method = method.lower()
    key = (method, nBits)

    # Morgan/ECFP
    if method.startswith(('morgan', 'ecfp')):
        radius = int(method.replace('morgan', '').replace('ecfp', ''))
        if key not in _cache_fp:
            _cache_fp[key] = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
        fp = _cache_fp[key].GetFingerprint(mol)
        return fp_to_numpy(fp, nBits) if as_numpy else fp

    # MACCS
    elif method == 'maccs':
        fp = MACCSkeys.GenMACCSKeys(mol)
        return fp_to_numpy(fp, fp.GetNumBits()) if as_numpy else fp

    # RDKit topological
    elif method == 'rdkit':
        if key not in _cache_fp:
            _cache_fp[key] = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=nBits)
        fp = _cache_fp[key].GetFingerprint(mol)
        return fp_to_numpy(fp, nBits) if as_numpy else fp

    # AtomPair
    elif method == 'atompair':
        if key not in _cache_fp:
            _cache_fp[key] = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=nBits)
        fp = _cache_fp[key].GetFingerprint(mol)
        return fp_to_numpy(fp, nBits) if as_numpy else fp

    # Topological torsion
    elif method in ['topotorsion', 'torsion']:
        if key not in _cache_fp:
            _cache_fp[key] = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=nBits)
        fp = _cache_fp[key].GetFingerprint(mol)
        return fp_to_numpy(fp, nBits) if as_numpy else fp

    # Avalon
    elif method == 'avalon':
        fp = pyAvalonTools.GetAvalonFP(mol, nBits)
        return fp_to_numpy(fp, nBits) if as_numpy else fp

    # MAP4
    elif method == 'map4':
        from map4 import MAP4Calculator
        if key not in _cache_fp:
            _cache_fp[key] = MAP4Calculator(nBits)
        fp = _cache_fp[key].calculate(mol)
        return np.array(fp) if as_numpy else fp

    # MHFP
    elif method in ['mhfp', 'mhfp6']:
        if key not in _cache_fp:
            from mhfp.encoder import MHFPEncoder
            _cache_fp[key] = MHFPEncoder(nBits)
        fp = _cache_fp[key].EncodeMol(mol, isomeric=False)
        return np.array(fp) if as_numpy else fp

    # HLFP
    elif method == 'hl':
        from . import HLFP_generation as HLFP
        res = HLFP.mol2local(mol, onehot=True, pca=True)
        atom_pca = np.array(res.f_atoms_pca)
        bond_pca = np.array(res.f_bonds_pca)
        fp = np.concatenate((atom_pca, bond_pca), axis=1)
        if pca:
            second_pca = PCA(n_components=50)
            fp = second_pca.fit_transform(fp)
        return fp if as_numpy else res  # return raw HLFP object if requested

    # Biosynfoni
    elif method == 'biosynfoni':
        from biosynfoni import Biosynfoni
        fp = Biosynfoni(mol).fingerprint
        return np.array(fp) if as_numpy else fp

    else:
        raise ValueError(f"Unknown fingerprint method: {method}")


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


def tanimoto_similarity_matrix(fp_matrix1, fp_matrix2=None):
    """
    Vectorized Tanimoto similarity for dense fingerprint arrays.
    """
    if fp_matrix2 is None:
        fp_matrix2 = fp_matrix1

    dot = np.dot(fp_matrix1, fp_matrix2.T)
    a_sum = fp_matrix1.sum(axis=1).reshape(-1, 1)
    b_sum = fp_matrix2.sum(axis=1).reshape(1, -1)
    denom = a_sum + b_sum - dot
    return dot / np.maximum(denom, 1e-9)


_cache_sim = {}
def get_similarity(fp1, fp2, method='morgan2', nBits=2048):
    """
    Calculate similarity between two fingerprints based on the specified method.
    More info at: https://github.com/cosconatilab/PyRMD/blob/main/PyRMD_v1.03.py
    """
    key = (method, nBits)

    # Methods using RDKit FingerprintSimilarity (Tanimoto)
    if method in ['morgan2', 'morgan3', 'maccs', 'rdkit', 'torsion', 'topotorsion', 'avalon', 'atompair']:
        return round(DataStructs.FingerprintSimilarity(fp1, fp2), 3)

    elif method == 'map4':
        import tmap as tm
        if key not in _cache_sim:
            _cache_sim[key] = tm.Minhash(nBits)
        enc = _cache_sim[key]
        return round(1 - enc.get_distance(fp1, fp2), 3)

    elif method in ['mhfp', 'mhfp6']:
        if key not in _cache_sim:
            _cache_sim[key] = MHFPEncoder(nBits)
        mhfp = _cache_sim[key]
        return round(1 - mhfp.Distance(fp1, fp2), 3)


def count_tanimoto(mat1, mat2, block_size=500):
    """
    Compute counted (multiset) Tanimoto similarity matrix in blocks with a tqdm progress bar.
    mat1: shape (n1, d)
    mat2: shape (n2, d)
    block_size: number of rows from mat1 to process at a time
    Returns: (n1, n2) similarity matrix
    """
    n1, d = mat1.shape
    n2 = mat2.shape[0]
    sim_matrix = np.zeros((n1, n2), dtype=float)

    for start in tqdm(range(0, n1, block_size), desc="Counted Tanimoto", unit="block"):
        end = min(start + block_size, n1)
        block1 = mat1[start:end, :, None]   # shape (block, d, 1)
        block2 = mat2.T[None, :, :]        # shape (1, d, n2)

        mins = np.minimum(block1, block2).sum(axis=1)
        maxs = np.maximum(block1, block2).sum(axis=1)
        sim_matrix[start:end, :] = np.where(maxs == 0, 0.0, mins / maxs)

    return sim_matrix


def get_similarity_matrix(fps_list1, fps_list2=None, method='morgan2', nBits=2048, block_size=500):
    """
    Compute similarity matrix between two fingerprint lists.
    - Binary fingerprints: vectorized Tanimoto
    - Counted fingerprints: blockwise vectorized Tanimoto with tqdm progress
    - Other methods: fallback to pairwise get_similarity
    """
    if fps_list2 is None:
        fps_list2 = fps_list1

    binary_methods = {'morgan2', 'morgan3', 'maccs', 'rdkit', 'ecfp2'}

    # Case 1: binary methods (vectorised)
    if method.lower() in binary_methods:
        print("\n*** Using vectorized Tanimoto similarity calculation ***\n")
        mat1 = np.array([fp_to_numpy(fp) for fp in fps_list1])
        mat2 = mat1 if fps_list1 is fps_list2 else np.array([fp_to_numpy(fp) for fp in fps_list2])
        sim_mat = tanimoto_similarity_matrix(mat1, mat2)
        return np.round(sim_mat, 3)

    # Case 2: counted Tanimoto
    if method.lower() == 'biosynfoni':
        print("\n*** Using blockwise counted Tanimoto similarity calculation ***\n")
        mat1 = np.stack(fps_list1.to_numpy(), axis=0).astype(int)
        mat2 = mat1 if fps_list1 is fps_list2 else np.stack(fps_list2.to_numpy(), axis=0).astype(int)
        sim_mat = count_tanimoto(mat1, mat2, block_size=block_size)
        return np.round(sim_mat, 3)

    # Case 3: fallback (other similarity methods)
    sim_matrix = np.zeros((len(fps_list1), len(fps_list2)))
    for i in range(len(fps_list1)):
        for j in range(len(fps_list2)):
            if fps_list1 is fps_list2 and j < i:
                sim_matrix[i, j] = sim_matrix[j, i]
            else:
                sim = get_similarity(fps_list1[i], fps_list2[j], method=method, nBits=nBits)
                sim_matrix[i, j] = sim
    return np.round(sim_matrix, 3)


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
