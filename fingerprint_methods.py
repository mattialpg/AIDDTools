import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from map4 import MAP4Calculator
import HLFP_generation as HLFP
from rdkit.Chem import rdMHFPFingerprint
from sklearn.decomposition import PCA
import numba

def calculate_fp(mol, method='morgan2', nBits=1024, pca=False, save=False):
	if method == 'maccs':
		FP = MACCSkeys.GenMACCSKeys(mol)

	elif 'morgan' in method:
		n = int(method.split('morgan')[1])
		FP = AllChem.GetMorganFingerprintAsBitVect(mol, n, nBits)

	elif method == 'rdkit':
		FP = AllChem.RDKFingerprint(mol)

	elif method == 'map4':
		# Calculate MAP4 fingerprint
		#? MAP4 = MAP4Calculator(nBits, is_folded=True)	# "False" raises a problem with tmap
		#? FP = MAP4.calculate(mol)
		MAP4 = MAP4Calculator(nBits)
		FP = MAP4.calculate_many([mol])[0]  # <- workaround

	elif method == 'mhfp':
		MHFP = Chem.rdMHFPFingerprint.MHFPEncoder(nBits)
		FP = MHFP.EncodeMol(mol, isomeric=True)

	elif method == 'hl':
		# Return np.array, not list!!
		# Calculate Hadipour-Liu fingerprint
		res = HLFP.mol2local(mol, onehot=True, pca=True)
		# if pca:
		atom_pca = np.array(res.f_atoms_pca)
		bond_pca = np.array(res.f_bonds_pca)
		FP = np.concatenate((atom_pca, bond_pca), axis=1)
		print(FP)
		# second_pca = PCA(n_components = 50) 
		# FP = second_pca.fit_transform(FP)
		# # else:
		# # 	atom = np.array(res.f_atoms)
		# # 	bond = np.array(res.f_bonds)
		# # 	FP = atom
		# 	# FP = np.concatenate((atom, bond), axis=1)
	else:
		FP = None

	# if save is True:
	# 	FP64 = FPs_list.ToBase64()
	# 	open(method + '.fp64','a').write(FP64 + '\n')
	return FP

# def read_fp64(filename):
# 	df = pd.read_csv(filename, sep=';\n', header=None, engine='python', names=['FP64'])
# 	list_FP = []
# 	for fp in df['FP64']:
# 		fp_from_base64 = ExplicitBitVect(nBits)
# 		fp_from_base64.FromBase64(fp)
# 		list_FP.append(fp_from_base64)
# 	return list_FP

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

"""When using MAP4 for machine learning, a custom kernel (or a custom loss function) is needed because the similarity between
two MinHashed fingerprints cannot be assessed with "standard" Jaccard, Manhattan, or Cosine functions. Using tmap MinHash is
the same as calculating the distance 1 - np.float(np.count_nonzero(np.array(fp1) == np.array(fp2))) / len(np.array(fp1))."""

import tmap as tm
from rdkit import DataStructs
def get_simmatr(fps_list, method='morgan2', nBits=1024):
	"""
	fps_list: list of explicit bit vectors (SparseBitVects not supported)
	More info at: https://github.com/cosconatilab/PyRMD/blob/main/PyRMD_v1.03.py
	"""
	
	MAP4 = MAP4Calculator(nBits)
	ENC = tm.Minhash(nBits)
	MHFP = Chem.rdMHFPFingerprint.MHFPEncoder(nBits)

	simmatr = np.zeros((len(fps_list), len(fps_list)))
	for i, fp1 in enumerate(fps_list):
		for j, fp2 in enumerate(fps_list):
			if method == 'map4':
				simmatr[i,j] = 1 - ENC.get_distance(fp1, fp2)
			elif method == 'mhfp':
				simmatr[i,j] = MHFP.Distance(fp1, fp2)
			else:
				simmatr[i,j] = DataStructs.FingerprintSimilarity(fp1, fp2)
	# simvect = GetTanimotoSimMat(fps_list)  # 1-D array containing the lower triangle elements of the distance matrix
	# simmatr = vec2matr(simvect)
# similarity_matr = similarity_matr.sort_values(by=0, axis=0, ascending=False).sort_values(by=0, axis=1, ascending=False)
	return simmatr