"""
This Python module defines several distance measures functions to be used in confidence estimation algorithm. 

Available distance measures:
  - Euclidean distance
  - Manhattan distance
  - Chebyshev distance
  - Lorentzian distance
  - Canberra distance
  - Bray Curtis distance
  - Cosine distance
  - Chord distance
  - Jaccard distance
  - Dice distance
  - Squared Chord distance
  - Vicis symmetric distance
  - Divergence distance
  - Clark distance
  - Rogers-Tanimoto

Usage:
The distance measures can be called by passing a 2D NumPy array to the respective function.
The functions return the distance matrix as a 1D NumPy array.

"""

import scipy.spatial.distance as dist
import numpy as np
from itertools import combinations

#-----------MINKOVSKI DISTANCE MEASURES---------------
#1-EUCLIDEAN
def euclidean(A):
    dist_vect = dist.pdist(A, "euclidean")
    return dist_vect

#2-MANHATTAN
def manhattan(A):
    dist_vect = dist.pdist(A, "cityblock")
    return dist_vect

#3-CHEBYSHEV
def chebyshev(A):
    dist_vect = dist.pdist(A, "chebyshev")
    return dist_vect

#-------------L1 DISTANCE MEASURES---------------
#4-LORENTZIAN
def lorentzian(A):
    A = np.array(A) if isinstance(A, list) else A
    dist_vect = np.zeros([len(list(combinations(range(A.shape[0]), 2)))])
    j=0
    for index in list(combinations(range(A.shape[0]), 2)):
        x = A[index[0],:]
        y = A[index[1],:]
        dist_vect[j] = np.sum(np.log(1+np.abs(x-y)))
        j+=1 
    return dist_vect

#5-CANBERRA
def canberra(A):
    dist_vect = dist.pdist(A, "canberra")
    return dist_vect

#6-SORENSEN(BRAY_CURTIS)
def braycurtis(A):
    dist_vect = dist.pdist(A, "braycurtis")
    return dist_vect

#-------------INNER PRODUCT DISTANCE MEASURES--------------
#7 COSINE
def cosine(A):
    dist_vect = dist.pdist(A, "cosine")
    return dist_vect

#8-CHORD
def chord(A):
    A = np.array(A) if isinstance(A, list) else A
    dist_vect = np.zeros([len(list(combinations(range(A.shape[0]), 2)))])
    j=0
    for index in list(combinations(range(A.shape[0]), 2)):
        x = A[index[0],:]
        y = A[index[1],:]
        top = (x*y).sum()
        bot = np.sqrt((x**2).sum()) * np.sqrt((y**2).sum())
        dist_vect[j] = np.sqrt(2-2*(top/bot))
        j+=1        
    return dist_vect

#9-JACCARD
# jaccard distance is calculated between two boolean 1-D arrays. 

def jaccard(A):
    A = np.array(A) if isinstance(A, list) else A
    dist_vect = np.zeros([len(list(combinations(range(A.shape[0]), 2)))])
    j=0
    for index in list(combinations(range(A.shape[0]), 2)):
        x = A[index[0],:]
        y = A[index[1],:]
        top = ((x-y)**2).sum()
        bot = (x**2).sum() + (y**2).sum() - (x*y).sum()
        dist_vect[j] = top/bot
        j+=1        
    return dist_vect

#10-DICE
# Dice distance is calculated between two boolean 1-D arrays. 
def dice(A):
    A = np.array(A) if isinstance(A, list) else A
    dist_vect = np.zeros([len(list(combinations(range(A.shape[0]), 2)))])
    j=0
    for index in list(combinations(range(A.shape[0]), 2)):
        x = A[index[0],:]
        y = A[index[1],:]
        top = 2*((x*y).sum())
        bot = (x**2).sum() + (y**2).sum()
        dist_vect[j] = 1 - top/bot
        j+=1        
    return dist_vect

#-------------SQUARED CHORD DISTANCE MEASURES--------------
#11-Squared Chord
def squared_chord(A, axis=0, keepdims=False):
    A = np.array(A) if isinstance(A, list) else A
    dist_vect = np.zeros([len(list(combinations(range(A.shape[0]), 2)))])
    j=0
    for index in list(combinations(range(A.shape[0]), 2)):
        p = A[index[0],:]
        q = A[index[1],:]
        p=np.maximum(p, 1e-12)
        q=np.maximum(q, 1e-12)
        dist_vect[j] = ((np.sqrt(p)-np.sqrt(q))**2).sum()
        j+=1
    return dist_vect

#-------------Vicissitude Distance Measures -------------
#12-Vicis symmetric 
def vicis_symmetric(A):
    A = np.array(A) if isinstance(A, list) else A
    dist_vect = np.zeros([len(list(combinations(range(A.shape[0]), 2)))])
    j=0
    for index in list(combinations(range(A.shape[0]), 2)):
        x = A[index[0],:]
        y = A[index[1],:]
        top = (x-y)**2
        bot = (np.min(x-y))**2
        dist_vect[j] = np.sum(top/bot)
        j+=1        
    return dist_vect

#13-DIVERGENCE
def divergence(A):
    A = np.array(A) if isinstance(A, list) else A
    dist_vect = np.zeros([len(list(combinations(range(A.shape[0]), 2)))])
    j=0
    for index in list(combinations(range(A.shape[0]), 2)):
        x = A[index[0],:]
        y = A[index[1],:]        
        dist_vect[j] = 2*((((x-y)**2)/((x+y)**2+1e-12)).sum())
        j+=1        
    return dist_vect

#14-CLARK
def clark(A):
    A = np.array(A) if isinstance(A, list) else A
    dist_vect = np.zeros([len(list(combinations(range(A.shape[0]), 2)))])
    j=0
    for index in list(combinations(range(A.shape[0]), 2)):
        x = A[index[0],:]
        y = A[index[1],:]
        dist_vect[j] = np.sqrt(np.sum(((x-y)/(np.abs(x)+np.abs(y)+1e-12))**2))
        j+=1        
    return dist_vect

#15-Squared Euclidean 
def squared_euclidean(A):
    dist_vect = dist.pdist(A, "sqeuclidean")    
    return dist_vect

#16-Averaged Euclidean
def average_euclidean(A):
    A = np.array(A) if isinstance(A, list) else A
    dist_vect = np.zeros([len(list(combinations(range(A.shape[0]), 2)))])
    j=0
    for index in list(combinations(range(A.shape[0]), 2)):
        x = A[index[0],:]
        y = A[index[1],:]
        dist_vect[j] = np.sqrt((np.sum((x-y)**2))/len(x)) #bana len(x)'e bolmek daha mantikli geliyor.
        j+=1        
    return dist_vect

#17-Squared Chi-Squared
def chi_squared(A):
    A = np.array(A) if isinstance(A, list) else A
    dist_vect = np.zeros([len(list(combinations(range(A.shape[0]), 2)))])
    j=0
    for index in list(combinations(range(A.shape[0]), 2)):
        x = A[index[0],:]
        y = A[index[1],:]
        top = (x-y)**2
        bot = np.abs(x+y)+1e-12
        dist_vect[j] = np.sum(top/bot)
        j+=1        
    return dist_vect

#18-Mean Censored Euclidean
def mean_cen_euc(A):
    A = np.array(A) if isinstance(A, list) else A
    dist_vect = np.zeros([len(list(combinations(range(A.shape[0]), 2)))])
    j=0
    for index in list(combinations(range(A.shape[0]), 2)):
        x = A[index[0],:]
        y = A[index[1],:]
        top = np.sum((x-y)**2)
        bot = len([s for s in x**2+y**2 if s != 0])
        dist_vect[j] = np.sqrt(top/bot)
        j+=1        
    return dist_vect

#19-JENSEN DIFFERENCE
from scipy.stats import entropy
from scipy import special
def jensendifference(A):
    A = np.array(A) if isinstance(A, list) else A
    dist_vect = np.zeros([len(list(combinations(range(A.shape[0]), 2)))])
    j=0
    for index in list(combinations(range(A.shape[0]), 2)):
        p = A[index[0],:]
        q = A[index[1],:]
        p=np.maximum(p, 1e-12)
        q=np.maximum(q, 1e-12)
        m = (p + q) / 2.0
        left = (special.entr(p)+special.entr(q))/2
        right = special.entr(m)
        jd = np.sum(left-right, axis=0)
        dist_vect[j] = np.abs(jd / 2.0)
        j+=1
    return dist_vect

from scipy.special import rel_entr
#20-JEFFREYS
def jeffreys(A):
    A = np.array(A) if isinstance(A, list) else A
    dist_vect = np.zeros([len(list(combinations(range(A.shape[0]), 2)))])
    j=0
    for index in list(combinations(range(A.shape[0]), 2)):
        p = A[index[0],:]
        q = A[index[1],:]
        p=np.maximum(p, 1e-12)
        q=np.maximum(q, 1e-12)
        left = p-q
        right = rel_entr(1,(q/p))
        dist_vect[j] = np.sum(left*right, axis=0)
        j+=1
    return dist_vect

#21-KullbackLeibler
def kullbackleibler(A):
    A = np.array(A) if isinstance(A, list) else A
    dist_vect = np.zeros([len(list(combinations(range(A.shape[0]), 2)))])
    j=0
    for index in list(combinations(range(A.shape[0]), 2)):
        p = A[index[0],:]
        q = A[index[1],:]
        p=np.maximum(p, 1e-12)
        q=np.maximum(q, 1e-12)
        kl = entropy(p,q)
        dist_vect[j] = kl
        j+=1        
    return dist_vect

#------------------OTHER DISTANCE MEASURES--------------
#22-Average Distance
def average_dist(A):
    manh = dist.pdist(A, "cityblock") 
    cheb = dist.pdist(A, "chebyshev")
    dist_vect = (manh + cheb)/2
    return dist_vect

#23-Whittakers index of association
def WIAD(A):
    A = np.array(A) if isinstance(A, list) else A
    dist_vect = np.zeros([len(list(combinations(range(A.shape[0]), 2)))])
    j=0
    for index in list(combinations(range(A.shape[0]), 2)):
        x = A[index[0],:]
        y = A[index[1],:]
        left = x/(x.sum())
        right = y/(y.sum())
        dist_vect[j] = np.sum(np.abs(left-right))/2
        j+=1        
    return dist_vect

#24-Squared Pearson Distance
def squared_pearson(A):
    A = np.array(A) if isinstance(A, list) else A
    dist_vect = np.zeros([len(list(combinations(range(A.shape[0]), 2)))])
    j=0
    for index in list(combinations(range(A.shape[0]), 2)):
        x = A[index[0],:]
        y = A[index[1],:]
        corr, _ = pearsonr(x, y)
        dist_vect[j] = 1-(corr**2)
        j+=1        
    return dist_vect

#25-CORRELATION
#Compute the correlation distance between two 1-D arrays.
def correlation(A):
    dist_vect = dist.pdist(A, "correlation") #formulde 2ye boluyor ama bunlar bolmemis
    return dist_vect

#26-Pearson Distance
from scipy.stats import pearsonr
def pearson(A):
    A = np.array(A) if isinstance(A, list) else A
    dist_vect = np.zeros([len(list(combinations(range(A.shape[0]), 2)))])
    j=0
    for index in list(combinations(range(A.shape[0]), 2)):
        x = A[index[0],:]
        y = A[index[1],:]
        corr, _ = pearsonr(x, y)
        dist_vect[j] = 1-(corr)
        j+=1        
    return dist_vect

#27-Motyka Distance
def motyka(A):
    A = np.array(A) if isinstance(A, list) else A
    dist_vect = np.zeros([len(list(combinations(range(A.shape[0]), 2)))])
    j=0
    for index in list(combinations(range(A.shape[0]), 2)):
        x = A[index[0],:]
        y = A[index[1],:]
        top = (np.maximum(x, y)).sum() 
        bot = (x+y).sum()
        dist_vect[j] = np.abs(top/bot)
        j+=1        
    return dist_vect

#28-HASSANAT
def hassanat(k):   
    x1 = []
    x1.extend([[p1,p2] for p1,p2 in combinations (k[:,0] ,2)])
    y1 = []
    y1.extend([[p1,p2] for p1,p2 in combinations (k[:,1] ,2)])
    dist = np.zeros(len(x1))
    for i in range(len(x1)):
        x = [x1[i]]
        y = [y1[i]]
        total = 0
        for xi, yi in zip(x, y):
            min_value = np.min([xi, yi])
            max_value = np.max([xi, yi])
            total += 1  # we sum the 1 in both cases
            if min_value >= 0:
                total -= (1 + min_value) / (1 + max_value)
            else:
                # min_value + abs(min_value) = 0, so we ignore that
                total -= 1 / (1 + max_value + abs(min_value))                
        dist[i] = total
    return dist

#29-Mutual Information Distance 
from sklearn.metrics import mutual_info_score
def mutual_information(A):
    A = np.array(A) if isinstance(A, list) else A
    dist_vect = np.zeros([len(list(combinations(range(A.shape[0]), 2)))])
    j=0
    for index in list(combinations(range(A.shape[0]), 2)):
        x = A[index[0],:]
        y = A[index[1],:]
        sim = mutual_info_score(x, y)
        dist_vect[j] = 1-sim
        j+=1        
    return dist_vect

_all_= [
    'euclidean',
    'manhattan',
    'chebyshev',
    'lorentzian',
    'canberra',
    'braycurtis',
    'cosine',
    'chord',
    'jaccard',
    'dice',
    'squared_chord',
    'vicis_symmetric',
    'divergence',
    'clark',
    'squared_euclidean',
    'average_euclidean',
    'chi_squared',
    'mean_cen_euc',
    'jensendifference',
    'jeffreys',
    'kullbackleibler',
    'average_dist',
    'WIAD',
    'squared_pearson'
    'correlation',
    'pearson',
    'motyka',
    'hassanat',
    'mutual_information']

from scipy.spatial.distance import squareform
def scipy_simmatr(B):
    simmatr = 1 - squareform(B)
    return simmatr
    
##### FIRNGERPRINT SIMILARITIES #####

from rdkit import DataStructs
metrics = {'Tani':DataStructs.TanimotoSimilarity, 'Dice':DataStructs.DiceSimilarity,
                'Cos':DataStructs.CosineSimilarity, 'Sok':DataStructs.SokalSimilarity,
                'Kul':DataStructs.KulczynskiSimilarity, 'McC':DataStructs.McConnaugheySimilarity}

def rdkit_simmatr(FPs_list, metr='Tani'):
    n = len(FPs_list) #@@
    simmatr = np.zeros((n,n))
    for i in range(1, n):
        sim = [DataStructs.FingerprintSimilarity(FPs_list[i], mol, metric=metrics[metr]) for mol in FPs_list[:i]]
        simmatr[i, :i] = sim
        simmatr[:i, i] = sim
    return simmatr