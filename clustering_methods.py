from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit import DataStructs

def K_means_clustering(X, n_clusters, **kwargs):
	# Find the best number of clusters
	# Sum_of_squared_distances = []
	# K = range(1,51,2)
	# for k in K:
		# model = KMeans(n_clusters=k)
		# model.fit(X)
		# Sum_of_squared_distances.append(model.inertia_)
	# plt.plot(K, Sum_of_squared_distances, 'bx-')
	# plt.xticks(K)
	# plt.show()

	from sklearn.cluster import KMeans
	model = KMeans(n_clusters, **kwargs)
	model.fit(X)
	cluster_labels = model.labels_ + 1 
	return cluster_labels
	
def hierarchical_clustering(distance_vector, n_clusters, metric='jaccard', dendrogram=False):
	from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
	Z = linkage(distance_vector, method='ward', metric=metric, optimal_ordering=True)
	if dendrogram is True:
		dendrogram(Z)
		plt.xticks([])
		plt.savefig('h-clustering.png', dpi=300, bbox_inches='tight')
		plt.show()
	cluster_labels = fcluster(Z, n_clusters, criterion='maxclust')
	return cluster_labels
	
# def butina_clustering(X, distance_matrix, cutoff=0.8):
	# from rdkit.ML.Cluster import Butina
	# clusters = Butina.ClusterData(distance_matrix, len(X), cutoff, isDistData=True)
	# clusters = sorted(clusters, key=len, reverse=True)
	
	# # Give a short report about the numbers of clusters and their sizes
	# num_clust_g1 = sum(1 for c in clusters if len(c) <= 5)
	# num_clust_g5 = sum(1 for c in clusters if len(c) > 5)
	# num_clust_g25 = sum(1 for c in clusters if len(c) > 25)
	# num_clust_g100 = sum(1 for c in clusters if len(c) > 100)

	# print("Total num. of clusters: ", len(clusters))
	# print("Num. of clusters with 1 to 5 compounds: ", num_clust_g1)
	# print("Num. of clusters with >5 compounds: ", num_clust_g5)
	# print("Num. of clusters with >25 compounds: ", num_clust_g25)
	# print("Num. of clusters with >100 compounds: ", num_clust_g100)
	# return clusters