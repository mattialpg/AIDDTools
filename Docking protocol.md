# Docking of a backup compound

The first step is perform a cluster analysis of all the available structure for the target protein and find the *medoid*. In general terms, a medoid is an object of a cluster whose dissimilarity to all the objects in the cluster is minimal. In the case of proteins, the medoid is an actual protein that represents the most representative structure in the cluster.

Let's suppose we want to find the medoid of the androgen receptor (AR). To this end, we connect to the RCSB protein databank and look for "androgen receptor". To filter the results, we select in the Refinements panel only the proteins belonging to the *Homo sapiens* proteome and also select an appropriately low resolution (1.0 to 2.5 A in this case). By doing so we are left with 64 structures which can be dowloaded by clicking on the "Download all" icon on the top of the Search summary bar.  
