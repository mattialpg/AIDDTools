# Docking of a backup compound

The first step is perform a cluster analysis of all the available structure for the target protein and find the *medoid*. In general terms, a medoid is an object of a cluster whose dissimilarity to all the objects in the cluster is minimal. In the case of proteins, the medoid is an actual protein that represents the most representative structure in the cluster.

Let's suppose we want to find the medoid of the androgen receptor (AR). To this end, we connect to the RCSB protein databank and look for "androgen receptor". To filter the results, we select in the Refinements panel only the proteins belonging to the *Homo sapiens* proteome and also select an appropriately low resolution (1.0 to 2.5 A in this case). By doing so we are left with 64 structures which can be dowloaded by clicking on the "Download all" icon on the top of the Search summary bar. In the next page select "Data File Format: PDB" and click on "Generate File Batches for Download". At this point you can download the archives shown on the right, unzip them into the same folder, and unzip all the single structure files. At the end you will have a folder containing 64 *.ent* files.

Open a new PyMOL session and load all the AR structures. To clean up the view, remove all chains but A with the following command:
```
remove visible and not chain A
```
At this point we have to align the proteins. We can do it via the GUI by working on the first object in the list and clicking
```
Action > align > all to this
```
However, this method often leaves some objects non-aligned. In this case you should align them amnually one by one. As an alternative, you can use the following commands which give a better (but maybe slower) result:
```
list = cmd.get_object_list();
for obj in list[1:]: cmd.align(obj, list[0])
```
Now that we have an aligned cluster of proteins, we need to reduce the number of structures by deleting the structures that:
 - are not the AR protein;
 - clearly stick out of the cluster because have a different 3D arrangement;
 - are mutations of the AR protein;
 - form a non-representative sub-cluster of the AR protein;
 - have isolated ligands.
To do so, we select the outlier objects with the cursor and run the following command:
```
for l in cmd.get_object_list('sele'): cmd.delete(l)
```
With this reduced protein cluster, we can now have a look at the structure of the binding pocket. For this, select the ligand and show the   
