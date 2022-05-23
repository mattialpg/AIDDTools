# Docking of a backup compound

The first step is to perform a cluster analysis of all the available structure for the target protein and find the *medoid*. In general terms, a medoid is an object of a cluster whose dissimilarity to all the objects in the cluster is minimal. In the case of proteins, the medoid is an actual protein that represents the most representative structure in the cluster.

Let's suppose we want to find the medoid of the androgen receptor (AR). To this end, we connect to the RCSB protein databank and look for "androgen receptor". To filter the results, we select in the Refinements panel only the proteins belonging to the *Homo sapiens* proteome and also select an appropriately low resolution (1.0 to 2.5 A in this case). By doing so, we are left with 64 structures which can be downloaded by clicking on the "Download all" icon on the top of the Search summary bar. In the next page, select "Data File Format: PDB" and click on "Generate File Batches for Download". At this point you can download the archives shown on the right, unzip them into the same folder, and unzip all the single structure files. At the end, you will have a folder containing 64 *.ent* files.

Open a new PyMOL session and load all the AR structures. To clean up the view, remove all chains but A with the following command:
```
PyMOL>   remove visible and not chain A
```
At this point we have to align the proteins. We can do it via the GUI by working on the first object in the list and clicking
```
GUI     action, align, all to this
```
However, this method often leaves some objects non-aligned. In this case, you have to select manually the non-aligned ones and run the following commands:
```
PyMOL>   list = cmd.get_object_list('sele')
PyMOL>   for obj in list[1:]: cmd.align(obj, list[0])
```
Now that we have an aligned cluster of proteins, we need to reduce the number of structures by deleting the structures that:
 - Are not the AR protein;
 - Clearly stick out of the cluster because have a different 3D arrangement;
 - Are mutations of the AR protein;
 - Form a non-representative sub-cluster of the AR protein;
 - Have isolated ligands.
To do so, we select the outlier objects with the cursor and run the following command:
```
PyMOL>   for l in cmd.get_object_list('sele'): cmd.delete(l)
```
With this reduced protein cluster, we can now have a look at the structure of the binding pocket. Select the cluster of ligands and show the residues within 4 A from it:
```
GUI     (sele), action, modify, around, residues within 4 A
```
