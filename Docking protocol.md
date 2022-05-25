# Docking of a backup compound

The first step is to perform a cluster analysis of all the available structure for the target protein and find the *medoid*. In general, a medoid is an object of a cluster whose dissimilarity to all the objects in the cluster is minimal. In the case of proteins, the medoid is an actual protein that represents the most representative structure in the cluster.

Suppose we want to find the medoid of the androgen receptor (AR). To this end, we connect to the RCSB protein databank and look for "androgen receptor". In the Refinements panel select *Homo sapiens* as a source organism and 1.0 to 2.5 A as a refinement resolution (or any appropriately low resolutions). In this way, we are left with 64 structures which can be downloaded by clicking on the "Download all" icon on the top of the Search summary bar. In the next page, select "Data File Format: PDB" and click on "Generate File Batches for Download". At this point download the archives shown on the right, unzip them into the same folder, and unzip all the single structure files. At the end, you will have a folder containing 64 *.ent* files.

Open a new PyMOL session and load all the AR structures. To clean up the view, remove all chains but A with the following command:
```
PyMOL>  remove visible and not chain A
```
At this point we have to align the proteins. We can do it via the GUI by working on the first object in the list and clicking
```
GUI     action, align, all to this
```
However, this method often fails at aligning some of the objects. In this case, select the non-aligned structuresr and run the following commands:
```
PyMOL>  list = cmd.get_object_list('sele')
PyMOL>  for obj in list[1:]: cmd.align(obj, list[0])
```
We now need to reduce the number of objects in the aligned cluster by deleting the structures that:
 - do not describe the androgen receptor;
 - stick out of the cluster because have a different 3D arrangement;
 - are mutated version of the androgen receptor;
 - form a non-representative sub-cluster of the protein;
 - have isolated ligands.

To do so, select the outlier objects with the cursor and run the following command:
```
PyMOL>  for l in cmd.get_object_list('sele'): cmd.delete(l)
```
At this point, we can have a closer look at the structure of the binding pocket. Select the cluster of ligands and show the residues within 4 A from it:
```
GUI     (sele), action, modify, around, residues within 4 A
```
