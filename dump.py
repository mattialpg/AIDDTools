# Keep conformers with energy within 1 kcal from the most stable conformation
energies = [tpl[1] for tpl in AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0, mmffVariant='MMFF94s')]
energies = [(id, x) for id, x in enumerate(energies) if abs(x-min(energies)) >= 1]
confids = [tpl[0] for tpl in energies]

# Save conformers in an SDF file
AllChem.AlignMolConformers(mol)
w = Chem.SDWriter('C:/Users/Idener/MEGA/DEVSHEALTH/Q1_FragLIB/confs.sdf')
for confid in range(mol.GetNumConformers()):
    w.write(mol, confId=confid)
w.close()

dists = []
for confid in confids:
    coord_0 = np.array(outmol.GetConformer(confid).GetAtomPosition(dummy_1))
    coord_1 = np.array(outmol.GetConformer(confid).GetAtomPosition(dummy_2))
    dists.append(float(pdist(np.vstack((coord_0, coord_1))).round(3)))
print(dists)