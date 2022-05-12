# MD Simulations

1. Create .mol2 and .frcmod files for all ligands:
```bash
antechamber -i substrate.esp -fi gesp -o substrate.mol2 -fo mol2 -rn CAR -c resp

parmchk2 -i substrate.mol2 -f mol2 -s 2 -o substrate.fcrmod
```
2. Build library for ligands wth *tleap*:
```bash
tleap -f CAR_noncov.inp
```
3. Open the mo2 file with pymol together with the model pdb. Do:
```pymol
Wizard > Pair fit
```

