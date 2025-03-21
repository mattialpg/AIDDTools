import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from copy import deepcopy
from functools import reduce
from itertools import product, combinations, permutations

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Lipinski, Descriptors

from rdkit.Chem.Draw import rdDepictor
rdDepictor.SetPreferCoordGen(True)

parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))
from CADDTools import molecular_methods as MolMtd


def flatten(list):
    flat_list = [item for sublist in list for item in sublist]
    return(flat_list)


class GraphRecomp:
    def __init__(self):
        self.multiplet = []
        self.multiplet_Mol = Chem.Mol()
        self.combo_Mols = []
        self.probe_dict = {}  # {atom_idx: probe_name}
        self.bond_dict = {}   # {(dummy_pair): neigh_distance}
        self.linker_dict = {} # {(dummy_pair): [linker_idxs]}
        self.combo_list = []  # [(probe_dummies, linker_idx, linker_dummies)]
        self.jsmiles = []
        self.jmols = []
        self.df_linkers = pd.DataFrame()

    def prepare_multiplet(self, multiplet, probe_dir):
        self.multiplet = multiplet

        # Load probes and combine them into a single Mol
        probe_mols = [Chem.MolFromMolFile(f"{probe_dir}/{x}.sdf") for x in multiplet]
        self.multiplet_Mol = reduce(Chem.CombineMols, probe_mols)

        self.probe_dict = {f"P{i}": x for i, x in enumerate(AllChem.GetMolFrags(self.multiplet_Mol))}  # {"probe_id": (probe_atoms)}
        self.probe_dict = {i: k for k, v in self.probe_dict.items() for i in v}

        #! REMEMBER: Distance are between neighbors of dummy atoms!
        dist_matrix = MolMtd.distance_matrix(self.multiplet_Mol)
        neigh_list = set(MolMtd.get_dummy_neighbor_idxs(self.multiplet_Mol))
        neigh_to_dummy = {x: [atom.GetIdx() for atom in self.multiplet_Mol.GetAtomWithIdx(x).GetNeighbors()
                              if atom.GetSymbol() == '*'] for x in neigh_list}
        
        # Get only upper-triangle matrix indices
        neigh_pairs = set(tuple(sorted(x)) for x in product(neigh_list, neigh_list) if x[0] != x[1])
        # Remove indices of atoms in the same fragment; #TODO use self.probe_dict
        neigh_pairs = [x for x in neigh_pairs if not AllChem.GetShortestPath(self.multiplet_Mol, x[0], x[1])]
        # Create a dictionary of all possible bonds between probes
        self.bond_dict = {x: dist_matrix[i, j] for i, j in neigh_pairs
                          for x in product(neigh_to_dummy[i], neigh_to_dummy[j])}

        return self.multiplet_Mol


    def prepare_linkers(self, df_linkers, n_linkers=None):
        self.df_linkers = df_linkers

        # Create a dictionary of all the linkers whose length range comprises the bond distance
        for dummy_pair, dist in self.bond_dict.items():
            df_aux = df_linkers[df_linkers['LENGTHS'].apply(lambda lengths:
                        any(a <= dist <= b for a, b in lengths.values()))]
            
            if len(df_aux) == 0:
                raise Exception#(' > No linker available')
            
            if n_linkers and n_linkers < len(df_aux):
                df_aux = df_aux.sample(n_linkers)
                # df_aux = df_aux.head(n_linkers)

            self.linker_dict[dummy_pair] = df_aux.index.tolist()
        return self.linker_dict


    def prepare_combos(self):
        # Get bond combinations
        bond_combos = [list(x) for x in combinations(self.linker_dict.keys(), len(self.multiplet)-1)]
        # Keep combinations of non-repeated dummies
        bond_combos = [x for x in bond_combos if len(flatten(x)) == len(set(flatten(x)))]  
        # Keep combinations linking all probes
        bond_combos = [x for x in bond_combos if len(set(self.probe_dict[y]
                        for y in flatten(x))) == len(self.multiplet)]
        # Sort bond_combos for easy debug
        # bond_combos = sorted([sorted(x) for x in bond_combos], key=lambda x: x)

        # Create combo_list by merging bonding information of probes and linkers
        for bond_combo in bond_combos:
            for linker_combo in product(*[self.linker_dict[x] for x in bond_combo]):
                aux = []
                for a, b in zip(bond_combo, linker_combo):
                    bond_distance = self.bond_dict[a]
                    linker_lengths = self.df_linkers.at[b, 'LENGTHS']
                    # Select dummy atom pairs satisfying the distance condition
                    linker_dummies = [list(k) for k, v in linker_lengths.items() if v[0] <= bond_distance <= v[1]]
                    aux.append([(*a, b, *x) for x in linker_dummies])
                aux = [list(x) for x in product(*aux)]  # This is for multiple dummy atom pairs satisfying the distance condition
                self.combo_list.extend(aux)

        # Expand combo_list allowing for combinations of flipped linkers
        self.combo_list = [list(x) for combo in self.combo_list
                  for x in product(*[[tpl[:-2] + perm
                  for perm in permutations(tpl[-2:], 2)]
                  for tpl in combo])]
        return self.combo_list
        

    def join_fragments_old(self):
        for combo in self.combo_list:
            combo_Mol = self.multiplet_Mol

            # Combine probes and linkers
            bonds_to_form = []
            for tpl in combo:
                linker_mol = deepcopy(self.df_linkers.at[tpl[2], 'ROMol'])

                # Update atom indices after CombineMols
                # Probe atom idxs are the same, linker atom idxs follows in order
                probe_dummy_idxs = tpl[:2]
                linker_dummy_idxs = tuple(x + combo_Mol.GetNumAtoms() for x in tpl[-2:]) 

                combo_Mol = reduce(Chem.CombineMols, [combo_Mol, linker_mol])

                # Make a list of bonds to be formed
                bonds_to_form.extend([(probe_dummy_idxs[0], linker_dummy_idxs[0]),
                                      (probe_dummy_idxs[1], linker_dummy_idxs[1])])

            #!-- Use this to check probe+linker Mol --!#
            # print(combo)
            # print(bonds_to_form)
            # print('')
            # MolMtd.show_atom_indices(combo_Mol)
            # display(MolMtd.reset_view(combo_Mol))
            #!----------------------------------------!#

            #TODO: Change chirality of neighs before adding linkers.
            #TODO: If not inverted, linkers end up in the opposite configuration

            # Join fragments
            rwMol = Chem.RWMol(combo_Mol)
            Chem.Kekulize(rwMol, clearAromaticFlags=True)

            for dummy1, dummy2 in bonds_to_form:
                neigh1 = rwMol.GetAtomWithIdx(dummy1).GetNeighbors()[0].GetIdx()  #<-- Assuming only one neighbor
                neigh2 = rwMol.GetAtomWithIdx(dummy2).GetNeighbors()[0].GetIdx()  #<-- Assuming only one neighbor
                
                #!-- Use this to check bond errors --!#
                # atnum = rwMol.GetAtomWithIdx(neigh1).GetAtomicNum()
                # rwMol.GetAtomWithIdx(neigh1).SetAtomicNum(atnum + 8)
                # atnum = rwMol.GetAtomWithIdx(neigh2).GetAtomicNum()
                # rwMol.GetAtomWithIdx(neigh2).SetAtomicNum(atnum + 8)
                #!-----------------------------------!#

                rwMol.AddBond(neigh1, neigh2, order=Chem.rdchem.BondType.SINGLE)  #! not always single!!
                # rwMol.GetAtomWithIdx(neigh1).SetNumExplicitHs(0)
                # rwMol.GetAtomWithIdx(neigh2).SetNumExplicitHs(0)

            dummy_list = [a.GetIdx() for a in rwMol.GetAtoms() if a.GetSymbol() == '*']
            for dummy in sorted(dummy_list, reverse=True):  # Reverse sorting is fundamental!
                rwMol.RemoveAtom(dummy)
            
            # display(MolMtd.reset_view(rwMol))
            rwMol = Chem.RemoveHs(rwMol)
            Chem.SanitizeMol(rwMol)
            Chem.rdmolops.SanitizeFlags.SANITIZE_NONE
            jsmile = Chem.MolToSmiles(Chem.Mol(rwMol))
            jsmile = jsmile.replace({'[C]': 'C', '[CH]': 'C'}) #<-- Temporarily fixes radical carbons
            self.jsmiles.append(jsmile) if jsmile not in self.jsmiles else self.jsmiles

        self.jmols = [Chem.MolFromSmiles(x) for x in self.jsmiles]
        return self.jmols
    
    def join_fragments(self):
        # Calculate the number of C atoms needed to link fragments
        carbon_counts = {k: round(v/1.28) - 1 for k, v in self.bond_dict.items()}

        # Get bond combinations
        bond_combos = list(combinations(self.bond_dict.keys(), len(self.multiplet)-1))
        # Keep combinations of non-repeated dummies
        bond_combos = [list(x) for x in bond_combos if len(flatten(x)) == len(set(flatten(x)))]
        # Keep combinations linking all probes
        bond_combos = [x for x in bond_combos if len(set(self.probe_dict[y]
                       for y in flatten(x))) == len(self.multiplet)]
        # Keep combinations with appropriate number of carbons
        bond_combos = [y for y in bond_combos if all(2 <= carbon_counts[x] <= 5 for x in y)]  #<-- This filters a lot of combinations

        # Combine probes and linkers
        for i in range(len(bond_combos)):
            combo_Mol = self.multiplet_Mol
            bonds_to_form = []
            for bond in bond_combos[i]:
                n_atoms = combo_Mol.GetNumAtoms()
                linker_mol = Chem.MolFromSmiles('C' * carbon_counts[bond])
                combo_Mol = reduce(Chem.CombineMols, [combo_Mol, linker_mol])

                # Make a list of bonds to be formed
                neighs = [combo_Mol.GetAtomWithIdx(bond[i]).GetNeighbors()[0].GetIdx() for i in range(2)]
                bonds_to_form.extend([(neighs[0], n_atoms),
                                      (neighs[1], combo_Mol.GetNumAtoms()-1)])
            # print(bonds_to_form)
            # display(MolMtd.show_atom_indices(MolMtd.reset_view(combo_Mol)))

            # Join fragments
            rwMol = Chem.RWMol(combo_Mol)
            Chem.Kekulize(rwMol, clearAromaticFlags=True)

            for atom1, atom2 in bonds_to_form:
                rwMol.AddBond(atom1, atom2, order=Chem.rdchem.BondType.SINGLE)
                rwMol.GetAtomWithIdx(atom1).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
                rwMol.GetAtomWithIdx(atom2).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)

            dummy_list = [a.GetIdx() for a in rwMol.GetAtoms() if a.GetSymbol() == '*']
            for dummy in sorted(dummy_list, reverse=True):  #! Reverse sorting is fundamental
                rwMol.RemoveAtom(dummy)
            
            # rwMol = Chem.RemoveHs(rwMol)
            Chem.SanitizeMol(rwMol)
            Chem.rdmolops.SanitizeFlags.SANITIZE_NONE
            jmol = Chem.Mol(rwMol)

            # Neutralize radicals
            for a in jmol.GetAtoms():
                if a.GetNumRadicalElectrons() == 1 and a.GetFormalCharge() == 1:
                    a.SetNumRadicalElectrons(0)         
                    a.SetFormalCharge(0)

            self.jsmiles.append(Chem.MolToSmiles(jmol))

        self.jmols = [Chem.MolFromSmiles(x) for x in set(self.jsmiles)]
        return self.jmols

