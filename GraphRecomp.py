import os, sys
import numpy as np
from rdkit import Chem
from copy import deepcopy

from itertools import product, combinations, chain
from functools import reduce
from rdkit.Chem.Draw import rdDepictor
rdDepictor.SetPreferCoordGen(True)

from tools import utils


def lipinski_filter(mol):
    # Filter by Lipinski rules
    from rdkit.Chem import Crippen, Lipinski, Descriptors

    violations = 0
    if Lipinski.NumHDonors(mol) > 5: violations += 1
    if Lipinski.NumHAcceptors(mol) > 10: violations += 1
    if Descriptors.MolWt(mol) > 500: violations += 1
    # if Crippen.MolLogP(mol) > 5: violations += 1

    return violations

    # if violations < 2:
    #     return True
    # return False

def reos_filter(mol):  
    # Filter by REOS rules
    from rdkit.Chem import GetFormalCharge, Lipinski, Descriptors

    violations = 0
    if not 200 < Descriptors.MolWt(mol) < 500: violations += 1
    if not -5 < Descriptors.MolLogP(mol) < 5: violations += 1
    if not 15 < mol.GetNumHeavyAtoms() < 50: violations += 1
    if not -2 < GetFormalCharge(mol) < 2: violations += 1

    if Lipinski.NumHDonors(mol) > 5: violations += 1
    if Lipinski.NumHAcceptors(mol) > 10: violations += 1
    if Descriptors.NumRotatableBonds(mol) > 8: violations += 1

    if violations < 2:
        return True
    return False


def get_ringlinkers(probe_info, df_ringlink):
    # Filter rings by distance
    ringlink_mols, ringlink_info = [], []
    for tpl in probe_info:
        dist = tpl[-1]
        df_aux = df_ringlink[df_ringlink['LENGTHS'].apply(lambda x: any(a < dist < b for a, b in x.values()))].sample(2)
        
        # Get rings mols and info
        aux_mols = []
        for i in df_aux.index.tolist():
            mol = df_aux.at[i, 'ROMol_FRAG']
            mol.SetProp('_id', f"L{i}")
            mol.SetProp('_dist', f"{dist}")
            aux_mols.append(mol)

            dum = [[k, v] for k,v in df_aux.at[i, 'LENGTHS'].items() if (v[0] < dist and dist < v[1])]
            dum = sorted(dum, key=lambda x: x[1][1] - x[1][0])[0][0]  # Keep dummies corresponding to the smallest distance interval
            dum = (f"L{i}_{dum[0]}", f"L{i}_{dum[1]}", dist)  # Include name and distance into linker dummies
            ringlink_info.append(dum)
        ringlink_mols.append(aux_mols)
    return ringlink_mols, ringlink_info


from functools import reduce
def prepare_multiplet(multiplet, dir):
    probe_mols = []
    coord_dict = {}
    for probe in multiplet:
        mol = Chem.MolFromMolFile(f"{dir}/{probe}.sdf")
        mol.SetProp('_id', f"P{probe.split('_')[2]}")
        probe_mols.append(mol)
        
        # Make neighbor coordinate dictionary
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == '*':
                neigh = atom.GetNeighbors()# if a.GetNeighbors().GetAtomicNum() != 1
                if len(neigh) > 1: raise KeyError
                c_aux = mol.GetConformer().GetAtomPosition(neigh[0].GetIdx())
                coords = np.array((c_aux.x, c_aux.y, c_aux.z))
                coord_dict[f"{mol.GetProp('_id')}_{atom.GetIdx()}"] = coords
                atom.SetProp('_label', f"{mol.GetProp('_id')}_{atom.GetIdx()}")
            else:
                atom.SetProp('_label', '')

    # Get all possible bonds between probes
    probe_info = []
    dummies = list(coord_dict.keys())
    for combo in combinations(dummies, 2):
        if combo[0].split('_')[0] == combo[1].split('_')[0]:  # Skip combo of dummies in the same fragment
            continue
        dist = np.linalg.norm(coord_dict[combo[0]] - coord_dict[combo[1]])
        if dist < 9:
            probe_info.append((combo[0], combo[1], dist.round(2)))

    pairs = [tpl[:2] for tpl in probe_info]
    pair_combos = [list(x) for x in combinations(pairs, len(multiplet)-1)]  # Get combinations of pairs
    pair_combos = [x for x in pair_combos if len(utils.flatten(x)) == len(set(utils.flatten(x)))]  # Keep combos with non-repeated dummies
    # pair_combos = [x for x in pair_combos if len(x) == max([len(y) for y in pair_combos])]  # Keep longest combos
    pair_combos = [x for x in pair_combos if len(set([y.split('_')[0] for y in utils.flatten(x)])) == len(multiplet)]  # Keep combos covering all probes
    pair_combos = [[[x for x in probe_info if x[0] == pair[0] and x[1] == pair[1]][0]
                    for pair in combo] for combo in pair_combos]  # Restore dist associated with combos

    # Create combination of probes
    combo_mol = reduce(Chem.CombineMols, probe_mols)

    # Add bond info to combo_mol
    combo_probes = []
    for pair_combo in pair_combos:
        combo_probe = deepcopy(combo_mol)
        combo_probe.SetProp('_info', f"{pair_combo}")
        combo_probes.append(combo_probe)
    return combo_probes


from math import isclose
def prepare_linkers(multiplet_mol, df_linkers, n_linkers=None):
    # Retrieve bond info from multiplet mol object
    probe_info = eval(multiplet_mol.GetProp('_info'))

    # Filter linkers by distance
    linker_mols = []
    for tpl in probe_info:
        dist = tpl[-1]
        # df_aux = df_linkers[df_linkers['LENGTHS'].apply(
        #     lambda x: len(x.keys()) == len(probe_info))]  # Keep linkers with as much dummies as probes
        df_aux = df_linkers[df_linkers['LENGTHS'].apply(lambda x:
            any(a < dist < b and isclose(a, dist, rel_tol=0.15)
            and isclose(b, dist, rel_tol=0.15) for a, b in x.values()))]

        if n_linkers:
            try: df_aux = df_aux.sample(n_linkers)
            except: pass

        tpl_mols = []
        for i in df_aux.index.tolist():
            mol = deepcopy(df_aux.at[i, 'ROMol'])
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == '*':
                    atom.SetProp('_label', f"L{i}_{atom.GetIdx()}")
                else:
                    atom.SetProp('_label', '')

            # Filter pair of dummies satisfying dist requirement
            info = [(mol.GetAtomWithIdx(k[0]).GetProp('_label'),
                     mol.GetAtomWithIdx(k[1]).GetProp('_label'),
                     dist) for k, v in df_aux.at[i, 'LENGTHS'].items()
                     if v[0] < dist < v[1]]
            mol.SetProp('_id', f"L{i}")
            mol.SetProp('_info', f"{info}")
            tpl_mols.append(mol)
        linker_mols.append(tpl_mols)

    return linker_mols


def join_fragments(multip_mol, linker_mols):

    joined_mols, out_smiles = [], []
    for combo_linkers in product(*linker_mols):
        # Correct labels of duplicated linkers
        smi_linkers = [Chem.MolToSmiles(x) for x in combo_linkers]
        dupl_ids = [i for i, x in enumerate(smi_linkers) if smi_linkers.count(x) > 1]
        for id in dupl_ids:
            info = combo_linkers[id].GetProp('_info').replace('_', f"-{id}_")  # This is still a string
            combo_linkers[id].SetProp('_info', info)
            for atom in combo_linkers[id].GetAtoms():
                label = atom.GetProp('_label').replace('_', f"-{id}_")
                atom.SetProp('_label', label)

        # Add linkers to multiplet
        mols = [multip_mol] + list(combo_linkers)
        combo_mol = reduce(Chem.CombineMols, mols)

        # Associate probe pairs with linkers (P1, P2): (L1, L2)
        multip_dummies = [tuple(x[:2]) for x in eval(multip_mol.GetProp('_info'))]
        linker_dummies = [[x[:2] for x in eval(mol.GetProp('_info'))] for mol in combo_linkers]
        combo_dict = {m:l for m, l in zip(multip_dummies, linker_dummies)}

        # Expand probe-linker bond combinations (P1, P2): [(P1, L1) (P2, L2)], [(P1, L2) (P2, L1)]
        for k, v in combo_dict.items():
            combo_pairs = [list(product(*[k, vv])) for vv in v]
            combo_pairs = utils.flatten([[[p[0], p[3]], [p[1], p[2]]] for p in combo_pairs])
            combo_dict[k] = combo_pairs

        # Get combinations of bonds
        combo_dummies = list(utils.flatten(x) for x in product(*combo_dict.values()))       

        # CombineMols changes atom numbers so we need a conversion
        # dictionary to map old to new dummy idx 
        new_dummy_dict = {}
        for atom in combo_mol.GetAtoms():
            if atom.GetSymbol() == '*':
                new_dummy_dict[atom.GetProp('_label')] = atom.GetIdx()

        for dummy_pairs in combo_dummies:
            # Convert old to new probe dummy idx
            new_combo_dummies = [(new_dummy_dict[x[0]], new_dummy_dict[x[1]]) for x in dummy_pairs]
            # print(new_combo_dummies, dummy_pairs)

    #!-- Use this to check probe+linker combos --!#
    #     rdDepictor.Compute2DCoords(combo_mol)
    #     rdDepictor.StraightenDepiction(combo_mol)
    #     joined_mols.append(combo_mol)
    # return joined_mols
    #!-------------------------------------------!#

            # Find neighbors of dummy atoms
            combo_neighs = [tuple(combo_mol.GetAtomWithIdx(x).GetNeighbors()[0].GetIdx()
                                for x in dummy_pair) for dummy_pair in new_combo_dummies]

            rwMol = Chem.RWMol(combo_mol)
            Chem.Kekulize(rwMol, clearAromaticFlags=True)

            # Merge fragments
            for neigh_pair in combo_neighs:
                # #TODO: Change chirality of neighs before adding linkers.
                # #TODO: If not inverted, linkers end up in the opposite configuration
                #!-- Use this to check bond errors --!#
                # atnum = rwMol.GetAtomWithIdx(neigh_pair[0]).GetAtomicNum()
                # rwMol.GetAtomWithIdx(neigh_pair[0]).SetAtomicNum(atnum + 8)
                # atnum = rwMol.GetAtomWithIdx(neigh_pair[1]).GetAtomicNum()
                # rwMol.GetAtomWithIdx(neigh_pair[1]).SetAtomicNum(atnum + 8)
                #!-----------------------------------!#

                rwMol.AddBond(neigh_pair[0], neigh_pair[1], order=Chem.rdchem.BondType.SINGLE)  #! not always single!!
                # rwMol.GetAtomWithIdx(neigh_pair[0]).SetNumExplicitHs(0)
                # rwMol.GetAtomWithIdx(neigh_pair[1]).SetNumExplicitHs(0)

                #!-- Comment this to check bond errors --!#
                # Xe_list = [a.GetIdx() for a in rwMol.GetAtoms() if a.GetSymbol() == 'Xe']
                # for dummy in sorted(dummy_list, reverse=True):
                #     rwMol.RemoveAtom(dummy)
                #!-----------------------------------!#

            dummy_list = [a.GetIdx() for a in rwMol.GetAtoms() if a.GetSymbol() == '*']
            for dummy in sorted(dummy_list, reverse=True):
                rwMol.RemoveAtom(dummy)
            
            rwMol = Chem.RemoveHs(rwMol)
            Chem.SanitizeMol(rwMol)
            Chem.rdmolops.SanitizeFlags.SANITIZE_NONE
            smi = Chem.MolToSmiles(Chem.Mol(rwMol))
            smi = smi.replace('[C]', 'C').replace('[CH]', 'C') #<-- Temporarily fixes radical carbons
            out_smiles.append(smi) if smi not in out_smiles else out_smiles
    return [Chem.MolFromSmiles(x) for x in out_smiles]

def create_bonds(mol, dummy_pairs):
    rwMol = Chem.RWMol(deepcopy(mol))
    Chem.Kekulize(rwMol, clearAromaticFlags=True)
    
    # Neutralise active dummies by turning them into Xe or H
    for atom in rwMol.GetAtoms():
        if atom.GetSymbol() == '*':
            rwMol.GetAtomWithIdx(atom.GetIdx()).SetAtomicNum(54)  #<-- Use this to check bond errors

    for dummy_pair in dummy_pairs:
        neigh_dummies = []
        for dummy_idx in dummy_pair:
            dummy_atom = rwMol.GetAtomWithIdx(dummy_idx)
            for neighbor in dummy_atom.GetNeighbors():
                if neighbor.GetAtomicNum() != 1:  # Non-hydrogen neighbors
                    neigh_dummies.append(neighbor.GetIdx())

        # Merge fragments
        # #TODO: Change chirality of neighs before adding linkers.
        # #TODO: If not inverted, linkers end up in the opposite configuration
        #!-- Use this to check bond errors --!#
        # atnum = rwMol.GetAtomWithIdx(neigh_dummies[0]).GetAtomicNum()
        # rwMol.GetAtomWithIdx(neigh_dummies[0]).SetAtomicNum(atnum + 8)
        # atnum = rwMol.GetAtomWithIdx(neigh_dummies[1]).GetAtomicNum()
        # rwMol.GetAtomWithIdx(neigh_dummies[1]).SetAtomicNum(atnum + 8)
        #!-----------------------------------!#
        rwMol.AddBond(neigh_dummies[0], neigh_dummies[1], order=Chem.rdchem.BondType.SINGLE)  #! not always single!!
        rwMol.GetAtomWithIdx(neigh_dummies[0]).SetNumExplicitHs(0)
        rwMol.GetAtomWithIdx(neigh_dummies[1]).SetNumExplicitHs(0)

    # #!-- Comment this to check bond errors --!#
    Xe_list = [a.GetIdx() for a in rwMol.GetAtoms() if a.GetSymbol() == 'Xe']
    for dummy in sorted(Xe_list, reverse=True):
        for b in rwMol.GetAtomWithIdx(dummy).GetBonds():
            b.SetBondType(Chem.rdchem.BondType.SINGLE)
        rwMol.RemoveAtom(dummy)
    # #!-----------------------------------!#

    rwMol = Chem.RemoveHs(rwMol)
    Chem.SanitizeMol(rwMol)
    Chem.rdmolops.SanitizeFlags.SANITIZE_NONE
    out_smiles = Chem.MolToSmiles(Chem.Mol(rwMol), canonical=False)
    out_smiles = out_smiles.replace('[C]', 'C').replace('[CH]', 'C') #<-- Temporarily fixes radical carbons
    if len(out_smiles.split('.')) == 1:
        return Chem.MolFromSmiles(out_smiles)
