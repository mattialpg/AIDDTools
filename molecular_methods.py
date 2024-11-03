"""
This module provides a set of utilities for molecular manipulation and visualization.
"""
import os, sys, math
import numpy as np
import pandas as pd
from urllib import parse

from natsort import natsorted, natsort_keygen
from copy import deepcopy
from itertools import combinations
from functools import reduce

from rdkit import Chem
from rdkit.Chem import Draw, AllChem, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
# from openmm.app import PDBFile
# from pdbfixer import PDBFixer
# from openbabel import openbabel

# import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Suppress warnings
import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import *

# -------------------------------------------- #

from meeko import MoleculePreparation, PDBQTWriterLegacy
def export_mol(mol, outfile, addHs=False, verbose=False):
    mol = deepcopy(mol)
    if 'pdbqt' in outfile:
        mol = Chem.AddHs(mol, addCoords=True)
        AllChem.EmbedMolecule(mol)
        mol_setup = MoleculePreparation().prepare(mol)[0]
        pdbqt_string = PDBQTWriterLegacy.write_string(mol_setup)[0]
        with open(outfile, 'w') as f:
            f.write(pdbqt_string)

        # Fixing problems with G0 atoms (vina cannot handle them)
        if 'G0' in pdbqt_string:
            with open(outfile, 'r') as f:
                lines = f.read().splitlines()
                lines = [x for x in lines if ' G0' not in x]
                lines = [x.replace('CG0', 'C') for x in lines]
            with open(outfile, 'w') as f:
                f.write('\n'.join(lines))

    elif 'sdf' in outfile:
        if addHs:
            mol = Chem.AddHs(mol, addCoords=True)
        # AllChem.EmbedMolecule(mol)
        with open(outfile, 'w') as fw:
            Chem.SDWriter(fw).write(mol)
    
    if verbose:
        print(f"*** Succesfully exported {outfile} ***")
        

# def draw_mols(mols, filename=None, align=False):

#     # from rdkit.Chem import rdCoordGen
#     # mols = []
#     # for smi in df.SMILES:
#         # mol = Chem.MolFromSmiles(smi)
#         # mols.append(mol)
#         # # AllChem.Compute2DCoords(mol)
#         # ##OR##
#         # rdCoordGen.AddCoords(mol)

#     # # Condense functional groups (e.g. -CF3, -AcOH)
#     # abbrevs = rdAbbreviations.GetDefaultAbbreviations()
#     # mol = rdAbbreviations.CondenseMolAbbreviations(mol,abbrevs,maxCoverage=0.8)

#     # Calculate Murcko Scaffold Hashes
#     # regio = [rdMolHash.MolHash(mol,Chem.rdMolHash.HashFunction.Regioisomer).split('.') for mol in mols]
#     # common = list(reduce(lambda i, j: i & j, (set(x) for x in regio)))
#     # long = max(common, key=len)

#     # Murcko scaffold decomposition
#         # scaffold = MurckoScaffold.GetScaffoldForMol(mol)
#         # generic = MurckoScaffold.MakeScaffoldGeneric(MurckoScaffold.GetScaffoldForMol(mol))
#         # plot = Draw.MolsToGridImage([mol, scaffold, generic], legends=['Compound', 'BM scaffold', 'Graph framework'], 
#                 # molsPerRow=3, subImgSize=(600,600)); plot.show()

#     if align is True:
#         # Perform match with a direct substructure m1
#         m1 = Chem.MolFromSmiles('CC1=CC=C(C=C1)C(C)C')
#         sub1 = [mol for mol in mols if mol.HasSubstructMatch(m1)]
#         sub2 = [mol for mol in mols if mol not in sub1]
#         # print(Chem.MolToMolBlock(m1)) # Print coordinates
#         AllChem.Compute2DCoords(m1)
        
#         # # Find Maximum Common Substructure m2
#         # mcs1 = rdFMCS.FindMCS(sub2 + [m1])
#         # m2 = Chem.MolFromSmarts(mcs1.smartsString)
#         # AllChem.Compute2DCoords(m2)
#         # # plot = Draw.MolToImage(m2); plot.show()
#         # OR #
#         # Find generic substructure m2
#         params = AllChem.AdjustQueryParameters()
#         params.makeAtomsGeneric = True
#         params.makeBondsGeneric = True
#         m2 = AllChem.AdjustQueryProperties(Chem.RemoveHs(m1), params)
#         AllChem.Compute2DCoords(m2)
#         # plot = Draw.MolToImage(m2); plot.show()
       
#         # Rotate m1 and m2 by an angle theta
#         theta = math.radians(-90.)
#         transformation_matrix = np.array([
#             [ np.cos(theta), np.sin(theta), 0., 3.],
#             [-np.sin(theta), np.cos(theta), 0., 2.],
#             [            0.,            0., 1., 1.],
#             [            0.,            0., 0., 1.]])
#         AllChem.TransformConformer(m1.GetConformer(), transformation_matrix)
#         AllChem.TransformConformer(m2.GetConformer(), transformation_matrix)

#         plot = Draw.MolsToGridImage([m1, m2], molsPerRow=2, subImgSize=(600,300),
#                                     legends=['Core substructure','Generic substructure']); plot.show()

#         # Align all probe molecules to m1 or m2
#         for s in sub1:
#             AllChem.GenerateDepictionMatching2DStructure(s,m1)
#         for s in sub2:
#             AllChem.GenerateDepictionMatching2DStructure(s,m2)
#         subs = sub1 + sub2
#     else: subs = mols
    
#     img1 = Draw.MolsToGridImage(subs, molsPerRow=3, subImgSize=(600,400),
#                                 legends=[s.GetProp('_Name') for s in subs])    
#     img2 = Draw.MolsToGridImage(subs, molsPerRow=3, subImgSize=(600,400),
#                                 legends=[s.GetProp('_Name') for s in subs], useSVG=True)
#                                 # highlightAtomLists=highlight_mostFreq_murckoHash
                                
#     if filename:
#         img1.save(filename)
#         open(filename.split('.')[0] + '.svg','w').write(img2)
#     else: img1.show()
    
#     # Manually rotate molecules and draw
#     # d = Draw.rdMolDraw2D.MolDraw2DCairo(512, 512)
#     # # d.drawOptions().rotate = -90
#     # d.DrawMolecule(m1)
#     # d.FinishDrawing()
#     # d.WriteDrawingText("0.png")


def mol2svg(mol):
    try:
        Chem.rdmolops.Kekulize(mol)
    except:
        pass
    drawer = rdMolDraw2D.MolDraw2DSVG(600, 600)
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    svg_string = drawer.GetDrawingText()

    # In some cases the smiles must be modified when dealing with rings
    # smi = smi[0].upper() + smi[1:]
    # svg_string = smi2svg(smi)
    
    impath = 'data:image/svg+xml;charset=utf-8,' + parse.quote(svg_string, safe="")

    return impath


import py3Dmol
def drawit(mol, confId=-1):
    """Draw molecule in 3D
    
    Args:
    ----
        mol: rdMol, molecule to show
        size: tuple(int, int), canvas size
        style: str, type of drawing molecule
               style can be 'line', 'stick', 'sphere', 'carton'
        surface, bool, display SAS
        opacity, float, opacity of surface, range 0.0-1.0
    Return:
    ----
        viewer: py3Dmol.view, a class for constructing embedded 3Dmol.js views in ipython notebooks.
    """
    p = py3Dmol.view(width=700, height=450)
    p.removeAllModels()
    p.addModel(Chem.MolToMolBlock(mol, confId=confId),'sdf')
    p.setStyle({'stick':{}})
    p.setBackgroundColor('0xeeeeee')
    p.zoomTo()
    return p.show()

# from ipywidgets import interact, fixed  #<-- this gives `JUPYTER_PLATFORM_DIRS=1` error
# def drawit_slider(mol):    
#     p = py3Dmol.view(width=700, height=450)
#     interact(drawit, mol=fixed(mol), p=fixed(p), confId=(0, mol.GetNumConformers()-1))

def drawit_bundle(mol, confIds=None):
    p = py3Dmol.view(width=700, height=450)
    p.removeAllModels()

    if not confIds: confIds = range(mol.GetNumConformers())
    for confId in confIds:
        p.addModel(Chem.MolToMolBlock(mol, confId=confId), 'sdf')
    p.setStyle({'stick':{}})
    p.setBackgroundColor('0xeeeeee')
    p.zoomTo()
    return p.show()

def show_atom_indices(mols, label='atomNote', prop=None):
    "label: [atomNote, molAtomMapNumber]"
    mol_list = [mols] if not isinstance(mols, list) else mols
    for mol in mol_list:
        for atom in mol.GetAtoms():
            if prop:
                if not atom.HasProp(prop):
                    atom.SetProp(label, '')  # Fill prop with dumb values
                else:
                    atom.SetProp(label, str(atom.GetProp(prop)))
            else:
                atom.SetProp(label, str(atom.GetIdx()))

    if not isinstance(mols, list):
        return mol_list[0]
    else:
        return mol_list


def remove_atom_indices(mols, label):
    for mol in mols:
        for atom in mol.GetAtoms():
            atom.ClearProp(label)
    return mols

#TODO Check!!
from PIL import Image
from io import BytesIO
def show_bond_indices(mol):
    def show_mol(d2d, mol, legend='', highlightAtoms=[]):
        d2d.DrawMolecule(mol,legend=legend, highlightAtoms=highlightAtoms)
        d2d.FinishDrawing()
        bio = BytesIO(d2d.GetDrawingText())
        return Image.open(bio)

    d2d = Draw.MolDraw2DCairo(600,400)
    dopts = d2d.drawOptions()
    dopts.addBondIndices = True
    show_mol(d2d, mol)

def reset_view(mol):
    mol = deepcopy(mol)
    # Reset coordinates for display
    rdDepictor.SetPreferCoordGen(True)
    rdDepictor.Compute2DCoords(mol)
    rdDepictor.StraightenDepiction(mol)
    # Delete substructure highlighting
    del mol.__sssAtoms
    return mol

## Comment miniconda3/envs/my-chem/Lib/site-packages/prody/proteins/pdbfile.py, lines 314-316
## to hide log message when using prody.parsePDB.

# import time
# start_time = time.time()
# print("--- %s seconds ---" % (time.time() - start_time))

def replace_dummies(mol, replace_with='H', keep_info=True):
    mol = deepcopy(mol)
    rwMol = Chem.RWMol(mol)

    if keep_info:
        for atom in mol.GetAtoms():
            atom.SetProp('_symbol', atom.GetSymbol())

    dummies = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == '*']

    # # Replace dummy atoms (this does not retain atom properties!)
    # d = Chem.MolFromSmiles('*'); h = Chem.MolFromSmiles(replace_with)
    # mol = Chem.ReplaceSubstructs(mol, d, h, replaceAll=True)[0]
    # return mol

    # Replace dummy atoms and retain atom properties
    if replace_with == 'L':
        # Get atom type from fragmentation class
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                d = atom.GetProp('dummyLabel')
                rwMol.ReplaceAtom(atom.GetIdx(), Chem.Atom(d))
    elif replace_with == 'H':
        for dummy in sorted(dummies, reverse=True):
            # rwMol.RemoveAtom(dummy)
            rwMol.ReplaceAtom(dummy, Chem.Atom(1))
            # rwMol.GetAtomWithIdx(a).SetNumExplicitHs(0)
            # Chem.AddHs(rwMol)
    else:
        h = Chem.GetPeriodicTable().GetAtomicNumber(replace_with)
        for dummy in sorted(dummies, reverse=True):
            # rwMol.ReplaceAtom(atom.GetIdx(), Chem.Atom(h))
            rwMol.GetAtomWithIdx(dummy).SetAtomicNum(h)
    rwMol = Chem.RemoveHs(rwMol)
    Chem.SanitizeMol(rwMol)
    return Chem.Mol(rwMol)


def restore_dummies(mol):
    rwMol = Chem.RWMol(mol)
    for atom in mol.GetAtoms():
        if atom.GetProp('_symbol') == '*':
            # rwMol.GetAtomWithIdx(atom.GetIdx()).SetAtomicNum(0)
            rwMol.ReplaceAtom(atom.GetIdx(), Chem.Atom(0))
    return rwMol.GetMol()


from scipy.spatial.distance import pdist
def dist_between_dummies(mol, numConfs=100, replace_with='C'):
    try:
        dummies = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == '*']
        if len(dummies) < 2:
            return np.nan

        rwMol = deepcopy(mol)
        for dummy in dummies:
            n_atoms = rwMol.GetNumAtoms()
            neigh = rwMol.GetAtomWithIdx(dummy).GetNeighbors()[0].GetIdx()
            rwMol = Chem.RWMol(reduce(Chem.CombineMols, [rwMol, Chem.MolFromSmiles(replace_with)]))
            Chem.Kekulize(rwMol, clearAromaticFlags=True)
            bond_type = (rwMol.GetBondBetweenAtoms(dummy, neigh).GetBondType())
            rwMol.AddBond(neigh, n_atoms, order=bond_type)  # Creating bond with the first atom of "replace_with"
            rwMol.GetAtomWithIdx(n_atoms).SetProp('old_idx', str(dummy))

        dummies = [a.GetIdx() for a in rwMol.GetAtoms() if a.GetSymbol() == '*']
        for dummy in sorted(dummies, reverse=True):
            rwMol.RemoveAtom(dummy)

        rwMol = Chem.RemoveHs(rwMol)
        Chem.SanitizeMol(rwMol)
        Chem.rdmolops.SanitizeFlags.SANITIZE_NONE
        outmol = Chem.Mol(rwMol)

        # smi = Chem.MolToSmiles(Chem.Mol(rwMol))
        # smi = smi.replace('[CH]', 'C') #<-- Temporarily fixes radical carbons
        # outmol = Chem.MolFromSmiles(smi)
        # display(show_atom_indices(outmol))
        
        new_dummies = [a.GetIdx() for a in outmol.GetAtoms() if a.HasProp('old_idx')]

        # Generate conformers
        confids = AllChem.EmbedMultipleConfs(outmol, numConfs,
                                            useRandomCoords=True,
                                            randomSeed=2020,
                                            maxAttempts=100,
                                            numThreads=0)

        # Calculate distance range between dummy atoms
        dist_dict = {}
        for dummy_1, dummy_2 in combinations(new_dummies, 2):
            dists = [AllChem.GetBondLength(conf, dummy_1, dummy_2) for conf in outmol.GetConformers()]
            dist_dict[(int(outmol.GetAtomWithIdx(dummy_1).GetProp('old_idx')),
                    int(outmol.GetAtomWithIdx(dummy_2).GetProp('old_idx')))]\
                    = (round(min(dists), 3), round(max(dists), 3))
        return dist_dict
    
    except Exception as exc:
        print(exc)
        return {}


# def intmap(csv_int, pivot='RESN'):
#     df = pd.read_csv(csv_int, sep=';').loc[:, ['PDB', 'RESN', 'RESTYPE', 'INT_TYPE']]
#     if pivot == 'RESN':
#         res_dict = dict(zip(df['RESN'], df.replace({'RESTYPE': standard_AAs})['RESTYPE']\
#                                 + df['RESN'].astype(str)))
    
#     dict_int = {'hydrophobic':'1', 'hbond':'2', 'pistacking':'3', 'waterbridge':'4',
#                 'saltbridge':'5', 'pication':'6', 'halogen':'7', 'metal':'8', 'covalent':'9'}
#     df = df.replace({'INT_TYPE': dict_int})
#     df2 = df.pivot_table(index='PDB', columns=pivot, values='INT_TYPE', aggfunc='max', fill_value=np.nan)
#     df2 = df2.dropna(axis=1, how='all').fillna(0)        # Drop NaN columns and replace NaN with 0
#     df2 = df2.sort_values(by = 'PDB', key=natsort_keygen())
#     df2.reset_index(inplace=True)                        # Reset 'PDB' as a column, not index
#     if pivot == 'RESN':
#         df2 = df2.rename(res_dict, axis=1)                # Use compact aa notation for columns
#     df2.columns.name = None                                # Remove spurious 'RESN' label
#     open('intmap_complete.txt', 'w').write(df2.to_csv(sep='\t', line_terminator='\n', index=False))
#     return(df2)

# def heatmap(df3, ref=None, square=False, savename='heatmap.png'):
#     """ Plot heatmap """
#     if not isinstance(df3, pd.DataFrame): df3 = pd.read_csv(df3, sep='\t')
#     df3.replace(0, np.nan, inplace=True)

#     myColors = ('#ff9b37', '#c8c8ff', '#78c878', '#ff8c8c',
#                 '#8c8cff', '#82aab9', '#f5af91', '#ffd291', '#bfbfbf')
                
#     NGL_colors = ([0.90, 0.10, 0.29], [0.26, 0.83, 0.96], [1.00, 0.88, 0.10], [0.67, 1.00, 0.76],
#     [0.75, 0.94, 0.27], [0.27, 0.60, 0.56], [0.94, 0.20, 0.90], [0.90, 0.75, 1.00], [0.92, 0.93, 0.96])

#     n = len(NGL_colors)
#     cmap = LinearSegmentedColormap.from_list('Custom', myColors, n)

#     if ref:
#         df4 = pd.read_csv(ref, sep='\t')
#         df3 = pd.concat([df4,df3], axis=0).fillna(0)
#         df3 = df3.reset_index(drop=True)
#         df3 = df3.reindex(natsorted(df3.columns), axis=1)
#         col = df3.pop('PDB'); df3.insert(0,col.name,col) # Shift 'PDB' column back to first position

#     grid_kws = {'height_ratios': [5], 'width_ratios': [30,1], 'wspace': 0.1}
#     if len(df3.index) > 250:
#         df_split = np.array_split(df3, len(df3.index)//250)
#     else: df_split = [df3]
    
#     for idx,subdf in enumerate(df_split):
#         fig, (ax, axcb) = plt.subplots(1, 2, figsize=(12,9), gridspec_kw=grid_kws)
#         #< Need to use axcb for the colorbar in order to lock its size to that of the map >#
#         g = sns.heatmap(subdf.iloc[:,1:], ax=ax, cbar_ax=axcb,
#                         cmap=cmap, vmin=0, vmax=10,
#                         linecolor='white', linewidths=0.5,
#                         xticklabels=list(subdf.columns.values)[1:],
#                         yticklabels=list(subdf['PDB']),
#                         # square=square, cbar_kws={"shrink":0.6}
#                         )
#         g.set_facecolor('#fafbd8')
        
#         # Ticks and labels
#         ax.set_xlabel('RESN'); ax.set_ylabel('CCI')
#         ax.tick_params(axis='both')#, labelsize=7)
#         # if ref is not None: ax.hlines([len(df4.index)], *ax.get_xlim(), color='black', lw=0.4)
#         if len(subdf.index) > 55: ax.set_yticks([])

#         # Heatmap frame
#         ax.axhline(y=0, color='k',linewidth=0.8)
#         ax.axhline(y=subdf.iloc[:,1:].shape[0], color='k',linewidth=0.8)
#         ax.axvline(x=0, color='k',linewidth=0.8)
#         ax.axvline(x=subdf.iloc[:,1:].shape[1], color='k',linewidth=0.8)

#         # Colorbar settings
#         r = axcb.get_ylim()[1] - axcb.get_ylim()[0]
#         axcb.yaxis.set_ticks([axcb.get_ylim()[0] + 0.5*r/n + r*i/n for i in range(n)]) # Evenly distribute ticks
#         axcb.set_yticklabels(['Hydrophobic', 'H-bond', r'$\pi$-stacking' ,'Water bridge',
#                             'Salt bridge', r'$\pi$-cation', 'Halogen', 'Metal', 'Covalent'])
#         # for spine in ax.collections[0].colorbar.ax.spines.values():
#             # spine.set_visible(True) # Show the border of the colorbar
                            
#         plt.tight_layout()
#         plt.savefig(savename.replace('.', '_%i.' %(idx+1)), dpi=600)
#         # plt.show()
#     return

# def timemap(df3, savename='timemap.png', title=None):
#     """ Plot heatmap """
#     if not isinstance(df3, pd.DataFrame):
#         df3 = pd.read_csv(df3, sep='\t')

#     # if len(df3.index) <= 100:
#         # df3 = df3.reindex(list(range(0, 101))).reset_index(drop=True).fillna(0)

#     myColors = ('#fafbd8', '#ff9b37', '#c8c8ff', '#78c878', '#ff8c8c',
#                 '#8c8cff', '#82aab9', '#f5af91', '#ffd291', '#bfbfbf')
#     cmap = LinearSegmentedColormap.from_list('Custom', myColors, len(myColors))
    
#     sns.set();
#     ax = sns.heatmap(df3.iloc[:,1:].T, cmap=cmap, vmin=0, vmax=10,
#                     xticklabels=20, 
#                     yticklabels=list(df3.columns.values)[1:])

#     # ax.tick_params(axis='y', labelsize=7)
#     ax.invert_yaxis(); plt.yticks(rotation=0)
#     plt.xlabel('Time (ns)'); plt.ylabel('Res. Num.')
#     ax.tick_params(left=True, bottom=True)

#     ax.axhline(y=0, color='k', linewidth=2)
#     ax.axvline(x=0, color='k', linewidth=2)
#     ax.axhline(y=df3.iloc[:,1:].shape[1], color='k', linewidth=2)
#     ax.axvline(x=df3.iloc[:,1:].shape[0], color='k', linewidth=2)
    
#     colorbar = ax.collections[0].colorbar
#     colorbar.set_ticks(np.linspace(0,10,21)[1::2])
#     colorbar.set_ticklabels(['', 'hydrophobic', 'hbond', 'pistacking' ,'waterbridge',
#                         'saltbridge', 'pication', 'halogen', 'metal', 'covalent'])

#     # if len(df3.index) > 55: plt.yticks([])
#     if title is not None: plt.title(title, weight='bold', y=1.02)#, fontsize = 20)
#     plt.tight_layout()
#     plt.savefig(savename, dpi=600)
#     plt.show()
#     return


def view_interactions(viewer_obj, df_interactions, interaction_list):

    color_map = {
        "hydrophobic": [0.90, 0.10, 0.29],
        "hbond": [0.26, 0.83, 0.96],
        "waterbridge": [1.00, 0.88, 0.10],
        "saltbridge": [0.67, 1.00, 0.76],
        "pistacking": [0.75, 0.94, 0.27],
        "pication": [0.27, 0.60, 0.56],
        "halogen": [0.94, 0.20, 0.90],
        "metal": [0.90, 0.75, 1.00],
    }

    interacting_residues = []
    for interaction_type in interaction_list:
        color = color_map[interaction_type]
        df_filtered = df_interactions[df_interactions['INT_TYPE'] == interaction_type]

        from ast import literal_eval
        df_filtered['LIGCOO'] = df_filtered['LIGCOO'].apply(lambda x: list(map(float, literal_eval(x))))
        df_filtered['PROTCOO'] = df_filtered['PROTCOO'].apply(lambda x: list(map(float, literal_eval(x))))
        
        # Add cylinder between ligand and protein coordinate
        for row in df_filtered.itertuples():
            viewer_obj.shape.add_cylinder(row.LIGCOO, row.PROTCOO,
                                          color, 0.1)
            interacting_residues.append(str(row.RESNR))

    # Display interacting residues
    # res_sele = " or ".join([f"({r} and not _H)" for r in interacting_residues])
    # res_sele_nc = " or ".join([f"({r} and ((_O) or (_N) or (_S)))" for r in interacting_residues])
    # viewer_obj.add_ball_and_stick(sele=res_sele, colorScheme="chainindex", aspectRatio=1.5)
    # viewer_obj.add_ball_and_stick(sele=res_sele_nc, colorScheme="element", aspectRatio=1.5)
    for resn in interacting_residues:
        viewer_obj.add_licorice(resn)
    return viewer_obj

from rdkit.Chem import rdDistGeom
def get_coords(mol):
    mol = deepcopy(mol)
    mol = replace_dummies(mol, 'C')
    mol = Chem.AddHs(mol, addCoords=True)
    rdDistGeom.EmbedMolecule(mol)
    conf = mol.GetConformer()
    return conf.GetPositions()


from itertools import product, combinations
def distance_matrix(mol1, mol2=None, subset=None):
    """
    Get distance between all atoms
    subset: list([i-indices], [j-indices]) or list([indices])
    """
    if not mol2:
        mol2 = mol1
    
    coords1 = np.array([mol1.GetConformer().GetAtomPosition(i) for i in range(mol1.GetNumAtoms())])
    coords2 = np.array([mol2.GetConformer().GetAtomPosition(i) for i in range(mol2.GetNumAtoms())])
    
    dist_matr = np.linalg.norm(coords1[:, np.newaxis] - coords2, axis=2)

    if subset:
        dist_matr = dist_matr[np.ix_(subset[0], subset[1])]
     
    return np.round(dist_matr, 4)


def get_dummy_neighbor_idxs(mol):
    neigh_idxs = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == '*':
            neighs = atom.GetNeighbors()
            if len(neighs) > 1:
                raise KeyError("Dummy atom has more than one neighbor.")
            neigh_idxs.append(neighs[0].GetIdx())
    return neigh_idxs


def canonicalize(obj):
    if isinstance(obj, Chem.Mol):
        return Chem.MolFromSmiles(Chem.MolToSmiles(obj))
    elif isinstance(obj, str):
        return Chem.MolToSmiles(Chem.MolFromSmiles(obj))
    

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


def neutralize_atoms(mol):
    """ https://baoilleach.blogspot.com/2019/12/no-charge-simple-approach-to.html """
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    at_matches = mol.GetSubstructMatches(pattern)
    at_matches_list = [y[0] for y in at_matches]
    if len(at_matches_list) > 0:
        for at_idx in at_matches_list:
            atom = mol.GetAtomWithIdx(at_idx)
            chg = atom.GetFormalCharge()
            hcount = atom.GetTotalNumHs()
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(hcount - chg)
            atom.UpdatePropertyCache()
    del mol.__sssAtoms
    return mol


def MolTo3DView(mol, size=(300, 300), style="stick", surface=False, opacity=0.5):

    assert style in ('line', 'stick', 'sphere', 'carton')
    mblock = Chem.MolToMolBlock(mol)
    viewer = py3Dmol.view(width=size[0], height=size[1])
    viewer.addModel(mblock, 'mol')
    viewer.setStyle({style:{}})
    if surface:
        viewer.addSurface(py3Dmol.SAS, {'opacity': opacity})
    viewer.zoomTo()
    return viewer
    
def SmilesToConf(smiles):
    '''Convert SMILES to rdkit.Mol with 3D coordinates'''
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        return mol
    else:
        return None

from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
def moldrawsvg(mol, molSize=(400,300), kekulize=True):
    mc = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol( mol.ToBinary() )
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0], molSize[1])
    drawer.DrawMolecule( mc )
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    return svg.replace("svg:","")