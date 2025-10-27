"""
This module provides a set of utilities for molecular manipulation and visualization.
"""
import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os, sys
import numpy as np
import pandas as pd
from copy import deepcopy
from functools import reduce

from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw import rdDepictor
rdDepictor.SetPreferCoordGen(True)
import py3Dmol

# from openmm.app import PDBFile
# from pdbfixer import PDBFixer
# from openbabel import openbabel


def draw_mols(mols, filename=None, align=False):

    if align == 'MCS':
        mcs = rdFMCS.FindMCS(mols)
        template = Chem.MolFromSmarts(mcs.smartsString)
        AllChem.Compute2DCoords(template)

        for mol in mols:
            AllChem.GenerateDepictionMatching2DStructure(mol, template)

    if not filename:
        img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(300,200),
                                   returnPNG=False)
        display(img)#.show()

    # from rdkit.Chem import rdCoordGen
    # mols = []
    # for smi in df.SMILES:
    #     mol = Chem.MolFromSmiles(smi)
    #     mols.append(mol)
    #     # AllChem.Compute2DCoords(mol)
    #     ##OR##
    #     rdCoordGen.AddCoords(mol)

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


def drawit(mol, confId=-1, label_dict=None, show_dummy_indices=True):
    """
    Draw a single molecule or combined fragments in 3D using py3Dmol.

    Parameters:
    - mol: RDKit Mol or list of Mols
    - confId: Conformer ID to use
    - label_dict: Optional dict of labels {text: [x, y, z]}
    - show_dummy_indices: If True, label dummy atoms and their neighbours
    """
    if isinstance(mol, list):
        mol = reduce(Chem.CombineMols, mol)

    view = py3Dmol.view(width=700, height=450)
    view.addModel(Chem.MolToMolBlock(mol, confId=confId), 'sdf')
    view.setStyle({'stick': {}})
    view.setBackgroundColor('0xeeeeee')
    view.zoomTo()

    if label_dict:
        for text, pos in label_dict.items():
            view.addLabel(text, {"position": dict(zip("xyz", pos)), "fontSize": 12})

    if show_dummy_indices:
        conf = mol.GetConformer(confId)
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                pos = conf.GetAtomPosition(atom.GetIdx())
                view.addLabel(str(atom.GetIdx()), {
                    "position": dict(x=pos.x, y=pos.y, z=pos.z),
                    "fontSize": 12,
                    "backgroundColor": "grey",
                    "borderColor": "black",
                    "inFront": True
                })
                for neigh in atom.GetNeighbors():
                    npos = conf.GetAtomPosition(neigh.GetIdx())
                    view.addLabel(str(neigh.GetIdx()), {
                        "position": dict(x=npos.x, y=npos.y, z=npos.z),
                        "fontSize": 12,
                        "backgroundColor": "grey",
                        "borderColor": "black",
                        "inFront": True
                    })

    return view.show()


def draw_exploded(mols, confId=-1, scale=2.0, show_dummy_indices=True):
    """
    Draw multiple fragments separated radially (exploded view).

    Parameters:
    - mols: list of RDKit Mols (fragments)
    - confId: Conformer ID to use
    - scale: Explosion distance
    - show_dummy_indices: If True, label dummy atoms and their neighbours
    """
    def centroid(mol, confId=0):
        """Return centroid of molecule as a NumPy array."""
        conf = mol.GetConformer(confId)
        coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
        return coords.mean(axis=0)


    def translate(mol, offset, confId=0):
        """Return a copy of mol translated by offset (x, y, z)."""
        mol = Chem.Mol(mol)
        conf = mol.GetConformer(confId)
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            conf.SetAtomPosition(i, (pos.x + offset[0], pos.y + offset[1], pos.z + offset[2]))
        return mol

    combined = reduce(Chem.CombineMols, mols)
    global_centroid = centroid(combined, confId)

    exploded_frags = []
    for frag in mols:
        vec = centroid(frag, confId) - global_centroid
        if np.linalg.norm(vec) > 1e-6:
            vec = vec / np.linalg.norm(vec) * scale
        exploded_frags.append(translate(frag, vec, confId))

    exploded_mol = reduce(Chem.CombineMols, exploded_frags)
    return drawit(exploded_mol, confId=confId, show_dummy_indices=show_dummy_indices)

# from ipywidgets import interact, fixed  #<-- this gives `JUPYTER_PLATFORM_DIRS=1` error
# def drawit_slider(mol):    
#     view = py3Dmol.view(width=700, height=450)
#     interact(drawit, mol=fixed(mol), view=fixed(view), confId=(0, mol.GetNumConformers()-1))

def drawit_bundle(mol, confIds=None):
    view = py3Dmol.view(width=700, height=450)
    view.removeAllModels()

    if not confIds:
        confIds = range(mol.GetNumConformers())
        
    for confId in confIds:
        view.addModel(Chem.MolToMolBlock(mol, confId=confId), 'sdf')
    view.setStyle({'stick':{}})
    view.setBackgroundColor('0xeeeeee')
    view.zoomTo()
    return view.show()

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
    # del mol.__sssAtoms
    return mol