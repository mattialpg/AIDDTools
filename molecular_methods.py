import warnings
warnings.simplefilter('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os, sys, glob, subprocess, math, shutil
import numpy as np
import pandas as pd
from natsort import natsorted, natsort_keygen
from itertools import combinations
from copy import deepcopy

# import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from rdkit import Chem
from rdkit.Chem import Draw, AllChem, rdDepictor
# from pdbfixer import PDBFixer
from openmm.app import PDBFile
# from openbabel import openbabel

from tools import general_methods as GenMtd
from tools import utils
from tools import GraphRecomp


## Comment miniconda3/envs/my-chem/Lib/site-packages/prody/proteins/pdbfile.py, lines 314-316
## to hide log message when using prody.parsePDB.

# import time
# start_time = time.time()
# print("--- %s seconds ---" % (time.time() - start_time))

def prepare_rec(pdblist, ext='pdb'):
    pdblist = [pdblist] if isinstance(pdblist, str) else pdblist
    # pdblist = glob.glob1(os.getcwd(), '*receptor.pdb')
    
    # Fix PDB
    for pdb in pdblist:
        fixer = PDBFixer(pdb)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(True)
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.4)
        PDBFile.writeFile(fixer.topology, fixer.positions, open(pdb.replace('.pdb', '_fix.pdb'), 'w'))

    # Convert to pdbqt
    if ext == 'pdbqt':
        fixlist = glob.glob1(os.getcwd(), '*fix.pdb')
        for fix in fixlist:
            obconversion = openbabel.OBConversion()
            obconversion.SetInAndOutFormats('pdb', 'pdbqt')
            mol = openbabel.OBMol()
            fixfile = fix.replace('pdb', 'pdbqt')
            obconversion.ReadFile(mol, fix)
            obconversion.WriteFile(mol, fixfile)

            xfile = fix + 'qt'
            with open(xfile, 'r') as f:
                lines = f.readlines()
            with open(xfile.replace('_fix', ''), 'w') as f:
                for line in lines:
                    if 'ATOM' in line:
                        f.write(line)
            # os.remove(fix, fixfile)
    return


from scipy.spatial.distance import pdist
from rdkit.Chem import rdDistGeom
def dist_between_dummies(mol, numConfs=100, replace_with=None):
    try:
        mol_copy = deepcopy(mol)
        dummies = [a.GetIdx() for a in mol_copy.GetAtoms() if a.GetSymbol() == '*']
        a, b = dummies[0], dummies[1]

        if not replace_with:
            dummies = [a.GetIdx() for a in mol_copy.GetAtoms() if a.GetSymbol() == '*']
            mol_copy = GenMtd.replace_dummies(mol_copy, 'C')
            mol_copy = Chem.AddHs(mol_copy, addCoords=True)
        else:
            mol_copy = GraphRecomp.join_fragments([mol_copy], [Chem.MolFromSmiles(replace_with)])
            dummies = [a.GetIdx() for a in mol_copy.GetAtoms()\
                       if a.GetProp('_newbond') == 'True' and a.GetProp('_label') == 'L']
    
        if len(dummies) < 2:
            print('Number of dummy atoms less than 2')
            raise Exception

        # Generate conformers
        confids = AllChem.EmbedMultipleConfs(mol_copy, numConfs,
                                             useRandomCoords=True,
                                             randomSeed=0xf00d,
                                             maxAttempts=100,
                                             numThreads=0)

        # # Keep conformers with energy within 1 kcal from the most stable conformation
        # energies = [tpl[1] for tpl in AllChem.MMFFOptimizeMoleculeConfs(mol_copy, numThreads=0, mmffVariant='MMFF94s')]
        # energies = [(id, x) for id, x in enumerate(energies) if abs(x-min(energies)) >= 1]
        # confids = [tpl[0] for tpl in energies]

        # # Save conformers in an SDF file
        # AllChem.AlignMolConformers(mol_copy)
        # w = Chem.SDWriter('C:/Users/Idener/MEGA/DEVSHEALTH/Q1_FragLIB/confs.sdf')
        # for confid in range(mol_copy.GetNumConformers()):
        #     w.write(mol_copy, confId=confid)
        # w.close()

        # Formal but slower calculation
        # dist = [rdMolTransforms.GetBondLength(conf, dummies[0], dummies[1])
        #          for conf in mol.GetConformers()]
    
        # Faster alternative
        dist_dict = {}
        for dummy_pair in combinations(dummies, 2):
            dists = []
            for confid in confids:
                a, b = dummy_pair[0], dummy_pair[1]
                coord_0 = np.array(mol_copy.GetConformer(confid).GetAtomPosition(a))
                coord_1 = np.array(mol_copy.GetConformer(confid).GetAtomPosition(b))
                dists.append(pdist(np.vstack((coord_0, coord_1))).round(3))
                dist_dict[(a, b)] = (float(min(dists)), float(max(dists)))
        
        # Create a dictionary to map old to new dummy atom numbers
        if replace_with:
            dummy_dict = {}
            for dummy in dummies:
                neigh = mol_copy.GetAtomWithIdx(dummy).GetNeighbors()
                neigh = [n for n in neigh if n.GetProp('_label') == 'P']
                new_dummy = tuple([int(x) for x in neigh[0].GetProp('_neigh').split('_')])
                if new_dummy not in dummy_dict:
                    dummy_dict[new_dummy] = []
                dummy_dict[new_dummy].append(dummy)
            dummy_dict = {tuple(k): v for v, k in dummy_dict.items()}
            replacement_dict = {}
            for keys_tuple, values_tuple in dummy_dict.items():
                for i, key in enumerate(keys_tuple):
                    replacement_dict[key] = values_tuple[i]

            # Get original dummy atom numbers
            final_dict = {}
            for keys_tuple, values_tuple in dist_dict.items():
                replaced_keys = tuple(replacement_dict[key] for key in keys_tuple)
                final_dict[replaced_keys] = values_tuple
            dist_dict = final_dict

        return dist_dict
    except Exception as exc:
        print(exc)
        return {}


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

        #! Alternative: sdf via rdkit > pdbqt via babel > remove sdf (vina complains)
        # mol = Chem.AddHs(mol, addCoords=True)
        # AllChem.EmbedMolecule(mol)
        # fname = outfile.split('.')[0]
        # with open(f"{fname}.sdf", 'w') as fw:
        #     Chem.SDWriter(fw).write(mol)
        # subprocess.run(f"obabel {fname}.sdf -opdbqt -O {fname}.pdbqt", shell=True,
        #                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # os.remove(f"{fname}.sdf")

        # # Fix dummy atom type for vina
        # if '*' in [a.GetSymbol() for a in mol.GetAtoms()]:
        #     with open(f"{fname}.pdbqt", 'r') as f:
        #         lines = f.read().splitlines()
        #         for i in range(len(lines)):
        #             if '*' in lines[i]:
        #                 s = list(lines[i])
        #                 s[77] = 'A'
        #                 lines[i] = ''.join(s)
        #     with open(f"{fname}.pdbqt", 'w') as f:
        #         f.write('\n'.join(lines))

    
    elif 'sdf' in outfile:
        if addHs: mol = Chem.AddHs(mol, addCoords=True)
        AllChem.EmbedMolecule(mol)
        with open(outfile, 'w') as fw:
            Chem.SDWriter(fw).write(mol)
    
    # else:
    #     for mol in mols:
    #         subprocess.run(['babel', lig, '-O', molfile, '-h'],
    #                         stdout=subprocess.PIPE,
    #                         stderr=subprocess.DEVNULL,
    #                         text=True)
    if verbose:
        print(f"*** Succesfully exported {outfile} ***")
    return

def draw_mols(mols, filename=None, align=False):

    # from rdkit.Chem import rdCoordGen
    # mols = []
    # for smi in df.SMILES:
        # mol = Chem.MolFromSmiles(smi)
        # mols.append(mol)
        # # AllChem.Compute2DCoords(mol)
        # ##OR##
        # rdCoordGen.AddCoords(mol)

    # # Condense functional groups (e.g. -CF3, -AcOH)
    # abbrevs = rdAbbreviations.GetDefaultAbbreviations()
    # mol = rdAbbreviations.CondenseMolAbbreviations(mol,abbrevs,maxCoverage=0.8)

    # Calculate Murcko Scaffold Hashes
    # regio = [rdMolHash.MolHash(mol,Chem.rdMolHash.HashFunction.Regioisomer).split('.') for mol in mols]
    # common = list(reduce(lambda i, j: i & j, (set(x) for x in regio)))
    # long = max(common, key=len)

    # Murcko scaffold decomposition
        # scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        # generic = MurckoScaffold.MakeScaffoldGeneric(MurckoScaffold.GetScaffoldForMol(mol))
        # plot = Draw.MolsToGridImage([mol, scaffold, generic], legends=['Compound', 'BM scaffold', 'Graph framework'], 
                # molsPerRow=3, subImgSize=(600,600)); plot.show()

    if align is True:
        # Perform match with a direct substructure m1
        m1 = Chem.MolFromSmiles('CC1=CC=C(C=C1)C(C)C')
        sub1 = [mol for mol in mols if mol.HasSubstructMatch(m1)]
        sub2 = [mol for mol in mols if mol not in sub1]
        # print(Chem.MolToMolBlock(m1)) # Print coordinates
        AllChem.Compute2DCoords(m1)
        
        # # Find Maximum Common Substructure m2
        # mcs1 = rdFMCS.FindMCS(sub2 + [m1])
        # m2 = Chem.MolFromSmarts(mcs1.smartsString)
        # AllChem.Compute2DCoords(m2)
        # # plot = Draw.MolToImage(m2); plot.show()
        # OR #
        # Find generic substructure m2
        params = AllChem.AdjustQueryParameters()
        params.makeAtomsGeneric = True
        params.makeBondsGeneric = True
        m2 = AllChem.AdjustQueryProperties(Chem.RemoveHs(m1), params)
        AllChem.Compute2DCoords(m2)
        # plot = Draw.MolToImage(m2); plot.show()
       
        # Rotate m1 and m2 by an angle theta
        theta = math.radians(-90.)
        transformation_matrix = np.array([
            [ np.cos(theta), np.sin(theta), 0., 3.],
            [-np.sin(theta), np.cos(theta), 0., 2.],
            [            0.,            0., 1., 1.],
            [            0.,            0., 0., 1.]])
        AllChem.TransformConformer(m1.GetConformer(), transformation_matrix)
        AllChem.TransformConformer(m2.GetConformer(), transformation_matrix)

        plot = Draw.MolsToGridImage([m1, m2], molsPerRow=2, subImgSize=(600,300),
                                    legends=['Core substructure','Generic substructure']); plot.show()

        # Align all probe molecules to m1 or m2
        for s in sub1:
            AllChem.GenerateDepictionMatching2DStructure(s,m1)
        for s in sub2:
            AllChem.GenerateDepictionMatching2DStructure(s,m2)
        subs = sub1 + sub2
    else: subs = mols
    
    img1 = Draw.MolsToGridImage(subs, molsPerRow=3, subImgSize=(600,400),
                                legends=[s.GetProp('_Name') for s in subs])    
    img2 = Draw.MolsToGridImage(subs, molsPerRow=3, subImgSize=(600,400),
                                legends=[s.GetProp('_Name') for s in subs], useSVG=True)
                                # highlightAtomLists=highlight_mostFreq_murckoHash
                                
    if filename:
        img1.save(filename)
        open(filename.split('.')[0] + '.svg','w').write(img2)
    else: img1.show()
    
    # Manually rotate molecules and draw
    # d = Draw.rdMolDraw2D.MolDraw2DCairo(512, 512)
    # # d.drawOptions().rotate = -90
    # d.DrawMolecule(m1)
    # d.FinishDrawing()
    # d.WriteDrawingText("0.png")

def intmap(csv_int, pivot='RESN'):
    df = pd.read_csv(csv_int, sep=';').loc[:, ['PDB', 'RESN', 'RESTYPE', 'INT_TYPE']]
    if pivot == 'RESN':
        res_dict = dict(zip(df['RESN'], df.replace({'RESTYPE': standard_AAs})['RESTYPE']\
                                + df['RESN'].astype(str)))
    
    dict_int = {'hydrophobic':'1', 'hbond':'2', 'pistacking':'3', 'waterbridge':'4',
                'saltbridge':'5', 'pication':'6', 'halogen':'7', 'metal':'8', 'covalent':'9'}
    df = df.replace({'INT_TYPE': dict_int})
    df2 = df.pivot_table(index='PDB', columns=pivot, values='INT_TYPE', aggfunc='max', fill_value=np.nan)
    df2 = df2.dropna(axis=1, how='all').fillna(0)        # Drop NaN columns and replace NaN with 0
    df2 = df2.sort_values(by = 'PDB', key=natsort_keygen())
    df2.reset_index(inplace=True)                        # Reset 'PDB' as a column, not index
    if pivot == 'RESN':
        df2 = df2.rename(res_dict, axis=1)                # Use compact aa notation for columns
    df2.columns.name = None                                # Remove spurious 'RESN' label
    open('intmap_complete.txt', 'w').write(df2.to_csv(sep='\t', line_terminator='\n', index=False))
    return(df2)

def heatmap(df3, ref=None, square=False, savename='heatmap.png'):
    """ Plot heatmap """
    if not isinstance(df3, pd.DataFrame): df3 = pd.read_csv(df3, sep='\t')
    df3.replace(0, np.nan, inplace=True)

    myColors = ('#ff9b37', '#c8c8ff', '#78c878', '#ff8c8c',
                '#8c8cff', '#82aab9', '#f5af91', '#ffd291', '#bfbfbf')
                
    NGL_colors = ([0.90, 0.10, 0.29], [0.26, 0.83, 0.96], [1.00, 0.88, 0.10], [0.67, 1.00, 0.76],
    [0.75, 0.94, 0.27], [0.27, 0.60, 0.56], [0.94, 0.20, 0.90], [0.90, 0.75, 1.00], [0.92, 0.93, 0.96])

    n = len(NGL_colors)
    cmap = LinearSegmentedColormap.from_list('Custom', myColors, n)

    if ref:
        df4 = pd.read_csv(ref, sep='\t')
        df3 = pd.concat([df4,df3], axis=0).fillna(0)
        df3 = df3.reset_index(drop=True)
        df3 = df3.reindex(natsorted(df3.columns), axis=1)
        col = df3.pop('PDB'); df3.insert(0,col.name,col) # Shift 'PDB' column back to first position

    grid_kws = {'height_ratios': [5], 'width_ratios': [30,1], 'wspace': 0.1}
    if len(df3.index) > 250:
        df_split = np.array_split(df3, len(df3.index)//250)
    else: df_split = [df3]
    
    for idx,subdf in enumerate(df_split):
        fig, (ax, axcb) = plt.subplots(1, 2, figsize=(12,9), gridspec_kw=grid_kws)
        #< Need to use axcb for the colorbar in order to lock its size to that of the map >#
        g = sns.heatmap(subdf.iloc[:,1:], ax=ax, cbar_ax=axcb,
                        cmap=cmap, vmin=0, vmax=10,
                        linecolor='white', linewidths=0.5,
                        xticklabels=list(subdf.columns.values)[1:],
                        yticklabels=list(subdf['PDB']),
                        # square=square, cbar_kws={"shrink":0.6}
                        )
        g.set_facecolor('#fafbd8')
        
        # Ticks and labels
        ax.set_xlabel('RESN'); ax.set_ylabel('CCI')
        ax.tick_params(axis='both')#, labelsize=7)
        # if ref is not None: ax.hlines([len(df4.index)], *ax.get_xlim(), color='black', lw=0.4)
        if len(subdf.index) > 55: ax.set_yticks([])

        # Heatmap frame
        ax.axhline(y=0, color='k',linewidth=0.8)
        ax.axhline(y=subdf.iloc[:,1:].shape[0], color='k',linewidth=0.8)
        ax.axvline(x=0, color='k',linewidth=0.8)
        ax.axvline(x=subdf.iloc[:,1:].shape[1], color='k',linewidth=0.8)

        # Colorbar settings
        r = axcb.get_ylim()[1] - axcb.get_ylim()[0]
        axcb.yaxis.set_ticks([axcb.get_ylim()[0] + 0.5*r/n + r*i/n for i in range(n)]) # Evenly distribute ticks
        axcb.set_yticklabels(['Hydrophobic', 'H-bond', r'$\pi$-stacking' ,'Water bridge',
                            'Salt bridge', r'$\pi$-cation', 'Halogen', 'Metal', 'Covalent'])
        # for spine in ax.collections[0].colorbar.ax.spines.values():
            # spine.set_visible(True) # Show the border of the colorbar
                            
        plt.tight_layout()
        plt.savefig(savename.replace('.', '_%i.' %(idx+1)), dpi=600)
        # plt.show()
    return

def timemap(df3, savename='timemap.png', title=None):
    """ Plot heatmap """
    if not isinstance(df3, pd.DataFrame):
        df3 = pd.read_csv(df3, sep='\t')

    # if len(df3.index) <= 100:
        # df3 = df3.reindex(list(range(0, 101))).reset_index(drop=True).fillna(0)

    myColors = ('#fafbd8', '#ff9b37', '#c8c8ff', '#78c878', '#ff8c8c',
                '#8c8cff', '#82aab9', '#f5af91', '#ffd291', '#bfbfbf')
    cmap = LinearSegmentedColormap.from_list('Custom', myColors, len(myColors))
    
    sns.set();
    ax = sns.heatmap(df3.iloc[:,1:].T, cmap=cmap, vmin=0, vmax=10,
                    xticklabels=20, 
                    yticklabels=list(df3.columns.values)[1:])

    # ax.tick_params(axis='y', labelsize=7)
    ax.invert_yaxis(); plt.yticks(rotation=0)
    plt.xlabel('Time (ns)'); plt.ylabel('Res. Num.')
    ax.tick_params(left=True, bottom=True)

    ax.axhline(y=0, color='k', linewidth=2)
    ax.axvline(x=0, color='k', linewidth=2)
    ax.axhline(y=df3.iloc[:,1:].shape[1], color='k', linewidth=2)
    ax.axvline(x=df3.iloc[:,1:].shape[0], color='k', linewidth=2)
    
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks(np.linspace(0,10,21)[1::2])
    colorbar.set_ticklabels(['', 'hydrophobic', 'hbond', 'pistacking' ,'waterbridge',
                        'saltbridge', 'pication', 'halogen', 'metal', 'covalent'])

    # if len(df3.index) > 55: plt.yticks([])
    if title is not None: plt.title(title, weight='bold', y=1.02)#, fontsize = 20)
    plt.tight_layout()
    plt.savefig(savename, dpi=600)
    plt.show()
    return


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
    mol_copy = deepcopy(mol)
    mol_copy = GenMtd.replace_dummies(mol_copy, 'C')
    mol_copy = Chem.AddHs(mol_copy, addCoords=True)
    rdDistGeom.EmbedMolecule(mol_copy)
    conf = mol_copy.GetConformer()
    return conf.GetPositions()

import subprocess
def dock_fragments(script, lig_file, rec_file, out_file, center, edges,
                   poses=6, exh=10, log=False, verbose=True):
    try:
        command = f"{script} --ligand {lig_file} --receptor {rec_file} --out {out_file}\
                    --center_x {center[0]} --center_y {center[1]} --center_z {center[2]}\
                    --size_x {edges[0]} --size_y {edges[1]} --size_z {edges[2]}\
                    --num_modes {poses} --exhaustiveness {exh}"
        if log: command += f" --log {out_file.replace('pdbqt', 'log')}"

        output_text = subprocess.check_output(command, universal_newlines=True, shell=True)
        if verbose: print(f"*** Succesfully docked {out_file} ***")
        return output_text
    except Exception as exc:
        if verbose:
            print(f"*** Cannot dock {out_file} ***")
            print(exc)
    

def dihedral_filter(N1, D1, N2, D2):
    """
    Using the chemical sign convention
    """
    n1 = np.cross((D1 - N1), (N2 - N1))
    n2 = np.cross((N2 - N1), (D2 - N1))
    n1 /= np.linalg.norm(n1)
    n2 /= np.linalg.norm(n2)

    m1 = np.cross(n1, (N2 - N1)/np.linalg.norm(N2 - N1))
    x = np.dot(n1, -n2)
    y = np.dot(m1, n2)
    dihedral = np.degrees(np.arctan2(y, x)).round(2)

    if abs(dihedral) <= 30:
        return dihedral
    return 

def inclination_filter(N1, D1, N2, D2):
    v1 = D1 - N1
    v2 = N2 - N1
    v3 = D2 - N2
    v4 = N1 - N2
    
    # Calculate the dot product and magnitudes
    angle1 = np.degrees(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))).round(2)
    angle2 = np.degrees(np.arccos(np.dot(v3, v4) / (np.linalg.norm(v3) * np.linalg.norm(v4)))).round(2)

    if abs(angle1 - angle2) <= 30:
        if angle1 < 70 or angle2 < 70:
            return [angle1, angle2]
    return

from itertools import product, combinations
def full_distance(coord_set1, coord_set2):
    # Distance between all atoms
    dist_list = []
    for pair in product(coord_set1, coord_set2):
        dist = np.linalg.norm(pair[0] - pair[1]).round(3)
        dist_list.append(dist)
    return dist_list

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
    # Reset coordinates for display
    rdDepictor.Compute2DCoords(mol)
    rdDepictor.StraightenDepiction(mol)
    # Delete substructure highlighting
    # del mol.__sssAtoms