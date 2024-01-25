import os, sys, glob, subprocess, math
from io import StringIO
import numpy as np
import pandas as pd
import pickle
from natsort import natsorted, natsort_keygen
import requests
import contextlib
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from meeko import MoleculePreparation
import prody
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from pdbfixer import PDBFixer
from openmm.app import PDBFile
from openbabel import openbabel
from plip.structure.preparation import PDBComplex
from plip.exchange.report import BindingSiteReport
import seaborn as sns

import warnings
warnings.simplefilter('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Custom methods
sys.path.append(os.path.expanduser('~/MEGA/Algoritmi/drug_design'))
from utils import *

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

def export_mols(mols, extension):
    # Creating new directory if does not exist
    from pathlib import Path
    folder = Path(extension.upper() + 's').mkdir(parents=True, exist_ok=True)
    
    if extension == 'pdbqt':
        for mol in mols:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            MoleculePreparation().prepare(mol)
            mol_pdbqt = MoleculePreparation().write_pdbqt_file('PDBQTs/' + mol.GetProp('_Name') + '.pdbqt')
    elif extension == 'sdf':
        for mol in mols:
            w = Chem.SDWriter('SDFs/' + mol.GetProp('_Name') + '.sdf')
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            AllChem.UFFOptimizeMolecule(mol)
            w.write(mol)
            w.close()
    else:
        for mol in mols:
            subprocess.run(['babel', lig, '-O', molfile, '-h'],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
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

def _get_dict_sites(pdb, pymol=False):
    complex = PDBComplex()
    complex.load_pdb(pdb)
    # Find ligands and analyze interactions
    for ligand in complex.ligands:
        complex.characterize_complex(ligand)

    dict_sites = {}
    int_types = ['hydrophobic', 'hbond', 'waterbridge', 'saltbridge',
                 'pistacking', 'pication', 'halogen', 'metal']
    # Loop over binding sites
    for key, value in sorted(complex.interaction_sets.items()):
        if key.split(':')[0] not in modified_AAs:
            binding_site = BindingSiteReport(value)  # Collect data about interactions
            # Create interaction dictionary
            interactions = {x: [getattr(binding_site, x + '_features')] +
                            getattr(binding_site, x + '_info') for x in int_types}
            # Check covalent bonds
            # <The pdb file must contain LINK in the header>
            # <In a dict entry, the residue is the first species and the ligand is the second>
            interactions['covalent'] = [('RESNR', 'RESTYPE', 'RESCHAIN', 'RESNR_LIG', 'RESTYPE_LIG', 'RESCHAIN_LIG')]
            if complex.covalent != []:
                covlinkage = complex.covalent[0] #@Change [0]
                if covlinkage.id1 in [*standard_AAs] and covlinkage.id2 not in nonLOI_list:
                    interactions['covalent'].append((covlinkage.pos1, covlinkage.id1, covlinkage.chain1,
                                                        covlinkage.pos2, covlinkage.id2, covlinkage.chain2))
                elif covlinkage.id2 in [*standard_AAs] and covlinkage.id1 not in nonLOI_list:
                    interactions['covalent'].append((covlinkage.pos2, covlinkage.id2, covlinkage.chain2,
                                                        covlinkage.pos1, covlinkage.id1, covlinkage.chain1))
            dict_sites[key] = interactions

    if pymol is True:
        # Temporarily silence command-line
        with contextlib.redirect_stdout(None):
            from plip.basic import config
            from plip.basic.remote import VisualizerData
            from plip.visualization.visualize import visualize_in_pymol
            config.PYMOL = True  # Tell plip to save the pse file
            pym_complexes = [VisualizerData(complex, site) for site in sorted(complex.interaction_sets)
                                if not len(complex.interaction_sets[site].interacting_res) == 0]
            for pym in pym_complexes: visualize_in_pymol(pym)

    return dict_sites
    
def _get_df_site(dict_sites):
    int_types = ['hydrophobic', 'hbond', 'waterbridge', 'saltbridge',
                 'pistacking', 'pication', 'halogen', 'metal', 'covalent']
    df_site = pd.DataFrame()
    for it in int_types:
        a = dict_sites[it]
        da = pd.DataFrame(a[1:], columns=a[0])
        da['INT_TYPE'] = it
        df_site = pd.concat([df_site, da], axis=0)
    return df_site

def df_interactions(files, ligand=None, outfile='interactions.csv', pymol=False, apo=None):
    pdblist = read_files(files)
    
    df1 = pd.DataFrame()
    for pdb in pdblist:
        pdb_id = pdb.split('/')[-1].split('.')[0].split('_')[0].upper()

        try:
            dict_sites = _get_dict_sites(pdb, pymol=pymol)
            if ligand:
                dict_sites = {k:v for k, v in dict_sites.items() if ligand in k}

            for site in list(dict_sites.keys()):
                df_site = _get_df_site(dict_sites[site])
                df_site.insert(0, 'PDB', pdb_id)
                df1 = pd.concat([df1, df_site], axis=0)
                c = '- COVALENT' if len(dict_sites[site]['covalent']) > 1 else ''
            print(f"*** Found {pdb_id} with binding site(s) {list(dict_sites.keys())} *** {c}")
            # elif len(dict_sites) == 0 and apo in {'remove', 'delete'}:
            #     os.remove(pdb)
            #     print(f"*** {pdb_id} has no ligands. Deleted. ***")
            # elif len(dict_sites) == 0 and apo == 'rename':
            #     os.rename(pdb, pdb.replace('.pdb','_apo'))
            #     print(f"*** {pdb_id} has no ligands. Renamed. ***")
            # else:
            #     print(f"*** No interaction found in {pdb_id} ***")
        except:
            continue

    df1 = df1.reset_index(drop=True)
    df1 = df1.dropna(axis=1, how='all')
    df1 = df1.rename(columns=lambda x: x.replace('RESNR','RESI'))
    df1 = df1.rename(columns=lambda x: x.replace('RESTYPE','RESN'))
    df1['PDB'] = df1['PDB'].str.replace('_COMPLEX','')

    if outfile:
        df1.to_csv(outfile, header=True, index=False, sep=',', line_terminator='\n')
    return df1

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

def filter_int(df2):
    """
    Function to filter a dataframe of ligands based on their interactions.
    We only want:
     1. Residues involved in the binding of many different ligands
     2. Ligands with a minimum of 3 interactions
     3. Ligands with relevant interactions (eg. a minimum of 3 hydrophobic ints., 
        1 hydrophobic int. + 1 H-bond, 1 pi-stackig, etc.)
    """
    
    if not isinstance(df2, pd.DataFrame):
        df2 = pd.read_csv(df2, sep=';', index_col=0)

    # Occupancy indicates the num. of interations of each residue with the set of ligands
    # It is calculated as the sum of non-zero occurrences in a column
    df2.loc['Occupancy'] = df2.astype(bool).sum(axis=0)
    # Filter out residues with occupancy < 5%
    # This means that a residue is rejected if it interacts with a small number of ligands
    df2 = df2.loc[:,df2.loc['Occupancy'] > (len(df2.index)*0.05)]
    df2 = df2.iloc[:-1]                                # Remove 'Occupancy' row
    
    # Int_num indicates the total number of interactions of each ligand
    # It is calculated as the sum of non-zero occurrences in a row
    df3 = df2.assign(Int_num = df2.astype(bool).sum(axis=1))
    # Filter out ligands with less then 3 interactions
    df3 = df3[df3['Int_num'] >= 3]
    df3.drop('Int_num', axis=1, inplace=True)        # Remove 'Int_num' column

    # Int_score measures the 'strength' of the interation of each ligand
    # It is calculated as the row-wise sum of the interaction type numbers
    df3['Int_score'] = df3.sum(axis=1)
    # Filter out ligands with Int_score < 3
    df3 = df3[df3['Int_score'] >= 3]
    df3.drop('Int_score', axis=1, inplace=True)        # Remove 'Int_score' column

    # Split dataframe based on covalent/noncovalent interactions
    dfcov = df3[df3.values == 9]
    dfnoncov = df3[~df3.index.isin(dfcov.index)]

    open('intmap_filtered.txt', 'w').write(df3.to_csv(sep='\t', line_terminator='\n', index=False))
    open('intmap_cov.txt', 'w').write(dfcov.to_csv(sep='\t', line_terminator='\n', index=False))
    open('intmap_noncov.txt', 'w').write(dfnoncov.to_csv(sep='\t', line_terminator='\n', index=False))
    return(df3)

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

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description = 'Calculate radius of the circle')
    parser.add_argument('-m', '--method', type=str, help='')
    parser.add_argument('-f', '--file', help='input file')
    args = parser.parse_args()

    pdblist = read_files(args.file)

    if args.method == 'plip':
        df_interactions(pdblist, outfile='interactions.csv', pymol=True, rename_apo=False)

