import os, glob , sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MacFrag import *
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem import rdCoordGen
import prody

# Custom methods
sys.path.append(os.path.expanduser('~/MEGA/Algoritmi/drug_design'))
from feature_methods import *
from pubchem_methods import *
from utils import *

# Calculate Mol-obj
# df_lig = df_lig[(df_lig['MW']>300) & (df_lig['MW']<800)]

def get_mol(smiles, names=None):
    smiles = smiles if isinstance(smiles, list) else [smiles]
    names = names if isinstance(names, list) else [names]

    # if not names:
    #     names = [str(x+1) for x in range(len(smiles))]

    mols = []
    for smile, name in zip(smiles, names):
        try:
            mol = Chem.MolFromSmiles(smile)
            mol.SetProp('_Name', name)
            mol.SetProp('MyProp', smile)
            mols.append(mol)
        except Exception as err:
            print(f"*** Error with ligand {name}: {err} ***\n")
    return mols

def get_frags(mols, replace_dummies=False, drop_duplicates=False, **kwargs):
    mols = mols if isinstance(mols, list) else [mols]
    df_frag = pd.DataFrame()
    frag_mols, frag_smiles = [], []
    frag_ccis, frag_idx = [], []
    lig_smiles = []

    # Fragment molecules
    for mol in mols:
        try:
            frags = MacFrag(mol, asMols=True, **kwargs)
            frags = sorted(frags, key=lambda x: x.GetNumHeavyAtoms(), reverse=True)
            for idx, frag in enumerate(frags):
                if replace_dummies:
                    # Replace dummy atoms 
                    new_frag = Chem.ReplaceSubstructs(
                        frag,
                        Chem.MolFromSmiles('*'),
                        Chem.MolFromSmiles(replace_dummies),
                        replaceAll=True)[0]
                    try:
                        new_frag = Chem.RemoveHs(new_frag, sanitize=True)
                    except:
                        new_frag = Chem.RemoveHs(new_frag, sanitize=False)
                else:
                    new_frag = frag

                # Set frag name as CCI + frag. idx
                new_frag.SetProp('_Name', f"{mol.GetProp('_Name')}_{idx+1:03}")

                # Assign atom numbers
                for atom in new_frag.GetAtoms():
                    atom.SetProp('atomNote', str(atom.GetIdx()+1))

                # Make lists for dataframe
                frag_mols.append(new_frag)
                frag_ccis.append(mol.GetProp('_Name'))
                frag_idx.append(f"{idx+1:03}")
                frag_smiles.append(Chem.MolToSmiles(new_frag))
                lig_smiles.append(mol.GetProp('MyProp'))

        except Exception as err:
            print(f"*** Error while fragmenting ligand {mol.GetProp('_Name')}: {err} ***\n")

    df_frag = pd.DataFrame({'CCI': frag_ccis, 'FRAG_ID': frag_idx,
                            'Isomeric_SMILES_FRAG': frag_smiles,
                            # 'Isomeric_SMILES_LIG': lig_smiles,
                            'mol': frag_mols})
    
    # Count fragment occurrence and set it as a mol property
    ## The next filter is controversial. If duplicates are removed,
    ## we should search every fragment as a substructure of all the ligands.
    ## Keeping labelled duplicates makes this easier because
    ## we only need to superimpose each ligand to its own fragments
    if drop_duplicates:
        df_frag = df_frag.drop_duplicates(subset=['Isomeric_SMILES_FRAG'])
        df_frag['Count'] = df_frag['Isomeric_SMILES_FRAG'].map(df_frag['Isomeric_SMILES_FRAG'].value_counts())
        mols1 = []
        for _, row in df_frag.iterrows():
            mol1 = row['mol']
            mol1.SetProp('Count', str(row['Count']))
            mols1.append(mol1)
        df_frag['mol'] = mols1
    print(f"*** Extracted {len(df_frag)} fragment(s) ***")
    return df_frag

def save_frags(df_frag, output_dir='.', extension='mol'):
    # Save fragments as .mol files
    for i, row in df_frag.iterrows():
        try:
            frag_mol = Chem.AddHs(row['mol'])
            AllChem.EmbedMolecule(frag_mol, randomSeed=0xf00d)
            AllChem.MMFFOptimizeMolecule(frag_mol)
            block = Chem.MolToMolBlock(frag_mol)
            with open(f"{output_dir}/{frag_mol.GetProp('_Name')}.{extension}", 'w') as f:
                f.write(block)
            AllChem.Compute2DCoords(frag_mol)
        except:
            df_frag.drop([i])
    return df_frag

def draw_frags(mols, outfile=None, show_numbers=False, ref_compound=None, legend=None):
    if ref_compound:
        mols = flatten([ref_compound, mols])

    for mol in mols:
        # AllChem.Compute2DCoords(mol)
        # rdCoordGen.AddCoords(mol)
        if show_numbers is False:
            # Remove atom numbers
            for atom in mol.GetAtoms():
                atom.ClearProp('molAtomMapNumber')
                atom.ClearProp('atomNote')

    mol_chunks = chunks(mols, 1000)
    for i, chun in enumerate(mol_chunks):

        if legend is None:
            try: legend = [f"{x.GetProp('_Name')}\n\n({x.GetProp('Count')})" for x in mols]
            except: legend = []

        if not outfile or outfile == 'show':
            Draw.MolsToGridImage(chun, molsPerRow=4, subImgSize=(330,330)).show()
        else:
            img1 = Draw.MolsToGridImage(chun, molsPerRow=4, subImgSize=(330,330),
                                        legends=legend, legendFontSize=20, useSVG=True)
            open(outfile.replace('.svg', f"_{i+1}.svg"),'w').write(img1)
    return

def RDK_to_PDB(lig_mol, frag_mol):
    # Get ligand atom numbers as in PDB
    lig_on_PDB = [int(atom.GetProp('numB')) for atom in lig_mol.GetAtoms()]

    # Get ligand-fragment match using ligand RDKit numbering
    frag_on_lig = list(lig_mol.GetSubstructMatches(frag_mol))
    num = [atom.GetAtomicNum() for atom in frag_mol.GetAtoms()]
    frag_on_lig = [[b for b, n in zip(match, num) if n != 0] for match in frag_on_lig]  # Remove dummy atom matches

    # Convert ligand-fragment match numbering from RDKit to PDB
    frag_on_PDB = [[lig_on_PDB[i] for i in match] for match in frag_on_lig]

    # for atom in lig_mol.GetAtoms():
    #     atom.SetProp("atomNote", str(atom.GetIdx()))
    #     atom.SetProp("atomNote", str(atom.GetProp('molAtomMapNumber')))
    #     atom.ClearProp('molAtomMapNumber')
    # rdCoordGen.AddCoords(lig_mol)
    # Draw.MolToImage(lig_mol, size=(900,900), highlightAtoms=B).show()
    return frag_on_PDB

def get_fragdistr(df_merged, df_int):
    # Merge selected columns for an easier data extraction
    df_int['DONORIDX'] = df_int['DONORIDX'].combine_first(df_int['DONOR_IDX'])
    df_int['ACCEPTORIDX'] = df_int['ACCEPTORIDX'].combine_first(df_int['ACCEPTOR_IDX'])
    # Use cosine law to get rec-lig distance for waterbridge interactions 
    D = np.sqrt(np.square(df_int['DIST_A-W'].astype(float)) + \
                np.square(df_int['DIST_D-W'].astype(float)) - \
                2*df_int['DIST_A-W'].astype(float) * \
                    df_int['DIST_D-W'].astype(float) * \
                    np.cos(df_int['WATER_ANGLE'].astype(float))) # type: ignore
    df_int['DIST'] = df_int['DIST'].combine_first(D)
    df_int['DIST'] = df_int['DIST'].combine_first(df_int['DIST_D-A']) 

    # Extract parameters based on interaction type
    int_numA = []
    for int_idx, row in df_int.iterrows():
        if row['INT_TYPE'] in ('hbond', 'waterbridge'):
            if row['PROTISDON'] is True:
                recatomidx = [row['DONORIDX']]
                ligatomidx = [row['ACCEPTORIDX']]
                dist = row['DIST']
            else:
                recatomidx = [row['ACCEPTORIDX']]
                ligatomidx = [row['DONORIDX']]
                dist = row['DIST']
        elif row['INT_TYPE'] == 'hydrophobic':
            recatomidx = [row['PROTCARBONIDX']]
            ligatomidx = [row['LIGCARBONIDX']]
            dist = row['DIST']
        elif row['INT_TYPE'] in ('pication', 'saltbridge'):
            recatomidx = row['PROT_IDX_LIST'].split(',')
            ligatomidx = row['LIG_IDX_LIST'].split(',')
            dist = row['DIST']
        elif row['INT_TYPE'] == 'pistacking':
            recatomidx = row['PROT_IDX_LIST'].split(',')
            ligatomidx = row['LIG_IDX_LIST'].split(',')
            dist = row['CENTDIST']
        # elif row['INT_TYPE'] == 'halogen':
        #     recatomidx = row['PROT_IDX_LIST'].split(',')
        #     ligatomidx = row['LIG_IDX_LIST'].split(',')
        #     dist = row['CENTDIST']
        elif row['INT_TYPE'] in ('metal', 'covalent'):
            pass
        else: print(f"{row['INT_TYPE']} interaction not defined")
        recatomidx = [int(x) for x in recatomidx]
        ligatomidx_B = [int(x) for x in ligatomidx]

        # try:
        # Get atom type from receptor PDB
        file = f"RCSB_Fragments/{row['PDB']}.pdb"
        complex = prody.parsePDB(file, verbosity='none')
        rec = complex.select('protein')
        recatomtype = []
        for idx in recatomidx:
            sel = rec.select(f"index {int(idx)-2}")
            recatomtype.append(sel.getNames()[0])
        
        # Keep only interacting fragments
        subdf_merged = df_merged.loc[(df_merged['PDB'] == row['PDB']) & \
                                    (df_merged['CCI'] == row['RESN_LIG']) & \
                                    (df_merged['RESI_LIG'] == row['RESI_LIG']) & \
                                    (df_merged['RESCHAIN_LIG'] == row['RESCHAIN_LIG'])]

        # Map ligand(B) to fragment(A) atom numbering
        for idx in ligatomidx_B:
            for _, subrow in subdf_merged.iterrows():
                frag_num_PDB = subrow['frag_num_PDB']
                for match in frag_num_PDB:
                    if idx in match:
                        fragatomidx_A = match.index(idx)

                        int_numA.append([int_idx + 1, row['PDB'], 
                                         row['RESN_LIG'], row['RESCHAIN_LIG'], row['RESI_LIG'], idx,
                                         subrow['FRAG_ID'], fragatomidx_A,
                                         row['RESI'], row['RESN'], recatomtype,
                                         row['INT_TYPE'], dist,
                                         subrow['mol_LIG'], subrow['mol_FRAG']])
        # except (Exception, OSError, IOError) as err:
        #     print(f"*** Error: {err} ***")
    df_int_numA = pd.DataFrame(int_numA, columns=['INT_IDX', 'PDB',  # General info
                                                  'CCI', 'RESCHAIN_LIG', 'RESI_LIG', 'LIGATOMIDX',  # Ligand info
                                                  'FRAG_ID', 'FRAGATOMIDX',  # Fragment info
                                                  'RESI', 'RESN', 'PROTATOMTYPE',  # Amino acid info
                                                  'INT_TYPE', 'DIST',  # Interaction info
                                                  'mol_LIG', 'mol_FRAG'])  # Mol objects
    # df_int_numA = df_int_numA.dropna()
    # df_int_numA = df_int_numA[df_int_numA.astype(str)['FRAGATOMIDX'] != '[]']
    return df_int_numA
