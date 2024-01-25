
import os, glob
import redo, requests_cache
from io import StringIO
import pandas as pd
import pickle
import requests
from bs4 import BeautifulSoup
import prody
from rdkit import Chem
from rdkit.Chem import AllChem
from urllib.request import urlretrieve

import warnings
warnings.simplefilter('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=PendingDeprecationWarning)
    import pypdb

# Custom libraries
from utils import *

# Caching requests will speed up repeated queries to PDB
requests_cache.install_cache('rcsb_pdb', backend='memory')
@redo.retriable(attempts=10, sleeptime=2)
def describe_pdb(pdb_id):
    description = pypdb.describe_pdb(pdb_id)
    if not description:
        print(f"Error while fetching PDB {pdb_id}, retrying...")
        raise ValueError(f"Could not fetch PDB id {pdb_id}")
    return description

def download_pdb(pdb_ids, output_dir='.'):
    pdb_ids = read_files(pdb_ids)

    for pdb_id in pdb_ids:
        try:
            url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            urlretrieve(url, os.path.join(output_dir, pdb_id + '.pdb'))
            print(f"*** Downloading PDB {pdb_id} ***")
            return True
        except Exception as exc:
            print(f"*** Error with PDB {pdb_id}: {exc} ***")

def align_pdb(pdblist, target): #****#
    import pymol2

    if '*' in pdblist: pdblist = glob.glob1(os.getcwd(), pdblist)
    else: pdblist = [pdblist] if isinstance(pdblist, str) else pdblist
        
    with pymol2.PyMOL() as py:
        py.cmd.load(target, 'target')
        for pdb in pdblist:
            py.cmd.load(pdb, 'mobile')
            print('*** Aligning %s to %s ***' %(pdb,target))
            py.cmd.align('mobile','target')
            py.cmd.save('aligned/' + pdb, 'mobile')
    return

def split_multipdb(filename):
    """ Split multi-entry PDB into single files """
    # $ for f in babel 3_out.pdbqt -Opdb 3_.pdb -m
    with open(filename, 'r') as f:
        lines = f.readlines()
    i = 0
    names = []
    for j, line in enumerate(lines):
        if 'HEADER' in line:
            names.append(line.strip().split(' ')[-1].upper().replace('PDB',''))
        if 'END' in line:
            with open(names[-1] + '.pdb', 'w') as f:
                f.write(''.join(lines[i:j+1]))
            i = j+1
    return(names)
    return(names)

def add_header(filename, orig_folder='.'):
    """ Read header from the original PDB and prepend it to the modified file """
    with open(orig_folder + filename, 'r') as f1:
        # Get ligand-receptor covalent bonds
        links = [line for line in f1.readlines() if line.startswith('LINK')]
    with open(filename,'r+') as f2:
        cont = links + f2.readlines()
        f2.seek(0)            # Go back to the beginng of the file
        f2.writelines(cont)    # Overwrite file from the beginning 
        f2.truncate()        # Delete the remaining of the old file
    return

def _fetch_ligand_expo_info(ligand_expo_id):
    """
    Fetch ligand data from ligand-expo.rcsb.org.
    """
    r = requests.get(f"http://ligand-expo.rcsb.org/reports/{ligand_expo_id[0]}/{ligand_expo_id}/")
    r.raise_for_status()
    html = BeautifulSoup(r.text, features="lxml")
    info = {}
    for table in html.find_all("table"):
        for row in table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) != 2:
                continue
            key, value = cells
            if key.string and key.string.strip():
                info[key.string.strip()] = "".join(value.find_all(string=True))

    # Postprocess some known values
    info["Molecular weight"] = float(info["Molecular weight"].split()[0])
    info["Formal charge"] = int(info["Formal charge"])
    info["Atom count"] = int(info["Atom count"])
    info["Chiral atom count"] = int(info["Chiral atom count"])
    info["Bond count"] = int(info["Bond count"])
    info["Aromatic bond count"] = int(info["Aromatic bond count"])
    return info

def get_ligands(pdb_id, comp_type='NON-POLYMER'):
    """
    RCSB has not provided a new endpoint for ligand information yet. As a
    workaround we are obtaining extra information from ligand-expo.rcsb.org,
    using HTML parsing. Check Talktorial T011 for more info on this technique!
    """
    if comp_type == 'NON-POLYMER':
        query = ("""{entry(entry_id: "%s")
        {nonpolymer_entities{pdbx_entity_nonpoly{comp_id}}}}""" % pdb_id)
        query_url = f"https://data.rcsb.org/graphql?query={query}"
        response = requests.get(query_url)
        response.raise_for_status()
        pdb_info = response.json()
        
        try:
            ligand_expo_ids = [nonpolymer_entities["pdbx_entity_nonpoly"]["comp_id"]
                                for nonpolymer_entities in pdb_info["data"]["entry"]["nonpolymer_entities"]]
        except:
            ligand_expo_ids = []
    # elif comp_type == 'PEPTIDE-LIKE':

    dict_lig = {}
    for ligand_expo_id in ligand_expo_ids:
        ligand_expo_info = _fetch_ligand_expo_info(ligand_expo_id)
        dict_lig[ligand_expo_id] = ligand_expo_info
    return dict_lig

def split_reclig(files, split_list=['rec', 'lig'], apo='rename', output_dir='.', renumber=True):
    """ Split complex into receptor and ligand """
    pdblist = read_files(files)

    lig_list = []
    for pdb in pdblist:
        try:
            # ParsePDB reads local file if the extension is given.
            # Otherwise it fetches the PDB from the internet.
            complex = prody.parsePDB(pdb, verbosity='none') 
            pdb_id = pdb.split('/')[-1].split('.')[0].upper()
            dict_lig = get_ligands(pdb_id)
            prex(dict_lig)
            cci_list = [x for x in dict_lig.keys() if x not in nonLOI_list]
            if len(cci_list) > 0:
                for cci in cci_list:
                    rec = complex.select('protein')
                    lig_tmp = complex.select('resname ' + cci)  # Select lig by name
                    lig_resi = set(lig_tmp.getResnums())
                    for resi in lig_resi:
                        lig_tmp = complex.select(f"resname {cci} and resnum {resi}")  # Possible same name, different resi
                        lig_chains = set(lig_tmp.getChids())
                        for chain in lig_chains:
                            lig = complex.select(f"resname {cci} and resnum {resi} and chain {chain}")  # Possible same name and resi, different chain
                            # Extract substructure PDBs
                            if 'lig' in split_list:
                                prody.writePDB(f"{output_dir}/{pdb_id}_ligand_{str(resi)}{chain}.pdb", lig, renumber=renumber)
                                print(f"*** Extracted ligand {cci} {resi}-{chain} from PDB {pdb_id} ***")
                            if 'complex' in split_list:
                                prody.writePDB(f"{output_dir}/{pdb_id}_complex_{str(resi)}{chain}.pdb", rec + lig, renumber=renumber)
                                print(f"*** Extracted complex of {cci} {resi}-{chain} from PDB {pdb_id} ***")

                            # Create mol-obj for the ligand
                            stream = StringIO()
                            prody.writePDBStream(stream, lig, renumber=renumber)
                            lig_string = stream.getvalue()
                            lig_mol = Chem.MolFromPDBBlock(lig_string, flavor=1, sanitize=False)

                            # Assign bond order to force correct valence
                            lig_smiles = dict_lig[cci]['Stereo SMILES (OpenEye)']
                            template = Chem.MolFromSmiles(lig_smiles)
                            lig_mol = AllChem.AssignBondOrdersFromTemplate(template, lig_mol)

                            # Retrieve the original PDB numbering
                            for atom in lig_mol.GetAtoms():
                                atom.SetAtomMapNum(atom.GetPDBResidueInfo().GetSerialNumber())

                            # # Calculate gap between rec and lig numbering
                            # # More info i fragment_methods > map_frag2lig
                            # rec_serial = rec.getSerials()[-1]
                            # lig_serial = lig.getSerials()[0]
                            # gap = abs(lig_serial - rec_serial)

                            lig_list.append([pdb_id, cci, resi, chain, lig_smiles,
                                            dict_lig[cci]['Name'],
                                            dict_lig[cci]['InChIKey descriptor'],
                                            dict_lig[cci]['Molecular weight'],
                                            lig_mol])
                            
                if 'rec' in split_list:
                    prody.writePDB(f"{output_dir}/{pdb_id}_receptor.pdb", rec, renumber=renumber)
                    print(f"*** Extracted protein from PDB {pdb_id} ***")

            elif len(cci_list) == 0 and apo in {'remove', 'delete'}:
                os.remove(pdb)
                print(f"*** {pdb_id} has no ligands of interest. Deleted. ***")
            elif len(cci_list) == 0 and apo == 'rename':
                os.rename(pdb, pdb.replace('.pdb','_apo'))
                print(f"*** {pdb_id} has no ligands of interest. Renamed. ***")
        except Exception as err:
            if rec is None:
                os.remove(pdb)
                print(f"*** {pdb_id} cannot be processed. Deleted. ***")
            else:
                print(f"*** Error with PDB {pdb_id}: {err} ***")

    df_lig = pd.DataFrame(lig_list, columns=['PDB', 'CCI', 'RESI', 'RESCHAIN', 'Isomeric_SMILES',
                                             'Name', 'InChIKey', 'MW', 'mol'])
    pickle.dump(df_lig, open(f"{output_dir}/Ligand_Database.pkl", 'wb+'))
    df_lig.drop(columns=['mol']).to_csv(f"{output_dir}/Ligand_Database.csv", header=True, index=False, sep=';')
    return(df_lig)

def merge_reclig(recfile, ligfile):
    rec = prody.parsePDB(recfile)
    lig = prody.parsePDB(ligfile)
    prody.writePDB(ligfile.replace('ligand', 'complex'), rec + lig)


    # with open(ligpdb, 'r') as lig:
    #     liglist = [line for line in lig.readlines() if 'ATOM' in line]
    # with open(recpdb, 'r') as rec:
    #     reclist = [line for line in rec.readlines() if 'ATOM' in line]
        
    # # last_num = str(int(reclist[-1].split()[1])
    # lig_residue = str(int(reclist[-1].split()[5]) + 1)
    # liglist = [x.replace('UNL 1', 'UNL A ' + lig_residue) for x in liglist]
    # reclist += liglist
    # open(outfile, 'w').write(''.join(reclist))
    return

def check_pdbs(files):
    pdblist = read_files(files)

    checked_pdbs = []
    for pdb_id in pdblist:
        pdb_data = describe_pdb(pdb_id)
        if pdb_data['rcsb_entry_info']['deposited_polymer_entity_instance_count'] < 5 and \
            pdb_data['rcsb_entry_info']['polymer_entity_count_dna'] == 0 and \
            pdb_data['rcsb_entry_info']['polymer_entity_count_rna'] == 0 and \
            pdb_data['rcsb_entry_info']['polymer_entity_count_nucleic_acid_hybrid'] == 0:
                checked_pdbs.append(pdb_id)
        # else:
        #     print(f"*** PDB {pdb_id} is DNA, RNA, or multimeric. Skipped. ***")
    return checked_pdbs

# def df_ligands(mol, pdbfiles, apo='rename', input_dir='.', outfile='RCSB_Ligands.csv', renumber=True):
#     lig_list = []
#     for pdb_file in pdbfiles:
#         try:
#             pdb_id = pdb_file.split('/')[-1].split('.')[0]
#             complex = prody.parsePDB(pdb_file, verbosity='none')
#             cci = mol.GetProp('_Name')
#             lig = complex.select(f"resname {cci}")
#             resi_chains = set(zip(lig.getResnums(), lig.getChids()))
#             for (resi, chain) in resi_chains:
#                 lig = complex.select(f"resname {cci} and resnum {resi} and chain {chain}")

#                 # Create mol-obj for the ligand
#                 stream = StringIO()
#                 prody.writePDBStream(stream, lig, renumber=renumber)
#                 lig_string = stream.getvalue()
#                 lig_mol = Chem.MolFromPDBBlock(lig_string, flavor=1, sanitize=False)

#                 # Assign bond order to force correct valence
#                 lig_smiles = dict_pdb_ligs[pdb_id][cci]['Stereo SMILES (OpenEye)']
#                 template = Chem.MolFromSmiles(lig_smiles)
#                 lig_mol = AllChem.AssignBondOrdersFromTemplate(template, lig_mol)

#                 # Retrieve the original PDB numbering
#                 for atom in lig_mol.GetAtoms():
#                     atom.SetAtomMapNum(atom.GetPDBResidueInfo().GetSerialNumber())

#                 lig_list.append([pdb_id, cci, resi, chain, lig_smiles,
#                                 dict_pdb_ligs[pdb_id][cci]['Name'],
#                                 dict_pdb_ligs[pdb_id][cci]['InChIKey descriptor'],
#                                 dict_pdb_ligs[pdb_id][cci]['Molecular weight'],
#                                 lig_mol])
#         except Exception as err:
#             print(f"*** Error with PDB {pdb_id}: {err} ***")

#     df_lig = pd.DataFrame(lig_list, columns=['PDB', 'CCI', 'RESI', 'RESCHAIN', 'Isomeric_SMILES',
#                                              'Name', 'InChIKey', 'MW', 'mol'])
#     pickle.dump(df_lig, open(outfile.replace('csv','pkl'), 'wb+'))
#     df_lig.drop(columns=['mol']).to_csv(outfile, header=True, index=False, sep=';')
#     return(df_lig)

def filter_ligands(pdblist):
    dict_pdb_ligs = {}
    for pdb_id in pdblist:
        # Create a dictionary for all the ligands in the PDB file
        dict_lig = get_ligands(pdb_id)
        # Keep only LOI ligands
        LOI_ccis = [x for x in dict_lig.keys() if x not in nonLOI_list]
        if len(LOI_ccis) > 0:
            subdict = {}
            wanted_keys = ['Stereo SMILES (OpenEye)', 'Name', 'InChIKey descriptor', 'Molecular weight']
            for cci in LOI_ccis:
                # Copy dict_lig with only wanted_keys
                subdict[cci] = {key: dict_lig[cci][key] for key in wanted_keys}
            dict_pdb_ligs[pdb_id] = subdict
    return dict_pdb_ligs


def extract_lig(mol, pdbfiles, renumber=False):
    lig_list = []
    for pdb in pdbfiles:
        try:
            pdb_id = pdb.split('/')[-1].split('.')[0]
            complex = prody.parsePDB(pdb, verbosity='none')
            cci = mol.GetProp('_Name')

            lig = complex.select(f"resname {cci}")
            resi_chains = set(zip(lig.getResnums(), lig.getChids()))

            for (resi, chain) in resi_chains:
                lig = complex.select(f"resname {cci} and resnum {resi} and chain {chain}")

                # Create mol-obj for the ligand
                stream = StringIO()
                prody.writePDBStream(stream, lig, renumber=renumber)
                lig_string = stream.getvalue()
                lig_mol = Chem.MolFromPDBBlock(lig_string, flavor=1, sanitize=False)

                # Assign bond order to force correct valence
                # lig_smiles = Chem.MolToSmiles(mol)
                # template = Chem.MolFromSmiles(lig_smiles)
                lig_mol = AllChem.AssignBondOrdersFromTemplate(mol, lig_mol)

                # Retrieve the original PDB numbering
                from rdkit.Chem import rdCoordGen
                for atom in lig_mol.GetAtoms():
                    atom.SetProp('numA', str(atom.GetIdx()))
                    atom.SetProp('numB', str(atom.GetPDBResidueInfo().GetSerialNumber()))

                # rdCoordGen.AddCoords(lig_mol)
                # Draw.MolToImage(lig_mol, size=(900,900)).show()

                lig_mol.SetProp('_Name', f"{cci}:{chain}:{resi}")
                lig_list.append([pdb_id, cci, chain, resi, lig_mol])
        except Exception as err:
            continue
            print(f"*** Error with PDB {pdb_id}: {err} ***")

    df = pd.DataFrame(lig_list, columns=['PDB', 'CCI', 'RESCHAIN_LIG', 'RESI_LIG', 'mol'])
    return df