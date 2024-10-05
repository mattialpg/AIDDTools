import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    import pypdb

import os, sys
from pathlib import Path
import redo, requests_cache
from io import StringIO
import numpy as np
import pandas as pd
import json
import contextlib
from tqdm.auto import tqdm
from tqdm.contrib import tzip
from copy import deepcopy

import prody
from rdkit import Chem
from rdkit.Chem import AllChem, rdDepictor, Draw
from rdkit.Chem.PropertyMol import PropertyMol

# Custom libraries
from tools import utils
from tools import GraphDecomp as GD
from tools import molecular_methods as MolMtd

# Caching requests will speed up repeated queries to PDB
requests_cache.install_cache('rcsb_pdb', backend='memory')

def replace_dummies(mol, replace_with='H', keep_info=True):
    mol_copy = deepcopy(mol)
    rwMol = Chem.RWMol(mol_copy)

    if keep_info:
        for atom in mol_copy.GetAtoms():
            atom.SetProp('_symbol', atom.GetSymbol())

    dummies = [a.GetIdx() for a in mol_copy.GetAtoms() if a.GetSymbol() == '*']

    # # Replace dummy atoms (this does not retain atom properties!)
    # d = Chem.MolFromSmiles('*'); h = Chem.MolFromSmiles(replace_with)
    # mol_copy = Chem.ReplaceSubstructs(mol_copy, d, h, replaceAll=True)[0]
    # return mol_copy

    # Replace dummy atoms and retain atom properties
    if replace_with == 'L':
        # Get atom type from fragmentation class
        for atom in mol_copy.GetAtoms():
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


def get_frags(mol, replace_dummies_with=False, **graphdecomp_kwargs):
    try:
        # L dummies only work with keep_connectivity=True
        # Force this behaviour to avoid errors
        if replace_dummies_with == 'L':
            graphdecomp_kwargs['keep_connectivity'] = True

        # Create a PropertyMol to keep properties after pickle dumping
        mol_copy = PropertyMol(deepcopy(mol))
        cci = mol_copy.GetProp('_Name')
        frag_mols = GD.GraphDecomp(mol_copy, **graphdecomp_kwargs)
    
        info = []
        for frag_mol in frag_mols:
            # Save original atom numbering as a property
            for atom in frag_mol.GetAtoms():
                if atom.GetSymbol() == '*':
                    atom.SetProp('_RDKAtomIndex', '*')
                else:
                    atom.SetProp('_RDKAtomIndex', str(atom.GetAtomMapNum()))
            
            if replace_dummies_with:
                frag_mol = replace_dummies(frag_mol, replace_dummies_with)

            # Remove atom map to get a standard smiles
            for atom in frag_mol.GetAtoms():
                atom.ClearProp('molAtomMapNumber')

            # Make list corresponding to a dataframe row
            # cols = ['CCI', 'FRAG_TYPE', 'ISOSMILES_FRAG', 'ROMol_FRAG']
            info.append([cci, frag_mol.GetProp('Type'),
                         Chem.MolToSmiles(frag_mol), frag_mol])
        return info
    except Exception as exc:
        print(f"*** Error while fragmenting ligand {cci}: {exc} ***\n")
        return []


def extract_lig(mol, pdb_id, pdb_dir, renumber=False):
    try:
        mol = deepcopy(mol)
        # Open PBD with prody
        complex = prody.parsePDB(f"{pdb_dir}/{pdb_id}.pdb",
                                 model=1, altloc=True,
                                 verbosity='none')

        # Get protein info
        rec = complex
        seq = list(zip(rec.getSerials(), rec.getNames()))

        #! #####  WORKAROUNDS TO FIX PROTEIN SEQUENCE  #####
        """
        We cannot access PDB atoms by simply referring to their index in the
        protatomnames list created by prody. This is because TER residues are neglected by prody.
        To reestablish the correspondence PDB atom number <-> prody list index,
        we need to add dummy 'X' atoms at the position of TER residues.
        """
        # Add atoms in alternative locations to the sequence 
        altlocs = complex.getCSLabels()
        if len(altlocs) > 1:
            for altloc in altlocs[1:]:
                rec = prody.parsePDB(f"{pdb_dir}/{pdb_id}.pdb", model=1,
                                     altloc=altloc, verbose='none')
                rec = rec.select(altloc)
                seq_alt = list(zip(rec.getSerials(), rec.getNames()))
                seq.extend(seq_alt)
        
        # Add TER residues to the sequence
        ter = complex.select('pdbter')
        for serial in zip(ter.getSerials()):
            seq.append((serial[0] + 1, 'TER'))

        # Add missing residues
        missing_res = list(set(range(1, len(seq) + 1)) - set([x[0] for x in seq]))
        seq = seq + [(x, 'X') for x in missing_res]
        #! ################################################

        # Sort sequence by atom number and convert to a string
        seq = sorted(seq, key=lambda x: x[0])
        seq_str = '-'.join([x[1] for x in seq])

        # Get ligand(s) info
        cci = mol.GetProp('_Name')
        lig = complex.select(f"resname {cci}")
        resnum_chains = set(zip(lig.getResnums(), lig.getChids()))
        #^^^^^ should add also altloc for ligands but I dont know how to implement it 

        # Looping over multiple ligands in the same PDB
        info = []
        for resnum, chain in resnum_chains:
            lig = complex.select(f"resname {cci} and resnum {resnum} and chain {chain}")

            # Create a pristine mol-object for the ligand
            stream = StringIO()
            prody.writePDBStream(stream, lig, renumber=renumber)
            lig_string = stream.getvalue()
            lig_mol = Chem.MolFromPDBBlock(lig_string, sanitize=True, removeHs=True)
            lig_mol.SetProp('_Name', f"{cci}:{chain}:{resnum}")

            # Copy bond order from mol to fix valence
            lig_mol = AllChem.AssignBondOrdersFromTemplate(mol, lig_mol)

            # Map lig_mol numbering to mol numbering (PDB-to-RDK)
            # num_order = list(lig_mol.GetSubstructMatch(mol))
            # ligatomlist = []
            # for num in num_order:
            #     atom = lig_mol.GetAtomWithIdx(num)
            #     atom_map_number = atom.GetPDBResidueInfo().GetSerialNumber()
            #     ligatomlist.append(atom_map_number)

            # Create a PropertyMol to keep properties even after pickle-dumping
            out_mol = PropertyMol(deepcopy(mol))
            # Set atomic property to track original indices in PDB
            for i, idx in enumerate(out_mol.GetSubstructMatch(lig_mol)):
                atom_mol = out_mol.GetAtomWithIdx(idx)
                atom_lig = lig_mol.GetAtomWithIdx(i)
                atom_mol.SetProp('_PDBAtomIndex', str(atom_lig.GetPDBResidueInfo().GetSerialNumber()))

            # Reset coordinates and highlighting
            MolMtd.reset_coordinates(out_mol)

            # cols = ['CCI', 'PDB', 'RESNUM_LIG', 'RESCHAIN_LIG', 'PROTATOMSEQ', 'ROMol_LIG']
            info.append([cci, pdb_id, resnum, chain, seq_str, out_mol])
        return info

    except Exception as exc:
        # print(exc)
        pass


def get_interactions(pdb_id, cci, pdb_dir='.', pymol=False):
    with warnings.catch_warnings():
        from plip.structure.preparation import PDBComplex
        from plip.exchange.report import BindingSiteReport

    try:
        complex = PDBComplex()
        complex.load_pdb(f"{pdb_dir}/{pdb_id}.pdb")
        tmp_list = []
        for ligand in complex.ligands:
            if ligand.hetid == cci:
                complex.characterize_complex(ligand)  # Analyse ligand interactions
                for plip_obj in complex.interaction_sets.values():
                    binding_site = BindingSiteReport(plip_obj)
                    for int_type in utils.int_types:
                        for y in getattr(binding_site, f"{int_type}_info"):
                            dict_int = dict(zip(getattr(binding_site, f"{int_type}_features"), y))
                            dict_int['INT_TYPE'] = int_type
                            dict_int['PDB'] = pdb_id
                            tmp_list.append(dict_int)
        df_int = pd.DataFrame(tmp_list)

        # Assign correct dtype to columns
        for col in df_int.columns.tolist():
            if col.endswith('COO'):
                df_int[col] = df_int[col].apply(lambda t: tuple(float(f"{x:.3f}") for x in t) if t == t else np.nan)
            elif 'LIST' in col:
                df_int[col] = df_int[col].apply(lambda x: x.split(',') if x == x else np.nan)
            elif 'IDX' in col:
                df_int[col] = df_int[col].apply(lambda x: f"{int(x)}" if x == x else np.nan)
            elif 'DIST' in col or 'ANGLE' in col:
                df_int[col] = df_int[col].apply(lambda x: float(f"{float(x):.2f}") if x == x else np.nan)

    #     if pymol is True:
    #         # Temporarily silence command-line
    #         with contextlib.redirect_stdout(None):
    #             from plip.basic import config
    #             from plip.basic.remote import VisualizerData
    #             from plip.visualization.visualize import visualize_in_pymol

    #             config.PYMOL = True  # Set plip to save the pse file
    #             pym_complexes = [
    #                 VisualizerData(complex, site)
    #                 for site in sorted(complex.interaction_sets)
    #                 if not len(complex.interaction_sets[site].interacting_res) == 0]
    #             for pym in pym_complexes:
    #                 visualize_in_pymol(pym)
        return df_int
    except Exception as exc:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print(exc_tb.tb_lineno)
        print(exc)
        pass

def merge_interactions(df_int):
    # Fill null values in one column with non-null values from another
    try: df_int['DONORIDX'] = df_int['DONORIDX'].combine_first(df_int['DON_IDX'])
    except: pass
    try: df_int['DONORIDX'] = df_int['DONORIDX'].combine_first(df_int['DONOR_IDX'])
    except: pass
    try: df_int['ACCEPTORIDX'] = df_int['ACCEPTORIDX'].combine_first(df_int['ACC_IDX'])
    except: pass
    try: df_int['ACCEPTORIDX'] = df_int['ACCEPTORIDX'].combine_first(df_int['ACCEPTOR_IDX'])
    except: pass

    # Use cosine law to get rec-lig distance for waterbridge interactions 
    try:
        D = np.round(np.sqrt(np.square(df_int['DIST_A-W']) + np.square(df_int['DIST_D-W']) - \
                    2*df_int['DIST_A-W'] * df_int['DIST_D-W'] * np.cos(df_int['WATER_ANGLE'])),2)
        df_int['DIST'] = df_int['DIST'].combine_first(D)
    except: pass
    try: df_int['DIST'] = df_int['DIST'].combine_first(df_int['DIST_D-A'])
    except: pass
    try: df_int['DIST'] = df_int['DIST'].combine_first(df_int['CENTDIST']).round(2)
    except: pass

    # Extract parameters based on interaction type
    protatomidx, ligatomidx = [], []
    for row in df_int.itertuples():
        if row.INT_TYPE in ('hbond', 'waterbridge'):
            if row.PROTISDON is True:
                protatomidx.append([row.DONORIDX])
                ligatomidx.append([row.ACCEPTORIDX])
            else:
                protatomidx.append([row.ACCEPTORIDX])
                ligatomidx.append([row.DONORIDX])
        elif row.INT_TYPE == 'hydrophobic':
            protatomidx.append([row.PROTCARBONIDX])
            ligatomidx.append([row.LIGCARBONIDX])
        elif row.INT_TYPE in ('pication', 'saltbridge'):
            protatomidx.append(row.PROT_IDX_LIST)
            ligatomidx.append(row.LIG_IDX_LIST)
        elif row.INT_TYPE == 'pistacking':
            protatomidx.append(row.PROT_IDX_LIST)
            ligatomidx.append(row.LIG_IDX_LIST)
        elif row.INT_TYPE == 'halogen':
            protatomidx.append([row.ACC_IDX] if isinstance(row.ACC_IDX, float) else row.ACC_IDX.split(','))
            ligatomidx.append([row.DON_IDX] if isinstance(row.DON_IDX, float) else row.DON_IDX.split(','))
        
    df_int['LIGATOMIDX'] = ligatomidx
    df_int['PROTATOMIDX'] = protatomidx

    cols = ['PDB', 'RESNR', 'RESTYPE', 'RESCHAIN',
            'RESNR_LIG', 'RESTYPE_LIG', 'RESCHAIN_LIG',
            'PROTATOMIDX', 'LIGATOMIDX', 'INT_TYPE', 'DIST']
    return(df_int[cols])


import asyncio, aiohttp, aiofile
import nest_asyncio; nest_asyncio.apply()
import socket
import random
import time
def async_get_info(urls):
    #TODO: Update as in https://towardsdatascience.com/responsible-concurrent-data-retrieval-80bf7911ca06

    semaphore = asyncio.BoundedSemaphore(1)
    connector = aiohttp.TCPConnector(limit_per_host=1)
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=20, sock_read=20)

    retries = 5
    async def fetch_info(session, url):
        # for i_retry in range(retries):
        t_req = {'duration': 1.5}
        async with semaphore, session.get(url=url, timeout=timeout, trace_request_ctx=t_req) as resp:
            try:
                assert resp.status == 200
                data = await resp.json()
                print(data, end="\r")
                async with aiofile.async_open('C:/Users/Idener/DEVSHEALTH/Q1_FragLIB/pdb_info.txt', 'a') as outfile:
                    await outfile.write(f"{json.dumps(data)}'\n'")
                time.sleep(.2)
                # return data

                # # Critical sleep to ensure that load does not exceed PubChem's thresholds
                # min_time_per_request = 1.1
                # if t_req['duration'] < min_time_per_request:
                #     idle_time = min_time_per_request - t_req['duration']
                #     await asyncio.sleep(idle_time)

                # if resp.status == 200:  # Successful response
                #     return data
                # elif resp.status == 503:  # PubChem server busy, we will retry
                #     if i_retry == retries - 1:
                #         return data
                # else:  # Unsuccessful response
                #     print(f"Failed")
                #     return np.nan
            except Exception as exc:
                print(exc)

    async def main():
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [asyncio.ensure_future(fetch_info(session, url)) for url in urls]
            results = await asyncio.gather(*tasks)
        return results
    
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    info = asyncio.run(main())
    return info

def async_download_files(urls, format='txt', outdir='.', fname_sep="/", fnames=None):
    user_agents = [
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/22.0.1207.1 Safari/537.1",
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:55.0) Gecko/20100101 Firefox/55.0",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.101 Safari/537.36",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/22.0.1207.1 Safari/537.1",
        "Mozilla/5.0 (X11; CrOS i686 2268.111.0) AppleWebKit/536.11 (KHTML, like Gecko) Chrome/20.0.1132.57 Safari/536.11",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.6 (KHTML, like Gecko) Chrome/20.0.1092.0 Safari/536.6",
        "Mozilla/5.0 (Windows NT 6.0) AppleWebKit/536.5 (KHTML, like Gecko) Chrome/19.0.1084.36 Safari/536.5",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
        "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_0) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
        "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1062.0 Safari/536.3",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1062.0 Safari/536.3",
        "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
        "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
        "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.6 (KHTML, like Gecko) Chrome/20.0.1090.0 Safari/536.6",
        "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/19.77.34.5 Safari/537.1",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/536.5 (KHTML, like Gecko) Chrome/19.0.1084.9 Safari/536.5",
        "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.0 Safari/536.3",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/535.24 (KHTML, like Gecko) Chrome/19.0.1055.1 Safari/535.24",
        "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/535.24 (KHTML, like Gecko) Chrome/19.0.1055.1 Safari/535.24"]

    async def fetch_file(session, semaphore, url, fname):
        try:
            if not fname:
                fname = url.split(fname_sep)[-1]
            headers = {'User-Agent': random.choice(user_agents)}
            async with semaphore, session.get(url, headers=headers, timeout=3) as resp:
                resp.raise_for_status()
                if format == 'json':
                    data = await resp.json()
                    async with aiofile.async_open(f"{outdir}/{fname}.json", 'w') as outfile:
                        await outfile.write(json.dumps(data, indent=2))
                elif format == 'xml':
                    data = await resp.read()
                    async with aiofile.async_open(f"{outdir}/{fname}.xml", 'wb') as outfile:
                        await outfile.write(data)
                else:
                    data = await resp.read().decode('UTF-8')
                    async with aiofile.async_open(f"{outdir}/{fname}.txt", 'w') as outfile:
                        await outfile.write(data)
                print(f"*** [Downloading] File {fname} has been fetched ***")
        except Exception as exc:
            print(f"*** [Error] File {fname} cannot be fetched: {exc} ***", end='\x1b[1K\r')
            open(f"{outdir}/{fname}.txt", 'a').close()  # Create an empty file


    async def main():
        try:
            os.makedirs(outdir, exist_ok=True)
            semaphore = asyncio.BoundedSemaphore(100)  #3
            # connector = aiohttp.TCPConnector(limit_per_host=5)  # Number of simultaneous connections
            connector = aiohttp.TCPConnector(family=socket.AF_INET)  # Number of simultaneous connections

            async with aiohttp.ClientSession(connector=connector) as session:
                if fnames:
                    for url, fname in tzip(urls, fnames, desc="Downloading Files"):
                        await fetch_file(session, semaphore, url, fname)
                        await asyncio.sleep(.1)  # Add delay between downloads
                else:
                    for url in tqdm(urls, desc="Downloading Files"):
                        await fetch_file(session, semaphore, url, fname=None)
                        await asyncio.sleep(.1)  # Add delay between downloads
        except: pass

    # Run the event loop
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())


import py3Dmol
from ipywidgets import interact, interactive, fixed
def drawit(mol, confId=-1):
    p = py3Dmol.view(width=700, height=450)
    p.removeAllModels()
    p.addModel(Chem.MolToMolBlock(mol, confId=confId),'sdf')
    p.setStyle({'stick':{}})
    p.setBackgroundColor('0xeeeeee')
    p.zoomTo()
    return p.show()

def drawit_slider(mol):    
    p = py3Dmol.view(width=700, height=450)
    interact(drawit, mol=fixed(mol), p=fixed(p), confId=(0, mol.GetNumConformers()-1))

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

import signal, time

class Timeout():
    """Timeout class using ALARM signal"""
    class Timeout(Exception): pass
    
    def __init__(self, sec):
        self.sec = sec

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.raise_timeout)
        signal.alarm(self.sec)

    def __exit__(self, *args):
        signal.alarm(0) # disable alarm

    def raise_timeout(self, *args):
        raise Timeout.Timeout()
