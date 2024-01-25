import os
from pathlib import Path
import redo, requests_cache
from io import StringIO
import numpy as np
import pandas as pd
import json
import prody
from rdkit import Chem
from rdkit.Chem import AllChem, rdCoordGen, Draw
from urllib.request import urlretrieve
from graphdecomp import *
from plip.structure.preparation import PDBComplex
from plip.exchange.report import BindingSiteReport
import contextlib
from tqdm.auto import tqdm
from tqdm.contrib import tzip

import warnings
warnings.simplefilter('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=PendingDeprecationWarning)
    import pypdb

# Custom libraries
from utils import *
import feature_methods as FeatMtd

# Caching requests will speed up repeated queries to PDB
requests_cache.install_cache('rcsb_pdb', backend='memory')

def download_files(urls, output_dir):
    for url in tqdm(urls):
        try:
            fname = url.split("/")[-1]
            urlretrieve(url, os.path.join(output_dir, fname))
            print(f"*** [Downloading] File {fname} being fetched ***", end="\r")
        except:
            print(f"*** [Error] File {fname} cannot be fetched ***")
    return

def extract_lig(mol, pdb_id, pdb_dir, renumber=False):
    info = []
    try:
        # Open PBD with prody
        complex = prody.parsePDB(f"{pdb_dir}/{pdb_id}.pdb",
                                 model=1, verbosity='none')

        # Get protein info
        rec = complex.select('protein')
        protatomnames = rec.getNames()
        protatomnums = rec.getSerials()

        # Get ligand(s) info
        cci = mol.GetProp('_Name')
        lig = complex.select(f"resname {cci}")
        resi_chains = set(zip(lig.getResnums(), lig.getChids()))

        # Looping over multiple ligands in the same PDB
        for resi, chain in resi_chains:
            lig = complex.select(f"resname {cci} and resnum {resi} and chain {chain}")

            # Create mol-obj for the ligand
            stream = StringIO()
            prody.writePDBStream(stream, lig, renumber=renumber)
            lig_string = stream.getvalue()
            lig_mol = Chem.MolFromPDBBlock(lig_string, sanitize=True, removeHs=True)
            lig_mol.SetProp('_Name', f"{cci}:{chain}:{resi}")

            # Assign bond order to force correct valence (mol is template)
            lig_mol = AllChem.AssignBondOrdersFromTemplate(mol, lig_mol)

            # Use lig_copy to get the correct RDK numbering used during ligand fragmentation
            lig_copy = Chem.MolFromSmiles(mol.GetProp('Smiles'))
            num_order = list(lig_mol.GetSubstructMatch(lig_copy))  # Retrieve the original numbering order
            ligatomlist = []
            for num in num_order:
                atom = lig_mol.GetAtomWithIdx(num)
                ligatomlist.append(atom.GetPDBResidueInfo().GetSerialNumber())

            # Reset mol_LIG coordinates for display
            AllChem.Compute2DCoords(lig_mol)
            # Delete substructure highlighting
            del lig_mol.__sssAtoms

            info.append([pdb_id, cci, chain, resi, protatomnums,
                         protatomnames, ligatomlist, lig_mol])

    except Exception as err:
        pass
        # print(f"*** Error with PDB {pdb_id}: {err} ***")
        # print([f"{x.GetProp('_Name')} = {x.GetNumAtoms()}" for x in [mol, lig_mol]])
        # broken = [mol, lig_mol]
        # Draw.MolsToGridImage(broken, molsPerRow=2,
        #                     subImgSize=(1000,800),
        #                     legends=[x.GetProp('_Name') for x in broken]).show()
    
    return info

def substitute_dummies(mol, replace_with='[H]'):
    if replace_with == 'L':
        # Get atom type from fragmentation class
        mw = Chem.RWMol(mol)
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                d = atom.GetProp('dummyLabel')
                mw.ReplaceAtom(atom.GetIdx(), Chem.rdchem.Atom(d))
        return mw
    else:
        # Replace dummy atoms
        mol = Chem.ReplaceSubstructs(
            mol,
            Chem.MolFromSmiles('*'),
            Chem.MolFromSmiles(replace_with),
            replaceAll=True)[0]
        try:
            return Chem.RemoveHs(mol, sanitize=True)
        except:
            return Chem.RemoveHs(mol, sanitize=False)


def get_frags(mol, replace_dummies=False, **graphdecomp_kwargs):
    try:
        if replace_dummies == 'L':
            # connectivity dummies only WORK with keep_connectivity=True
            # Here we force this behaviour to avoid errors
            graphdecomp_kwargs['keep_connectivity'] = True

        cci = mol.GetProp('_Name')
        frag_mols = GraphDecomp(mol, **graphdecomp_kwargs)
    
        info = []
        for frag_id, frag_mol in enumerate(frag_mols):
            # Retrieve original atom numbering
            fragatomlist_rdk = []
            for atom in frag_mol.GetAtoms():
                if atom.GetAtomicNum() != 0:
                    fragatomlist_rdk.append(atom.GetAtomMapNum())
            
            # Delete atom numbering
            for atom in frag_mol.GetAtoms():
                atom.ClearProp('molAtomMapNumber')
                atom.ClearProp('atomNote')

            if replace_dummies:
                frag_mol = substitute_dummies(frag_mol, replace_dummies)

            # Make list corresponding to a dataframe row
            info.append([cci, f"{frag_id+1:02}", frag_mol.GetProp('Type'),
                         Chem.MolToSmiles(frag_mol), mol.GetProp('Smiles'), 
                         fragatomlist_rdk, frag_mol])
    except Exception as exc:
        print(f"*** Error while fragmenting ligand {cci}: {exc} ***\n")
    return info


# @profile
def _get_dict_sites(pdb, pymol=False):
    complex = PDBComplex()
    complex.load_pdb(pdb)

    # Analyse ligand interactions
    for ligand in complex.ligands:
        complex.characterize_complex(ligand)

    dict_sites = {}
    # Loop over binding sites
    for site, plip_obj in sorted(complex.interaction_sets.items()):
        # Collect data about interactions
        binding_site = BindingSiteReport(plip_obj)
        # Create interaction dictionary
        dict_ints = {x: [getattr(binding_site, x + '_features')] +
                           getattr(binding_site, x + '_info')
                           for x in int_types}
        dict_sites[site] = dict_ints

    if pymol is True:
        # Temporarily silence command-line
        with contextlib.redirect_stdout(None):
            from plip.basic import config
            from plip.basic.remote import VisualizerData
            from plip.visualization.visualize import visualize_in_pymol

            config.PYMOL = True  # Tell plip to save the pse file
            pym_complexes = [
                VisualizerData(complex, site)
                for site in sorted(complex.interaction_sets)
                if not len(complex.interaction_sets[site].interacting_res) == 0]
            for pym in pym_complexes:
                visualize_in_pymol(pym)

    return dict_sites


def _get_df_site(dict_ints):
    df_site = pd.DataFrame()
    for int_type in int_types:
        int_feature, *int_info = dict_ints[int_type]
        df_tmp = pd.DataFrame(int_info, columns=int_feature)
        df_tmp['INT_TYPE'] = int_type
        df_site = pd.concat([df_site, df_tmp], axis=0)
    return df_site


# @profile
def df_interactions(pdb_id, ligand, pdb_dir='.', pymol=False):# apo=None
    try:
        pdb = f"{pdb_dir}/{pdb_id}.pdb"

        dict_sites = _get_dict_sites(pdb, pymol=pymol)
        if ligand:
            dict_sites = {k:v for k, v in dict_sites.items() if ligand in k}

        if dict_sites:
            df_list = [_get_df_site(dict_ints) for dict_ints in dict_sites.tolist()()]
            df_int = pd.concat(df_list)
            df_int.insert(0, 'PDB', pdb_id)
            # print(f"*** [Analysing]  PDB {pdb_id} has binding site(s) {list(dict_sites.keys())} ***")
        # else:
        #     print(f"*** [Analysing]  PDB {pdb_id} has no binding sites ***")
        return df_int
    except Exception as exc:
        # pass
        print(f"*** Cannot analyse interations: {exc} ***")

    # try:
    #     # df1['LIGCARBONIDX'] = df1['LIGCARBONIDX'].astype(int)
    #     # df1['PROTCARBONIDX'] = df1['PROTCARBONIDX'].astype(int)
    #     if ligand:
    #         # Filter again because sometimes df_interactions fails with covalent ligands 
    #         df_int_cci = df_int_cci.loc[df_int_cci['RESN_LIG'] == ligand]
    #     # if outfile:
    #     #     df1.dropna(axis=1, how='all').to_csv(outfile, header=True, index=False, sep=',', line_terminator='\n') # type: ignore
    # except:
    #     pass


def merge_interactions(df_int):
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
    protatomidx, ligatomidx, dist = [], [], []
    for row in df_int.itertuples():
        if row.INT_TYPE in ('hbond', 'waterbridge'):
            if row.PROTISDON is True:
                protatomidx.append([row.DONORIDX])
                ligatomidx.append([row.ACCEPTORIDX])
                dist.append(row.DIST)
            else:
                protatomidx.append([row.ACCEPTORIDX])
                ligatomidx.append([row.DONORIDX])
                dist.append(row.DIST)
        elif row.INT_TYPE == 'hydrophobic':
            protatomidx.append([row.PROTCARBONIDX])
            ligatomidx.append([row.LIGCARBONIDX])
            dist.append(row.DIST)
        elif row.INT_TYPE in ('pication', 'saltbridge'):
            protatomidx.append(row.PROT_IDX_LIST.split(','))
            ligatomidx.append(row.LIG_IDX_LIST.split(','))
            dist.append(row.DIST)
        elif row.INT_TYPE == 'pistacking':
            protatomidx.append(row.PROT_IDX_LIST.split(','))
            ligatomidx.append(row.LIG_IDX_LIST.split(','))
            dist.append(row.CENTDIST)
        elif row.INT_TYPE == 'halogen':
            protatomidx.append([row.ACC_IDX] if isinstance(row.ACC_IDX, float) else row.ACC_IDX.split(','))
            ligatomidx.append([row.DON_IDX] if isinstance(row.DON_IDX, float) else row.DON_IDX.split(','))
            dist.append(row.DIST)
        
    df_int['LIGATOMIDX'] = ligatomidx
    df_int['PROTATOMIDX'] = protatomidx
    df_int['DIST'] = dist

    df_int['DIST'] = df_int['DIST'].astype(float).round(3)
    df_int['LIGATOMIDX'] = df_int['LIGATOMIDX'].apply(lambda lst: list(map(int, lst)))
    df_int['PROTATOMIDX'] = df_int['PROTATOMIDX'].apply(lambda lst: list(map(int, lst)))

    cols= ['PDB', 'RESI', 'RESN', 'RESCHAIN', 'RESI_LIG', 'RESN_LIG', 'RESCHAIN_LIG',
           'PROTATOMIDX', 'LIGATOMIDX', 'INT_TYPE', 'DIST']
    return(df_int[cols])


def get_fragdistr(df_merged, df_int):
    df_int = df_int.rename(columns={'RESN_LIG': 'CCI'})

    # Map all interactions to all fragments
    df_out = pd.merge(df_int, df_merged, on=['PDB', 'RESI_LIG', 'CCI', 'RESCHAIN_LIG'], how='left')
    df_out = df_out.dropna(subset=['FRAGATOMLIST_PDB'])

    # Map PROTATOMIDX to PROTATOMNUMS
    indices = [[list(y).index(x) if x in y else np.nan for x in X]
               for X, y in tzip(df_out['PROTATOMIDX'].tolist(), df_out['PROTATOMNUMS'].tolist(),
                                desc='Mapping PROTATOMIDX to PROTATOMNUMS')] 
    
    # Map PROTATOMNUMS to PROTATOMNAMES
    protatomtypes = [[list(x)[i] if i == i else np.nan for i in I]
                     for I, x in tzip(indices, df_out['PROTATOMNAMES'].tolist(),
                                      desc='Mapping PROTATOMNUMS to PROTATOMNAMES')]
    df_out['PROTATOMTYPES'] = protatomtypes
        
    # Map LIGATOMIDX to FRAGATOMLIST_PDB
    fragatomidx = [[list(y).index(x) if x in y else np.nan for x in X]
               for X, y in tzip(df_out['LIGATOMIDX'].tolist(), df_out['FRAGATOMLIST_PDB'].tolist(),
                                desc='Mapping LIGATOMIDX to FRAGATOMLIST_PDB')] 
    # Assign nan to tuples where atoms belongs to different fragments
    fragatomidx = [x if not np.isnan(x).any() else np.nan for x in fragatomidx]
    df_out['FRAGATOMIDX'] = fragatomidx

    df_out = df_out.dropna(subset=['PROTATOMTYPES', 'FRAGATOMIDX'])
    return df_out

import asyncio, aiohttp, aiofile
import nest_asyncio; nest_asyncio.apply()
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

def async_download_files(urls, format, outdir='.'):

    async def fetch_file(session, semaphore, url):
        try:
            fname = url.split("/")[-1]
            async with semaphore, session.get(url, timeout=3) as resp:
                resp.raise_for_status()
                if format == 'json':
                    data = await resp.json()
                    async with aiofile.async_open(f"{outdir}/{fname}.json", 'w') as outfile:
                        await outfile.write(json.dumps(data, indent=2))
                print(f"*** [Downloading] File {fname} being fetched ***", end="\r")
        except Exception as exc:
            print(f"*** [Error] File {fname} cannot be fetched: {exc} ***")

    async def main():
        os.makedirs(outdir, exist_ok=True)
        semaphore = asyncio.BoundedSemaphore(3)
        connector = aiohttp.TCPConnector(limit_per_host=5)  # Number of simultaneous connections

        async with aiohttp.ClientSession(connector=connector) as session:
            for url in tqdm(urls, desc="Downloading Files"):
                await fetch_file(session, semaphore, url)
                await asyncio.sleep(.15)  # Add delay between downloads

    # Run the event loop
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())