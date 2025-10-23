import os, time, sys, glob
import requests
from urllib.request import urlretrieve

import aiofile, aiohttp
import random
import socket
import asyncio, nest_asyncio
# Apply nest_asyncio to avoid RuntimeError in Jupyter
# nest_asyncio.apply()

import numpy as np
import pandas as pd

import json
import xml.etree.ElementTree as ET
from xml.dom import minidom

from tqdm.auto import tqdm
from tqdm.contrib import tzip

# Custom libraries
# from tools import utils
# from tools import GraphDecomp as GD
# from tools import molecular_methods as MolMtd

#!----------------- PubChem --------------------#

# def background(f):
#     def wrapped(*args, **kwargs):
#         return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)
#     return wrapped

def get_compound(entry, server, dbdir):
    """
        Entry is in the format 'namespace:id' (e.g. 'rc:RC00304')
        Parse local file or get from the server
    """
    namespace, identifier = entry.split(':')

    not_found = []

    if server == 'pubchem':
        """
            namespace: [cid, name, smiles, sdf, inchi, inchikey, formula]
        """
        import pubchempy as pcp

        if not os.path.exists(f"{dbdir}/{identifier}.{namespace}") or namespace != 'cid':
            try:
                url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/{namespace}/{identifier}/JSON'
                resp = requests.get(url)
                cid = resp.json()['PC_Compounds'][0]['id']['id']['cid']
                with open(f"{dbdir}/{cid}.cid", 'w') as f:
                    f.write(json.dumps(resp.json(), indent=1))
            except:
                not_found.append(entry)
                return {}

        content = json.load(open(f"{dbdir}/{cid}.cid", 'r'))
        comp = pcp.Compound(content['PC_Compounds'][0])
        dict_data = {'cid': str(comp.cid),
                     'isomeric_smiles': comp.isomeric_smiles,
                     'inchikey': comp.inchikey,
                     'iupac_name': comp.iupac_name,
                     'molecular_formula': comp.molecular_formula,
                     'molecular_weight': comp.molecular_weight,
                     'xlogp': comp.xlogp}
        return dict_data

    elif server == 'kegg':
        singleton = KEGGSingleton()  # Singleton instance
        kegg = singleton.kegg        # Access KEGG instance
        rest = singleton.rest        # Access REST instance

        try:
            with open(f"{dbdir}/{identifier}.{namespace}", 'r') as text:
                dict_data = kegg.parse(text.read())
            if verbose: print('Reading file...')
        except:  # Download from server
            kegg_entry = rest.kegg_get(entry).read()
            with open(f"{dbdir}/{identifier}.{namespace}", 'w', encoding='utf-8') as file:
                file.write(kegg_entry)
                dict_data = kegg.parse(kegg_entry)
            if verbose: print('Downloading file...')
        return dict_data

    if not_found:
        with open(f"{dbdir}/not_found.txt", 'r') as f:
            lines = f.readlines()
            lines = set(lines + not_found)
        with open(f"{dbdir}/not_found.txt", 'a') as f:
            f.write('\n'.join(lines))
    

# # @background
# def download_compound(identifier, namespace='cid', output_dir='.'):
#     """
#     namespace: [cid, name, smiles, sdf, inchi, inchikey, formula]
#     """
#     try:
#         url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/{namespace.lower()}/{identifier}/JSON'
#         resp = requests.get(url)
#         if namespace.lower() == 'cid':
#             cid = identifier
#         elif namespace.lower() == 'smiles':
#             cid = resp.json()['PC_Compounds'][0]['id']['id']['cid']

#         with open(f"{output_dir}/{cid}.comp", 'w') as f:
#             f.write(json.dumps(resp.json(), indent=1))
#     except Exception as exc:
#         print(exc)
#     return

# def get_compound(identifier, namespace='cid', output_dir='.'):
#     """
#     namespace: [cid, name, smiles, sdf, inchi, inchikey, formula]
#     USAGE:
#         cpds = [PubMtd.get_compound(x) for x in CID_list]
#         cols = ['cid', 'ISOSMILES', 'InChIKey', 'IUPAC_NAME', 'FORMULA', 'MW', 'xLogP']
#         df_cpds = pd.DataFrame(cpds, columns=cols)
#     """
#         try:
#             identifier = identifier.replace(' ', '%20') if namespace=='name' else identifier
#             url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/{namespace.lower()}/{identifier}/JSON'
#             resp = requests.get(url)
#             comp = pcp.Compound(resp.json()['PC_Compounds'][0])
#             data = [str(comp.cid), comp.isomeric_smiles, comp.inchikey, comp.iupac_name,
#                     comp.molecular_formula, float(comp.molecular_weight), comp.xlogp]
#             return data  #TODO: This should be a dictionary!!
#         except Exception as exc:
#             return [np.nan*7]

def get_taxonomy(cid, output_dir=None):
    try:
        if output_dir:
            sdq = 'https://pubchem.ncbi.nlm.nih.gov/sdq/sdqagent.cgi?infmt=json&outfmt=csv&query='
            url = sdq + '{"download":"*","collection":"consolidatedcompoundtaxonomy","where":{"ands":\
                         [{"cid":"%s"}]},"order":["cid,asc"],"start":1,"limit":10000}' % cid
            urlretrieve(url, f"{output_dir}/{cid}.taxonomy")
        else:
            sdq = 'https://pubchem.ncbi.nlm.nih.gov/sdq/sdqagent.cgi?infmt=json&outfmt=json&query='
            url = sdq + '{"select":"*","collection":"consolidatedcompoundtaxonomy","where":{"ands":\
                         [{"cid":"%s"}]},"order":["cid,asc"],"start":1,"limit":10000}' % cid
            resp = requests.get(url)
            return resp.json()['SDQOutputSet'][0]['rows']
    except Exception as exc:
        print(f"{cid}: {exc}")
    return


def get_from_taxid(taxid):
    try:
        sdq = 'https://pubchem.ncbi.nlm.nih.gov/sdq/sdqagent.cgi?infmt=json&outfmt=json&query='
        url = sdq + '{"select":"*","collection":"consolidatedcompoundtaxonomy","where":{"ands":[{"taxid":"%s"}]},"order":["cid,asc"]}' % taxid
        resp = requests.get(url)
        return resp.json()['SDQOutputSet'][0]['rows']
    except Exception:
        return {}


def download_info(cid, output_dir):
    try:
        url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON'
        resp = requests.get(url)
        if resp.status_code == 200:
            with open(f"{output_dir}/{cid}.info", 'w') as f:
                f.write(json.dumps(resp.json(), indent=1))
    except Exception as exc:
        print(f"{cid}: {exc}")
    return
    
def download_synonyms(cid, output_dir):
    try:
        url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/synonyms/JSON'
        resp = requests.get(url)
        if resp.status_code == 200:
            with open(f"{output_dir}/{cid}.syn", 'w') as f:
                f.write(json.dumps(resp.json(), indent=1))
    except Exception as exc:
        print(f"{cid}: {exc}")
    return

def load_data(cid, input_dir):
    import pubchempy as pcp

    cid = str(int(cid))
    try:
        content = json.load(open(f"{input_dir}/{cid}.comp", 'r'))
        comp = pcp.Compound(content['PC_Compounds'][0])
        data = [str(comp.cid), comp.isomeric_smiles, comp.inchikey, comp.iupac_name,
                comp.molecular_formula, comp.molecular_weight, comp.xlogp]
        return data
    except (IOError, OSError):
        return [cid] + [np.nan]*6

def load_name(cid, input_dir):
    cid = str(int(cid))
    try:
        content = json.load(open(f"{input_dir}/{cid}.info", 'r'))
        name = [str(content['Record']['RecordNumber']), ''.join(content['Record']['RecordTitle'])]
        return name
    except (IOError, OSError):
        return [cid, np.nan]
              
def load_synonyms(cid, input_dir):
    cid = str(int(cid))
    try:
        content = json.load(open(f"{input_dir}/{cid}.syn", 'r'))
        synonyms = [str(content['InformationList']['Information'][0]['cid']),
                    content['InformationList']['Information'][0]['Synonym']]
        return synonyms
    except (IOError, OSError):
        return [cid, np.nan]

def get_similar_compounds(smiles, threshold=95, n_records=10, attempts=5):
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/similarity/smiles/{smiles}/JSON?Threshold={threshold}&MaxRecords={n_records}"
        r = requests.get(url)
        r.raise_for_status()
        key = r.json()["Waiting"]["ListKey"]

        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/listkey/{key}/cids/JSON"
        # print(f"Querying for job {key} at URL {url}...", end="")
        while attempts:
            r = requests.get(url)
            r.raise_for_status()
            response = r.json()
            if "IdentifierList" in response:
                cids = response["IdentifierList"]["cid"]
                break
            attempts -= 1
            time.sleep(10)
        else:
            raise ValueError(f"Could not find matches for job key: {key}")
    except:
        cids = []
    return cids


#!------------------ KEGG ---------------------#

# class KEGGSingleton:
#     from Bio.KEGG import REST
#     from bioservices.kegg import KEGG
    
#     _instance = None

#     def __new__(cls, *args, **kwargs):
#         if cls._instance is None:
#             cls._instance = super(KEGGSingleton, cls).__new__(cls)
#             cls.kegg = KEGG(*args, **kwargs)  # Instantiate KEGG once
#             cls.rest = REST  # Instantiate REST once
#         return cls._instance

#!----------------- Others --------------------#

def download_chebi(chebi_id, outdir):
    chebi_url = f"https://www.ebi.ac.uk/webservices/chebi/2.0/test/getCompleteEntity?chebiId=CHEBI:{chebi_id}"
    response = requests.get(chebi_url)
    if response.status_code == 200:
        root = ET.fromstring(response.text)
        xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
        with open(f"{outdir}/{chebi_id}.comp", 'w') as file:
            file.write(xml_str)
    return



import asyncio, aiohttp, aiofile
import nest_asyncio; nest_asyncio.apply()
import socket
import random
import time

user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_2) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.3 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:122.0) Gecko/20100101 Firefox/122.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 OPR/108.0.0.0",
    "curl/7.88.1"
    ]

def async_get_info(urls):
    #TODO: Update as in https://towardsdatascience.com/responsible-concurrent-data-retrieval-80bf7911ca06

    global user_agents
    semaphore = asyncio.BoundedSemaphore(1)
    connector = aiohttp.TCPConnector(limit_per_host=1)
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=20, sock_read=20)
    retries = 5

    async def fetch_info(session, url):
        # for i_retry in range(retries):
        t_req = {'duration': 1.5}
        try:
            # # Critical sleep to ensure that load does not exceed PubChem's thresholds
            # min_time_per_request = 1.1
            # if t_req['duration'] < min_time_per_request:
            #     idle_time = min_time_per_request - t_req['duration']
            #     await asyncio.sleep(idle_time)
            async with semaphore, session.get(url=url, timeout=timeout, trace_request_ctx=t_req) as resp:
                if resp.status == 200:  # Successful response
                    data = await resp.json()  # Assuming the response is JSON
                    return data
                elif resp.status == 503:  # PubChem server busy, retrying
                    if i_retry == retries - 1:
                        return np.nan  # Return np.nan after retries
                    await asyncio.sleep(2)  # Wait before retrying
                else:  # Unsuccessful response
                    print(f"Failed with status {resp.status} for URL {url}")
                    return np.nan
        except Exception as exc:
            print(f"Error fetching {url}: {exc}")
            return np.nan

    async def main():
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [asyncio.ensure_future(fetch_info(session, url)) for url in urls]
            results = await asyncio.gather(*tasks)
        return results
    
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    info = asyncio.run(main())
    return info


async def async_download_files(urls, format='txt', outdir='.', fname_sep="/", fnames=None):
    global user_agents

    async def fetch_file(session, semaphore, url, fname):
        try:
            if not fname:
                parsed = urlparse(url)
                fname = os.path.basename(parsed.path) or "default_filename"

            headers = {'User-Agent': random.choice(user_agents)}

            async with semaphore, session.get(url, headers=headers, timeout=10) as resp:
                if format == 'json':
                    data = await resp.json()
                    async with aiofile.async_open(f"{outdir}/{fname}.json", 'w') as outfile:
                        await outfile.write(json.dumps(data, indent=2))
                else:
                    data = await resp.read()
                    async with aiofile.async_open(f"{outdir}/{fname}.{format}", 'wb') as outfile:
                        await outfile.write(data)
                print(f"*** [SUCCESS] File {fname} fetched ***")

            await asyncio.sleep(random.uniform(0.5, 3.0))  # Randomize sleep to look more "human" to the server

        except asyncio.TimeoutError:
            print(f"*** [TIMEOUT] File {fname} took too long ***")
        except Exception as exc:
            print(f"*** [ERROR] File {fname} cannot be fetched: {exc} ***")
            traceback.print_exc()


    async def main():
        print(f"[DEBUG] Starting downloads for {len(urls)} URLs")
        os.makedirs(outdir, exist_ok=True)
        semaphore = asyncio.BoundedSemaphore(1)
        connector = aiohttp.TCPConnector(family=socket.AF_INET)

        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [fetch_file(session, semaphore, url, fnames[i] if fnames else None)
                     for i, url in enumerate(urls)]
            await asyncio.gather(*tasks)

    await main()


def download_chembl(chembl_id, outdir, info='comp'):
    try:
        if info == 'comp':
            chembl_url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}.json"
        elif info == 'act':
            chembl_url = f"https://www.ebi.ac.uk/chembl/api/data/activity.json?molecule_chembl_id__in={chembl_id}&limit=10000"
        response = requests.get(chembl_url)
        
        # if seaching by inchi:
        # chembl_id = requests.get(chembl_url).json()['molecule_chembl_id']
        
        with open(f"{outdir}/{chembl_id}.{info}", 'w') as f:
            f.write(json.dumps(response.json(), indent=2))
    except: pass
    return

#!----------------- RCSB --------------------#
# Caching requests will speed up repeated queries to PDB
# import requests_cache
# requests_cache.install_cache('rcsb_pdb', backend='memory')

def download_ccis(cci_list, outdir):

    cci_downloaded = [os.path.basename(file).strip('.cif') for file in glob.glob(f"{outdir}/*cif")]
    cci_skipped = open(f"{outdir}/.skipped.txt").readlines()
    cci_to_download = [x for x in cci_list if x not in cci_downloaded and x not in cci_skipped]

    if cci_to_download:
        urls = [f"https://files.rcsb.org/ligands/view/{cci}.cif" for cci in cci_to_download]
        print(urls)
        async_download_files(urls, format='cif', outdir=outdir)

def download_pdb_info(pdb_id, outdir):

    try:
        pdb_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
        response = requests.get(pdb_url)
        with open(f"{outdir}/{pdb_id}.json", 'w') as f:
            f.write(json.dumps(response.json(), indent=2))
    except: pass
    return


import aiohttp
import asyncio
import os
import glob
from tqdm.asyncio import tqdm_asyncio

async def fetch_and_save(session, url, dest, pdb_id, sem):
    async with sem:  # limit concurrent downloads
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=300)) as resp:  # 5 min max per file
                if resp.status == 200:
                    with open(dest, "wb") as f:
                        async for chunk in resp.content.iter_chunked(8192):
                            f.write(chunk)
        except Exception as e:
            print(f"Error downloading {pdb_id}: {e}")

async def download_pdbs_async(pdb_ids, outdir='.', file_type="pdb", max_concurrent=5):
    """
    Download PDB or mmCIF structures asynchronously from the RCSB archive.

    Parameters
    ----------
    pdb_ids : list[str]
        List of PDB identifiers.
    outdir : str
        Output directory for downloaded files.
    file_type : str, 'pdb' or 'mmcif'
        Choose which format to download.
    max_concurrent : int
        Maximum number of concurrent downloads.
    """

    file_type = file_type.lower()
    if file_type not in {"mmcif", "pdb"}:
        raise ValueError("file_type must be 'mmcif' or 'pdb'")

    os.makedirs(outdir, exist_ok=True)

    if file_type == "mmcif":
        ext = ".cif"
        base_url = "https://files.rcsb.org/download"
    else:
        ext = ".pdb"
        base_url = "https://files.rcsb.org/download"

    # Already downloaded
    pdb_downloaded = [os.path.basename(f).replace(ext, "").lower()
                      for f in glob.glob(os.path.join(outdir, f"*{ext}"))]
    pdb_to_download = [x.lower().strip() for x in pdb_ids if x.lower().strip() not in pdb_downloaded]

    if not pdb_to_download:
        print("All requested PDBs already downloaded.")
        return

    connector = aiohttp.TCPConnector(limit=max_concurrent)
    sem = asyncio.Semaphore(max_concurrent)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for pdb_id in pdb_to_download:
            filename = f"{pdb_id}{ext}"
            url = f"{base_url}/{filename}"
            dest = os.path.join(outdir, filename)
            tasks.append(fetch_and_save(session, url, dest, pdb_id, sem))

        for f in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Downloading PDBs"):
            await f

# # Read BLAST results (json file)
# dict_hit = {}
# with open('ncbiblast.json') as f:
	# data = json.load(f)
	# consensus = data['hits'][0]['hit_hsps'][0]['hsp_qseq'].replace('-','')
	# for d in data['hits']:
		# dict_hit[d['hit_acc']] = d['hit_hsps'][0]['hsp_hseq'].replace('-','')

# # Query UniProt
# s = UniProt(verbose=False)
# df = pd.DataFrame()
# for h in dict_hit.keys():
	# result = s.search(h, frmt="tsv", columns="accession,id,length,organism_name,protein_existence,xref_pdb")
	# df1 = pd.read_table(io.StringIO(result.replace(';',',')))
	# df = pd.concat([df, df1], axis=0)

# # Add sequence column from dictionary	
# df['Sequence'] = df['Entry'].map(dict_hit)#.reset_index()
# df['Seq. Length']  = df['Sequence'].str.len()
# open('blast.csv', 'w').write(df.to_csv(sep=';', line_terminator='\n', index=False))

# # Drop rows containing specific words/values
# df = df[df['Organism'].str.contains('Zika|Dengue|Japanese|Nile') == True]
# df = df[df['Protein existence'].str.contains('homology') == False]
# df = df.loc[df['Seq. Length'] >= len(consensus)*0.9 ]
# df = df.sort_values('Entry Name')
# print(df)

# entries = df['Entry'].tolist()
# names = df['Entry Name'].tolist()
# with open('seq_all.fasta', 'w') as s:
	# for e,n in zip(entries,names):
		# fasta = df.loc[df['Entry'] == e, 'Sequence'].values[0]
		# wrapfasta = wrap(fasta, width=80)
		# s.write('>%s|%s\n' %(e,n))
		# s.write('\n'.join(wrapfasta) + '\n')
		
# # # Write txt file with pdb list
# # pdb_list = df['PDB'].tolist()
# # txtpdb = ''.join(pdb_list)
# # open('pdb_list.txt', 'w').write(txtpdb)

# # # Query RCSB
# # s = PDB(verbose=False)
# # for pdb in pdb_list:
	# # result = s.search(pdb, frmt="tsv", columns="entry,assembly,polymer_entity")
# # print(result)