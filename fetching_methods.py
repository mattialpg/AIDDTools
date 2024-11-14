import os, time, sys, glob
import asyncio, requests
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import pubchempy as pcp

import json
import xml.etree.ElementTree as ET
from xml.dom import minidom

from tqdm.auto import tqdm
from tqdm.contrib import tzip

# Custom libraries
# from tools import utils
# from tools import GraphDecomp as GD
# from tools import molecular_methods as MolMtd

# Caching requests will speed up repeated queries to PDB
import requests_cache
requests_cache.install_cache('rcsb_pdb', backend='memory')

#!----------------- PubChem --------------------#

# def background(f):
#     def wrapped(*args, **kwargs):
#         return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)
#     return wrapped

def get_compound(entry, server, dbdir, verbose=False):
    """
        Entry is in the format 'namespace:id' (e.g. 'rc:RC00304')
        Parse local file or get from the server
    """
    namespace, identifier = entry.split(':')
    filename = f"{identifier}.{namespace}"
    file_path = f"{dbdir}/{filename}"

    not_found = []

    if server == 'pubchem':
        """
            namespace: [cid, name, smiles, sdf, inchi, inchikey, formula]
        """
        if not os.path.exists(file_path) or namespace != 'cid':
            try:
                if verbose: print('Downloading file...')
                url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/{namespace}/{identifier}/JSON'
                resp = requests.get(url)
                cid = resp.json()['PC_Compounds'][0]['id']['id']['cid']
                with open(f"{dbdir}/{cid}.cid", 'w') as f:
                    f.write(json.dumps(resp.json(), indent=1))
            except:
                not_found.append(entry)
                return {}

        if verbose: print('Reading file...')
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
            with open(f"{dbdir}/{filename}", 'r') as text:
                dict_data = kegg.parse(text.read())
            if verbose: print('Reading file...')
        except:  # Download from server
            kegg_entry = rest.kegg_get(entry).read()
            with open(f"{dbdir}/{filename}", 'w', encoding='utf-8') as file:
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
from Bio.KEGG import REST
from bioservices.kegg import KEGG

class KEGGSingleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(KEGGSingleton, cls).__new__(cls)
            cls.kegg = KEGG(*args, **kwargs)  # Instantiate KEGG once
            cls.rest = REST  # Instantiate REST once
        return cls._instance

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
                    async with aiofile.async_open(f"{outdir}/{fname}.{format}", 'wb') as outfile:
                        await outfile.write(data)
                elif format == 'cif':
                    data = await resp.read()
                    async with aiofile.async_open(f"{outdir}/{fname}", 'wb') as outfile:
                        await outfile.write(data)
                else:
                    data = await resp.read().decode('UTF-8')
                    async with aiofile.async_open(f"{outdir}/{fname}.txt", 'w') as outfile:
                        await outfile.write(data)
                print(f"*** [Downloading] File {fname} has been fetched ***")
        except Exception as exc:
            print(f"*** [Error] File {fname} cannot be fetched: {exc} ***", end='\x1b[1K\r')
            # with open(f"{outdir}/.skipped.txt", 'a'):


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