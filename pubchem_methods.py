import numpy as np
import pandas as pd
import pubchempy as pcp

import requests
from urllib.request import urlretrieve

import json
import xml.etree.ElementTree as ET
from xml.dom import minidom

import time
import asyncio

#!----------------- PubChem --------------------#

def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)
    return wrapped

# @background
def download_compound(identifier, namespace='cid', output_dir='.'):
    try:
        url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/{namespace.lower()}/{identifier}/JSON'
        resp = requests.get(url)
        if namespace.lower() == 'cid':
            cid = identifier
        elif namespace.lower() == 'smiles':
            cid = resp.json()['PC_Compounds'][0]['id']['id']['cid']

        with open(f"{output_dir}/{cid}.comp", 'w') as f:
            f.write(json.dumps(resp.json(), indent=1))
    except Exception as exc:
        print(exc)
    return

def get_compound(identifier, namespace='cid'):
    """
    namespace: [cid, name, smiles, sdf, inchi, inchikey, formula]
    USAGE:
        cpds = [PubMtd.get_compound(x) for x in CID_list]
        cols = ['cid', 'ISOSMILES', 'InChIKey', 'IUPAC_NAME', 'FORMULA', 'MW', 'xLogP']
        df_cpds = pd.DataFrame(cpds, columns=cols)

    """
    try:
        identifier = identifier.replace(' ', '%20') if namespace=='name' else identifier
        url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/{namespace.lower()}/{identifier}/JSON'
        resp = requests.get(url)
        comp = pcp.Compound(resp.json()['PC_Compounds'][0])
        data = [str(comp.cid), comp.isomeric_smiles, comp.inchikey, comp.iupac_name,
                comp.molecular_formula, float(comp.molecular_weight), comp.xlogp]
        return data  #TODO: This should be a dictionary!!
    except Exception as exc:
        return [np.nan*7]

def download_taxonomy(cid, output_dir):
    try:
        cid = str(cid)
        url = 'https://pubchem.ncbi.nlm.nih.gov/sdq/sdqagent.cgi?infmt=json&outfmt=csv&query={"download":"*","collection":"consolidatedcompoundtaxonomy","where":{"ands":[{"cid":"%s"}]},"order":["cid,asc"]}' % cid
        urlretrieve(url, f"{output_dir}/{cid}.taxonomy")
    except Exception as exc:
        print(f"{cid}: {exc}")
    return

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

# Use it inside your function
def read_or_get(dbentry, dbdir, verbose=False):
    """
    dbentry is in the format 'namespace:id' (e.g. 'rc:RC00304')
    Parse local file or get from the server
    """
    singleton = KEGGSingleton()  # Singleton instance
    kegg = singleton.kegg         # Access KEGG instance
    rest = singleton.rest         # Access REST instance

    filename = '.'.join(dbentry.split(':')[::-1])
    try:
        with open(f"{dbdir}/{filename}", 'r') as text:
            dict_data = kegg.parse(text.read())
        if verbose: print('Reading file...')
    except:
        kegg_entry = rest.kegg_get(dbentry).read()
        with open(f"{dbdir}/{filename}", 'w', encoding='utf-8') as file:
            file.write(kegg_entry)
            dict_data = kegg.parse(kegg_entry)
        if verbose: print('Downloading file...')
    return dict_data

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

