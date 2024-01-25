import numpy as np
import pandas as pd
import pubchempy as pcp
import requests
import json
from urllib.request import urlretrieve

import asyncio
import time
def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)
    return wrapped

# @background
def download_compound(identifier, namespace='CID', output_dir='.'):
    try:
        if namespace.lower() == 'cid':
            url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{identifier}/JSON'
            resp = requests.get(url)
            cid = identifier
        elif namespace.lower() == 'smiles':
            url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{identifier}/JSON'
            resp = requests.get(url)
            cid = resp.json()['PC_Compounds'][0]['id']['id']['cid']

        with open(f"{output_dir}/{cid}.comp", 'w') as f:
            f.write(json.dumps(resp.json(), indent=1))
    except Exception as exc:
        print(exc)
    return

def download_taxonomy(CID, output_dir):
    try:
        CID = str(CID)
        url = 'https://pubchem.ncbi.nlm.nih.gov/sdq/sdqagent.cgi?infmt=json&outfmt=csv&query={"download":"*","collection":"consolidatedcompoundtaxonomy","where":{"ands":[{"cid":"%s"}]},"order":["cid,asc"]}' % CID
        urlretrieve(url, f"{output_dir}/{CID}.taxonomy")
    except Exception as exc:
        print(f"{CID}: {exc}")
    return

def download_info(CID, output_dir):
    try:
        url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{CID}/JSON'
        resp = requests.get(url)
        if resp.status_code == 200:
            with open(f"{output_dir}/{CID}.info", 'w') as f:
                f.write(json.dumps(resp.json(), indent=1))
    except Exception as exc:
        print(f"{CID}: {exc}")
    return
    
def download_synonyms(CID, output_dir):
    try:
        url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{CID}/synonyms/JSON'
        resp = requests.get(url)
        if resp.status_code == 200:
            with open(f"{output_dir}/{CID}.syn", 'w') as f:
                f.write(json.dumps(resp.json(), indent=1))
    except Exception as exc:
        print(f"{CID}: {exc}")
    return

def load_data(CID, input_dir):
    CID = str(int(CID))
    try:
        content = json.load(open(f"{input_dir}/{CID}.comp", 'r'))
        comp = pcp.Compound(content['PC_Compounds'][0])
        data = [str(comp.cid), comp.isomeric_smiles, comp.inchikey, comp.iupac_name,
                comp.molecular_formula, comp.molecular_weight, comp.xlogp]
        return data
    except (IOError, OSError):
        return [CID] + [np.nan]*6

def load_name(CID, input_dir):
    CID = str(int(CID))
    try:
        content = json.load(open(f"{input_dir}/{CID}.info", 'r'))
        name = [str(content['Record']['RecordNumber']), ''.join(content['Record']['RecordTitle'])]
        return name
    except (IOError, OSError):
        return [CID, np.nan]
              
def load_synonyms(CID, input_dir):
    CID = str(int(CID))
    try:
        content = json.load(open(f"{input_dir}/{CID}.syn", 'r'))
        synonyms = [str(content['InformationList']['Information'][0]['CID']),
                    content['InformationList']['Information'][0]['Synonym']]
        return synonyms
    except (IOError, OSError):
        return [CID, np.nan]
        
import requests
import time

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
                cids = response["IdentifierList"]["CID"]
                break
            attempts -= 1
            time.sleep(10)
        else:
            raise ValueError(f"Could not find matches for job key: {key}")
    except:
        cids = []
    return cids