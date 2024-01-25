#rest api: https://data.rcsb.org/#rest-api
#graphql: https://data.rcsb.org/#gql-api
#list of graphql attributes: https://data.rcsb.org/data-attributes.html


import json
import requests 
import urllib
from urllib.request import urlopen
import pandas as pd
from rdkit import Chem

def query_for_pdb(gene_name, n=1000):
    """Uses the REST API to query RCSB for *nonpolymer* entities
    return_type = "entry" would just be PDB id, but 
    non_polymer_entity refers to everything not protein (i.e. ligands)"""
    
    base_url = 'https://search.rcsb.org/rcsbsearch/v1/query'
    json_query_string = f"""
    {{
      "query": {{
        "type": "terminal",
        "service": "text",
        "parameters": {{
          "attribute": "rcsb_entity_source_organism.rcsb_gene_name.value",
          "operator": "exact_match",
          "value": "{gene_name}"
        }}
      }},
      "request_options": {{
        "pager": {{
          "start": 0,
          "rows": {n}
        }}
      }},
      "return_type": "non_polymer_entity"
    }}
    """
    
    #formulate the query:
    query = urllib.parse.quote(json_query_string)
    req_url = base_url+'?json={request}'
    url_query = req_url.format(request=query)

    #make the query:
    response = urlopen(url_query)
    
    #read the output into json:
    output = json.loads(response.read())
    
    return output

def query_for_component_id(nonpol_ids):
    #base graphql query:
    base_url = 'https://data.rcsb.org/graphql'
    
    #all graphql queries require format: ["id1", "id2", so on...]
    query_fmt = '[' + ', '.join(['"' + i + '"' for i in nonpol_ids]) + ']'
    
    #put the query together
    graphql_query = """
    {
      nonpolymer_entities(entity_ids:""" + query_fmt +""") {
        rcsb_nonpolymer_entity_container_identifiers {
          entry_id
          entity_id
          nonpolymer_comp_id
        }
      }
    }
    """
    
    #encode as an html request and get the json data
    r = requests.post(base_url, json={'query': graphql_query})
    output = r.json()
    return output

def query_for_smiles(component_ids):
    #base graphql query:
    base_url = 'https://data.rcsb.org/graphql'
    
    #all graphql queries require format: ["id1", "id2", so on...]
    query_fmt = '[' + ', '.join(['"' + i + '"' for i in component_ids]) + ']'
    
    #put the query together
    graphql_query = """
    {
      chem_comps(comp_ids:""" + query_fmt +""") {
        chem_comp {
          id
        }
        rcsb_chem_comp_descriptor {
          SMILES_stereo
        }
      }
    }
    """
    
    #encode as an html request and get the json data
    r = requests.post(base_url, json={'query': graphql_query})
    output = r.json()
    return output

    
def get_npPDB_entries(gene_name):
    """Query the REST API for PDB codes + nonpol IDs"""
    
    output = query_for_pdb(gene_name=gene_name)
    if output['total_count']>1000:
        output= query_for_pdb(gene_name=gene_name, n=output['total_count'])
    #parse the output into a sensible 
    return [i['identifier'] for i in output['result_set']]

def get_chemical_components(np_ids):
    output = query_for_component_id(np_ids)
    #parse output:
    entity_ids = list()
    chem_comps = list()
    for i in output['data']['nonpolymer_entities']:
        results = i['rcsb_nonpolymer_entity_container_identifiers']

        pdb_id = results['entry_id']
        entity_id = results['entity_id']
        chemical_component_id = results['nonpolymer_comp_id']
        entity_ids.append(pdb_id+'_'+entity_id)
        chem_comps.append(chemical_component_id)
        
    return pd.DataFrame( {'nonpol_ID' : entity_ids, 'chem_comp' : chem_comps} )

def get_smiles(chem_comp):
    chem_comp = list(set(chem_comp))
    output = query_for_smiles(chem_comp)
    names = list()
    smiles = list()
    for i in output['data']['chem_comps']:
        name = i['chem_comp']['id']
        smi = i['rcsb_chem_comp_descriptor']['SMILES_stereo']
        names.append(name)
        smiles.append(smi)
    return pd.DataFrame({'chem_comp':names, 'smiles':smiles})
    
df = pd.DataFrame()
np_ids = get_npPDB_entries('PARP1')
df['PDB_ID'] = [i[:-2] for i in np_ids]
df['nonpol_ID'] = np_ids
df = df.merge(get_chemical_components(np_ids), on='nonpol_ID')
df = df.merge(get_smiles(df['chem_comp']), on='chem_comp')
print(df)