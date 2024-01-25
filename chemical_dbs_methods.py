import io
import seaborn as sns
from bioservices import UniProt, PDB
from bioservices import pdb
import pandas as pd
import json
from pprint import pprint
from textwrap import wrap
import urllib, requests

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

def download_cci(cci, outdir):
    try:
        cci_url = f"https://files.rcsb.org/ligands/view/{cci}.cif"
        response = requests.get(cci_url)
        with open(f"{outdir}/{cci}.cif", 'w') as f:
            f.write(response.text)
    except: pass
    return

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