import sys, glob
import numpy as np
import pandas as pd
from pprint import pprint

# Dictionary of standard amino acids
standard_AAs = {'VAL':'V', 'ILE':'I', 'LEU':'L', 'GLU':'E', 'GLN':'Q', 'ASP':'D', 'ASN':'N',
                'HIS':'H', 'TRP':'W', 'PHE':'F', 'TYR':'Y', 'ARG':'R', 'LYS':'K', 'SER':'S',
                'THR':'T', 'MET':'M', 'ALA':'A',  'GLY':'G', 'PRO':'P', 'CYS':'C'}

# List of modified amino acids
modified_AAs = {'060': 'CSC[C@H](C(=O)O)N', '0QL': 'C(CSSC[C@@H](C(=O)O)N)N',
                '2CO': 'C([C@@H](C(=O)O)N)SOO', '85F': 'C([C@@H](C(=O)O)N)SC[C@@H](C(=O)O)N',
                'CAS': 'C[As](C)SC[C@@H](C(=O)O)N', 'CMT': 'COC(=O)[C@H](CS)N',
                'CSO': 'C([C@@H](C(=O)O)N)SO', 'CZ2': 'C([C@@H](C(=O)O)N)S[As](O)O',
                'CZZ': 'C([C@@H](C(=O)O)N)S[As]O', 'D11': 'C[C@@H]([C@H](C(=O)O)N)OP(=O)(O)O',
                'DAL': 'C[C@H](C(=O)O)N', 'DAR': 'C(C[C@H](C(=O)O)N)CNC(=[NH2+])N',
                'DAS': 'C([C@H](C(=O)O)N)C(=O)O', 'DCY': 'C([C@H](C(=O)O)N)S',
                'DGL': 'C(CC(=O)O)[C@H](C(=O)O)N', 'DGN': 'C(CC(=O)N)[C@H](C(=O)O)N',
                'DHI': 'c1c([nH+]c[nH]1)C[C@H](C(=O)O)N', 'DIL': 'CC[C@@H](C)[C@H](C(=O)O)N',
                'DLE': 'CC(C)C[C@H](C(=O)O)N', 'DLY': 'C(CCN)C[C@H](C(=O)O)N',
                'DPN': 'c1ccc(cc1)C[C@H](C(=O)O)N', 'DPR': 'C1C[C@@H](NC1)C(=O)O',
                'DSG': 'C([C@H](C(=O)O)N)C(=O)N', 'DSN': 'C([C@H](C(=O)O)N)O',
                'DTH': 'C[C@@H]([C@H](C(=O)O)N)O', 'DTR': 'c1ccc2c(c1)c(c[nH]2)C[C@H](C(=O)O)N',
                'DTY': 'c1cc(ccc1C[C@H](C(=O)O)N)O', 'DVA': 'CC(C)[C@H](C(=O)O)N',
                'ECX': 'CCSC[C@@H](C(=O)O)N', 'FGP': '[C@H]([C@@H](O)OP(=O)(O)O)(C(=O)O)N', 
                'H14': 'c1ccc(cc1)[C@H]([C@@H](C(=O)O)N)O', 'KPR': 'C[C@@H](C(=O)/C=C/OP(=O)(O)O)N',
                'M0H': 'C([C@@H](C(=O)O)N)SCO', 'MED': 'CSCC[C@H](C(=O)O)N', 
                'MIS': 'CC(C)O[P@](=O)(O)OC[C@@H](C(=O)O)N', 'OMH': 'CO[P@](=O)(O)OC[C@@H](C(=O)O)N',
                'PTR': 'c1cc(ccc1C[C@@H](C(=O)O)N)OP(=O)(O)O', 'SDP': 'CCOP(=O)(OCC)OC[C@@H](C(=O)O)N',
                'SEP': 'C([C@@H](C(=O)O)N)OP(=O)(O)O', 'TPO': 'C[C@H]([C@@H](C(=O)O)N)OP(=O)(O)O',
                'YTH': 'C[C@@H]([C@@H](C(=O)O)N)OP(=O)(O)O'}

similar_AAs = {'DAB': 'C(CN)C(C(=O)O)N', 'AIB': 'CC(C)(C(=O)O)N'}

# List of non-LOI (ligand of interest)
nonLOI_list = ['ACE', 'HEX', 'TMA', 'SOH', 'P25', 'CCN', 'PTN', 'TCN', 'BU1', 'BCN', 'CB3', 'HCS', 'GLC',
               'SO2', 'MO6', 'MOH', 'CAC', 'MLT', '6PH', 'MOS', 'MO3', 'CD3', 'ACM', 'LUT', 'NBN', 'OC3',
               'PMS', 'OF3', 'SCN', 'DHB', 'E4N', '13P', '3PG', 'CYC', 'BEN', 'NAO', 'PHQ', 'EPE', 'BME',
               'ETE', 'OES', 'EAP', 'ETX', 'BEZ', '5AD', 'OC2', 'OLA', 'GD3', 'CIT', 'DVT', 'OC6', 'MW1',
               'SRT', 'LCO', 'BNZ', 'PPV', 'STE', 'PEG', 'PGE', 'MPO', 'B3P', 'OGA', 'IPA', 'EDO', 'MAC',
               '9PE', 'IPH', 'MBN', 'C1O', '1PE', 'YF3', 'PEF', '8PE', 'DKA', 'GGD', 'SE4', 'LHG', '1PG',
               'SMO', 'DGD', 'CMO', 'MLI', 'MW2', 'DTT', 'DOD', '7PH', 'PBM', 'FOR', 'PSC', 'TG1', 'KAI',
               'DGA', 'PE4', 'VO4', 'ACN', 'MO4', 'OCL', '6UL', 'CHT', 'RHD', 'CPS', 'IR3', 'OC4', 'MTE',
               'HGC', 'PC1', 'HC4', 'TEA', 'BOG', 'PEO', 'PE5', '144', 'IUM', 'LMG', 'SQU', 'MMC', '2FU',
               'AU3', '3PH', 'PT4', 'PGO', 'ICT', 'OCM', 'BCR', 'PG4', 'L4P', 'OPC', 'OXM', 'SQD', 'PQ9',
               'PL9', 'IRI', '15P', 'MAE', 'MBO', 'FMT', 'L1P', 'DUD', 'PGV', 'CD1', 'P33', 'DTU', 'XAT',
               'THE', 'MW3', 'BHG', 'OCT', 'BET', 'MPD', 'HTO', 'IBM', 'D01', 'HAI', 'HED', 'CAD', 'NVP',
               'CUZ', 'TLA', 'SO4', 'OC5', 'ETF', 'MRD', 'PHB', 'URE', 'MLA', 'TGL', 'PLM', 'NET', 'LAC',
               'UNX', 'DMS', 'MO2', 'THJ', 'NHE', 'HAE', 'MO1', 'DAO', '3PE', 'LMU', 'DHJ', 'C10', 'AUC',
               'FLC', 'SAL', 'GAI', 'ORO', 'HEZ', 'TAM', 'TRA', 'NEX', 'CXS', 'LCP', 'OCN', 'PER', 'ACY',
               'ARS', '12P', 'L3P', 'PUT', 'NAW', 'GUN', 'CON', 'C2O', 'EMC', 'BO4', 'BNG', 'HTG', 'MH2',
               'MN5', 'CYN', 'H2S', 'MH3', 'YT3', 'P22', 'KO4', '1AG', 'IPL', 'PG6', 'MO5', 'F09', 'BAM',
               'TRS', 'EOH', 'GCP', 'MSE', 'AKR', 'NCO', 'PO4', 'L2P', 'LDA', 'SIN', 'DMI', 'DTD', 'PCF',
               'SGM', 'DIO', 'PPI', 'DDQ', 'DPO', 'HCA', 'CO5', 'NA6', 'NAG', 'ENC', 'NA5', 'NA2', 'DD9',
               'LI1', 'P4C', 'GLV', 'DMF', 'ACT', 'BTB', '6PL', 'BGL', 'OF1', 'N8E', 'LMT', 'THM', 'EU3',
               'FOL', '543', 'PEK', 'NSP', 'PEE', 'OCO', 'CHD', 'CO2', 'CMO', 'TBU', 'UMQ', 'MES', 'NH4', 'CD5', 
               'DEP', 'OC1', 'KDO', '2PE', 'PE3', 'IOD', 'NDG', 'TAU', 'ULI', 'PG5', 'A2G', '6PE', 'L44', 'PGR', 
               'TCA', 'SPD', 'SPM', 'SAR', 'SUC', 'PAM', 'SPH', 'BE7', 'P4G', 'OLC', 'OLB', 'LFA', 'D10', 'D12', 
               'HP6', 'R16', 'PX4', 'TRD', 'UND', 'FTT', 'MYR', 'RG1', 'IMD', 'DMN', 'KEN', 'C14', 'UPL', 'CMJ', 
               'MYS', 'TWT', 'M2M', 'P15', 'PG0', 'PEU', 'AE3', 'TOE', 'ME2', 'PE8', '6JZ', '7PE', 'P3G', '7PG', 
               '16P', 'XPE', 'PGF', 'AE4', '7E8', '7E9', 'MVC', 'TAR', 'DMR', 'LMR', 'NER', '02U', 'NGZ', 'LXB', 
               'BM3', 'NAA', 'NGA', 'LXZ', 'PX6', 'PA8', 'LPP', 'PX2', 'MYY', 'PX8', 'PD7', 'XP4', 'XPA', 'PEV', 
               'PEX', 'PEH', 'PTY', 'YB2', 'PGT', 'CN3', 'AGA', 'DGG', 'CD4', 'CN6', 'CDL', 'PG8', 'MGE', 'DTV', 
               'L2C', '4AG', 'B3H', '1EM', 'DDR', 'I42', 'CNS', 'PC7', 'HGP', 'PC8', 'HGX', 'LIO', 'PLD', 'PC2', 
               'MC3', 'P1O', 'PLC', 'PC6', 'HSH', 'BXC', 'HSG', 'DPG', '2DP', 'POV', 'PCW', 'GVT', 'CE9', 'CXE', 
               'CE1', 'SPJ', 'SPZ', 'SPK', 'SPW', 'HT3', 'HTH', '2OP', '3NI', 'BO3', 'DET', 'D1D', 'SWE', 'SOG',
               'CA', 'ZN', 'MG', 'KR', 'PB', 'TB', 'EU', 'NC', 'SR', 'RU', 'GD', 'LU', 'IR', 'AG', 'RB', 'YB',
               'AU', 'CR', 'NA', 'U1', 'Y1', 'LA', 'NI', 'TE', 'PT', 'PI', 'PR', 'GA', 'IN', 'CS', 'SB', 'SX'
               'HO', 'AL', 'OS', 'OH', 'PD', 'CE', 'CL', 'HG', 'XE', 'TL', 'BA', 'LI', 'BR', 'SM', 'CD', 'MN',
               'F', 'W', 'K', 'N', 'O', 'FE', 'FE2', 'CU', 'AR', 'CO3', 'BCT', '7YO', 'PCA', 'VO4', 'OXY'
               'IOD', 'CO', 'NO3', 'FLC', 'SCN', 'CAC', 'NH4', 'CU1', 'BA', 'ALF', 'NO2', 'MLT', 'OXL', 'SO3',
               'YB', 'OAA', 'WO4', 'YT3', 'IUM', 'PO3', 'EMC', '3CO', 'EU3', 'IR3', '2PO', 'AUC',
               'LCP', 'GA', 'RE', 'RH3', '3NI', 'SE4', 'PT4', 'PBM', 'D8U', 'ER3', 'RHD', 'VN3', 'RH',
               'SB', 'TH', '4TI', 'V', 'BS3', 'PTN', 'ND4', 'AM', '0BE', 'TCN', 'CF', 'LCO', 'CUL', 'ZCM',
               'DY', 'SFL', 'PDV', 'IN', 'OS4', '4PU', 'TA0', 'YB2', 'ZR', 'GOL', 'PEG', 'DMS', 'ACE',
               'MPD', 'MES', 'TRS', 'PG4', 'PGE', 'NH2', 'FMT', 'SF4', 'EPE', 'CIT', 'BME', 'ACY', 'IMD', '1PE',
               'MLI', 'FES', 'UNX', 'IPA', 'MRD', 'TLA', 'P6G', 'POP', 'F3S', 'BTB', 'EOH', 'DTT', 'NHE',
               'PYR', 'BEF', 'DIO', 'MLA', 'PGO', '2PE', 'XE', 'B3P', 'PE4', 'TAR', 'PPV', 'TAM', 'PG0',
               'CUA', '0QE', 'P33', 'AU', 'AF3', 'AG', 'ARS', 'BO3', 'ICS', 'CF0', 'TBR', 'AU3', 'TAS', 'BO4',
               '2T8', 'AST', 'ART', 'BF2', '6BP', 'D6N', 'BF4', 'ICE', '8AR', '0KA', 'RMO', 'HG2', '82N', 'ICH',
               'ICZ', 'ICG', '8P8', 'NOB', '202', 'HOH', 'DOD', '1PG', '15P', '12P', 'PG6', 'PE5', 'PE8', '7PE',
               'PE3', 'PG5', 'ETE', 'XPE', 'PEU', 'P15', '7PG', 'P4K', '33O', '9FO',
               # Saccharides
               '3MK', '4GL', '4N2', 'AFD', 'ALL', 'BGC', 'BMA', 'GAL', 'GIV', 'GL0', 'GLA', 'GLC', 
               'GUP', 'GXL', 'MAN', 'SDY', 'SHD', 'WOO', 'Z0F', 'Z2D', 'Z6H', 'Z8H', 'Z8T', 'ZCD',
               'ZEE',
               # Glycolsilated
               'AKA', 'AKY', '3VL', 'AKT', 'ERT', 'DRA', 'JB0', 'LIV', 'B6M', 'N30', 'PAR', 'RIO',
               'NMY', 'TOY', 'JS4', '7XP', '8UZ', '9CS', 'CJX', 'CK0', 'KAN', 'JS5', 'JS6', 'LUJ',
               'KNC', 'D5E', 'XXX', '7QM', '8I5', 'RPO', '3TS', 'R7Y'
               # Tetrapyrroles
               'HEM', 'COH', 'HNI', 'VOV', '6CO', '6CQ', 'CCH', 'DDH', 'HEB', 'HIR', 'ISW', 'FDE',
               'HEV', '1FH', 'WUF', '3ZZ', 'BW9', 'CV0', 'RUR', 'ZNH', '522', 'HP5', '76R', 'HCO',
               'HFM', 'MD9', 'ZND', 'CLN', 'ZEM', 'HEC', 'BLA', 'CHL', 'CL0', 'BCB', '07D', 'CLA',
               'G2O', 'F6C', 'G9R', 'GS0', 'PHO', 'BPH', 'BPB', '08I', 'OE9', 'BPH', 'PHO', 'BPB',
               '08I', 'GS0', '0UK', 'COJ', 'P8X', '8F5', '3Y8', 'SH0', 'HT9', 'DE9', 'PP9', 'CP3',
               'MMP', '76R', 'UP3', 'RUR', 'CV0',
               # Crystallization agents
               'GOL', '0V1', 'BGQ', 'SGM', 'POL', 'LNK', 'AAE', 
               # ATP-like
               'ATP', 'ADP', 'AMP', 'NAD', 'NDP', 'ADN', 'BKP', 'GDP', 'GTP', 'GNP', 'GMV', 'VE8',
                 'G',  '0G', '0O2', 'FAD', '5GP', '8XG', 'C1Z', 'G3D', 'G4P', 'GAO', 'QBQ', 'MYA',
               'CZC', 'U3J', 'GP3', 'GZF', 'CZF', 'CX0', 'GP5', 'GAV', 'GSP', 'NAP',
               # Triterpenoids
               'J4U', 'Q7G', 'YJ0', 'YUY', 'I7Y', '0DV', 'DU0', '0V4', '9Z9', 'YUV', 'YUV', 'DU0',
               '9Z9', 'YUY', 'I7Y', 'J4U', 'Q7G', 'YJ0', '1KG', '82R', 'FFA', 'R18', 'TES', '6VW',
               '3QZ', '1CA', 'AND', 'NDR', 'ANB', 'C0R', 'PLO', 'EXM', 'K2B', '3G6', 'STR', 'DL4',
               'ASD', 'XCA', 'YNV', 'TH2', 'B81', 'AND', 'ZQK', 'ZWY', 'MKM', 'CLR', 'MHQ', '94R', 
               'LNP', 'LAN', 'VD3', 'DVE', 'HC9', 'HC3', 'CO1', 'ECK', 'C3S', 'B81', '2OB', 'Y01', 
               'CLL', '5JK', 'HCR', 'HC2', 'HCD', '0GV', 'YK8', '2DC', 'PLO', 'AND', 'ERG',
               'AE2', 'AOI', 'AOX', 'BDT', 'DHT', 'EY4', 'P9N', 'Q6J', 'A8Z', 'NQ8', 'ENM', '3KL', 
               'AOM', 'AON', 'CI2', '5SD', 'ANO', '1N7', 'CPS', 'TCH', 'GCH', '6SB', '5D5', 'TUD',
               'CHD', 'CHO', '4QM', 'BNC', 'DXC', '4OA', 'CHD', 'DXC', 'IU5', 'JN3', 'CHC', 'FX0',
               'FKC', 'FKF', '3KL', 'QC7', 'LOA', 'S5H', 'EY4', 'P9N', 'QNJ', 'LHP', 'D0O', 'CHO',
               '9L1', '7SW', '2UI', '2WV', 'CHD', 'DXC', 'IU5', 'JN3', 'CHC', 'FX0', '4OA', 'LOA',
               'S5H', 'LHP', '8JG', 'FKC', 'FKF', '3KL', 'QC7', '6SB', 'GCH', '4QM', 'BNC', 'TCH',
               'AOM', 'AON', 'QNJ', 'CHO', '5VN', '0AS', '7VX', 'B8F', '6Q5', 'CBW', 'D7S', 'DL4',
               'DL7', 'A9H', '0AS',
               # Metal clusters
               'OEC', 'OEX', 'OER', 'IV9', 'IWL', 'IWO', 'J7T', 'J8B', 'QEB',
               # Calixarenes
               'EVB', 'FWQ', 'T3Y', '6VJ', 'T8W', '6VB', 'B4T', 'B4X', 'LVT', 'B4T', 'B4X',
               # Polyoxanes
               'DIO', 'EYO', 'O4B', '16P', 'P3G', 'P4G', '12P', '1PE', '2PE', '33O', '9FO', 'M2M', 'P33', 'P6G',
               'PE3', 'PE8', 'PG4', 'PG5', 'PG6', 'PGE', 'XPE',
               # Phosphates
               '3PO', '6YW', '6YX', '6YY', '7TT', '9PI', 'DPO', 'POP', 'PPV', 'RYT',
               # Radical nitrogen
               '1WT', '4EY', '5ZV', 'AUJ', 'BU0', 'IK6', 'K6G', 'K9G','SWR', 'UGO', 'UO3', 'V22',
               'V9G', 'VFY', 'VL9', 'VLW', 'VMI',
               # Manually selected
               '2K2', 'STU', '4ST', 'UCN', 
               ]

hetatms = ['He', 'Li', 'Be', 'B', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'Ar', 
           'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Kr',
           'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'Xe',
           'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
           'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
           'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf']

int_types = ['hydrophobic', 'hbond', 'waterbridge', 'saltbridge',
             'pistacking', 'pication', 'halogen','metal']

nonLOI_list = nonLOI_list + list(standard_AAs.keys()) + list(modified_AAs.keys())

def prex(*args):
    print(*args)
    exit()

def ppex(*args):
    np.set_printoptions(threshold=sys.maxsize)
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.precision', 3):
         pprint(*args)
    exit()

def flatten(list):
    flat_list = [item for sublist in list for item in sublist]
    return(flat_list)
    
def stripNone(data):
    if isinstance(data, dict):
        return {k:stripNone(v) for k, v in data.items() if k is not None and v is not None}
    elif isinstance(data, list):
        return [stripNone(item) for item in data if item is not None]
    elif isinstance(data, tuple):
        return tuple(stripNone(item) for item in data if item is not None)
    elif isinstance(data, set):
        return {stripNone(item) for item in data if item is not None}
    else:
        return data
    
def chunks(lst, n):
    """Return successive n-sized chunks from lst."""
    chunk_list = []
    for i in range(0, len(lst), n):
        chunk_list.append(lst[i:i + n])
    return chunk_list

def read_files(files):
    if '*' in files: pdblist = [x.replace('\\','/') for x in glob.glob(files)]
    else: pdblist = files if isinstance(files, list) else [files]
    return pdblist



# import functools
# import time
# def timeit(func):
#     @wraps(func)
#     def timeit_wrapper(*args, **kwargs):
#         start_time = time.perf_counter()
#         result = func(*args, **kwargs)
#         end_time = time.perf_counter()
#         total_time = end_time - start_time
#         print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
#         return result
#     return timeit_wrapper

def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter
        
 
def diff(df1, df2, col=None):
    if col:
        df_diff = df1.loc[~df1[col].isin(df2[col])].copy()
    else:
        df_diff = df1[~df1.index.isin(df2.index)]
        # df_diff = df1[[x for x in df1.columns if x not in df2.columns]].copy()
    return df_diff


from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import py3Dmol
def drawit(m,p=None,confId=-1):
        mb = Chem.MolToMolBlock(m,confId=confId)
        if p is None:
            p = py3Dmol.view(width=400,height=400)
        p.removeAllModels()
        p.addModel(mb,'sdf')
        p.setStyle({'stick':{}})
        p.setBackgroundColor('0xeeeeee')
        p.zoomTo()
        return p.show()

def MolTo3DView(mol, size=(300, 300), style="stick", surface=False, opacity=0.5):
    """Draw molecule in 3D
    
    Args:
    ----
        mol: rdMol, molecule to show
        size: tuple(int, int), canvas size
        style: str, type of drawing molecule
               style can be 'line', 'stick', 'sphere', 'carton'
        surface, bool, display SAS
        opacity, float, opacity of surface, range 0.0-1.0
    Return:
    ----
        viewer: py3Dmol.view, a class for constructing embedded 3Dmol.js views in ipython notebooks.
    """
    assert style in ('line', 'stick', 'sphere', 'carton')
    mblock = Chem.MolToMolBlock(mol)
    viewer = py3Dmol.view(width=size[0], height=size[1])
    viewer.addModel(mblock, 'mol')
    viewer.setStyle({style:{}})
    if surface:
        viewer.addSurface(py3Dmol.SAS, {'opacity': opacity})
    viewer.zoomTo()
    return viewer
    
def SmilesToConf(smiles):
    '''Convert SMILES to rdkit.Mol with 3D coordinates'''
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        return mol
    else:
        return None

from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
def moldrawsvg(mol, molSize=(400,300), kekulize=True):
    mc = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol( mol.ToBinary() )
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0], molSize[1])
    drawer.DrawMolecule( mc )
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    return svg.replace("svg:","")