import warnings

warnings.simplefilter('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os, sys, shutil
from pathlib import Path
import lxml.etree as ET
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml
from concurrent.futures import ProcessPoolExecutor

sys.path.append(str(Path(__file__).resolve().parent))
# with warnings.catch_warnings():
from pymol import cmd
from plip_modded.structure.preparation import PDBComplex
from plip_modded.exchange.report import BindingSiteReport
from plip_modded.basic import config
from plip_modded.basic.remote import VisualizerData
from plip_modded.visualization.visualize import visualize_in_pymol


import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)
# Silence plip logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.WARNING)


def xml_to_df(xml_data, pdb_id=None, ligand=None, site_id=None):
    """
    Parse one PLIP BindingSiteReport XML into a tidy DataFrame.
    """
    int_type_map = {
        "hydrophobic_interactions": "hydrophobic",
        "hydrogen_bonds": "hbond",
        "water_bridges": "waterbridge",
        "salt_bridges": "saltbridge",
        "pi_stacks": "pistacking",
        "pi_cation_interactions": "pication",
        "halogen_bonds": "halogen",
        "metal_complexes": "metal"}
    
    # Try to detect whether it's a file path or XML content
    if os.path.exists(xml_data):
        tree = ET.parse(xml_data)
        root = tree.getroot()
    else:
        # Assume it's an XML string
        root = ET.fromstring(xml_data)

    interactions_el = root.find("interactions")

    rows = []
    for category in interactions_el:
        tag = category.tag  # e.g. "hydrogen_bonds"
        mapped_type = int_type_map.get(tag, tag)  # map to your custom list

        for interaction in category:
            data = {"INT_TYPE": mapped_type}
            for elem in interaction:
                if len(elem) == 0:
                    # simple leaf
                    data[elem.tag.upper()] = elem.text
                elif elem.tag.endswith("coo"):
                    # coordinates (x,y,z)
                    for c in elem:
                        data[f"{elem.tag.upper()}_{c.tag.upper()}"] = c.text
                elif elem.tag.endswith("list"):
                    # list of indices
                    ids = [idx.text for idx in elem.findall("idx")]
                    data[elem.tag.upper()] = ",".join(ids)
                else:
                    data[elem.tag.upper()] = elem.text

            # add metadata
            if site_id:
                data["SITE_ID"] = site_id
            if pdb_id:
                data["PDB_ID"] = pdb_id
            if ligand:
                data["LIGAND"] = ligand

            rows.append(data)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def generate_pymol(complex, ligand, pdb_id, site, pdb_file,
                   pdb_outdir=None, pse_outdir=None):
    """
    Generate PyMOL session or PDB binding site file for a specific ligand binding site.
    """

    residues = complex.interaction_sets[site].interacting_res
    residues += [f"{ligand.position}{ligand.chain}"]
    basename = f"{pdb_id}_{site.replace(':', '-')}"

    # Save PyMOL session (.pse)
    if pse_outdir:
        os.makedirs(pse_outdir, exist_ok=True)
        vdata = VisualizerData(complex, site)
        visualize_in_pymol(vdata, residues_to_keep=residues)
        pse_path = os.path.join(pse_outdir, f"{basename}.pdb")
        shutil.move(f"{pdb_id.lower()}.pdb", pse_path)
        # logger.info(f"Saved PyMOL session {basename}.pdb")

    # Save binding site PDB
    if pdb_outdir:
        os.makedirs(pdb_outdir, exist_ok=True)
        outpath = os.path.join(pdb_outdir, f"{basename}.pdb")
        selection = " or ".join([f"(chain {r[-1]} and resi {r[:-1]})" for r in residues])
        print(residues)
        cmd.load(str(pdb_file), pdb_id)
        com = cmd.centerofmass(selection)
        cmd.translate([-com[0], -com[1], -com[2]], "all")
        cmd.save(outpath, selection)
        cmd.delete("all")
        # print(f"Saved binding site PDB {basename}.pdb")


def get_interactions(pdb_file, lig_id=None, xml_outdir=None,
                     pdb_outdir=None, pse_outdir=None):
    """
    Analyse interactions for a given PDB ID using PLIP.
    With lig_id = None, process all ligands in the structure.
    Optionally generates PyMOL session or PDB binding site files.
    """
    xml_data = {}
    try:
        pdb_id = Path(pdb_file).stem.upper()

        complex = PDBComplex()
        complex.load_pdb(pdb_file)

        # Select ligands to analyse
        ligands_to_analyse = ([lig for lig in complex.ligands if lig.hetid == lig_id]
            if lig_id else complex.ligands)
        if not ligands_to_analyse:
            # logger.warning(f"No ligand found matching {lig_id} in {pdb_id}")
            return {}
        
        for ligand in ligands_to_analyse:
            complex.interaction_sets = {}
            complex.characterize_complex(ligand)

            for plip_obj in complex.interaction_sets.values():
                binding_site = BindingSiteReport(plip_obj)
                xml_element = binding_site.generate_xml()
                xml_string = ET.tostring(xml_element, encoding="unicode")

                basename = f"{pdb_id}-{ligand.hetid}-{ligand.position}-{ligand.chain}"
                xml_data[basename] = xml_string

                if xml_outdir:
                    os.makedirs(xml_outdir, exist_ok=True)
                    xml_path = os.path.join(xml_outdir, f"{basename}.xml")
                    with open(xml_path, "w") as xml_file:
                        xml_file.write(xml_string   )
                    # logger.info(f"Saved PLIP XML for {basename}")

            # Generate PyMOL visualisation if requested
            if pdb_outdir or pse_outdir:
                config.PYMOL = True
                for site in sorted(complex.interaction_sets):
                    if complex.interaction_sets[site].interacting_res:
                        generate_pymol(complex, ligand, pdb_id, site, pdb_file,
                            pdb_outdir=pdb_outdir, pse_outdir=pse_outdir)

        return xml_data

    except Exception as exc:
        # logger.error(f"Error analysing {pdb_id}: {exc}")
        return {}


def read_interaction_files(indir):
    dfs = []
    for file in Path(indir).glob("*.xml"):
        df_tmp = xml_to_df(str(file))
        df_tmp.insert(0, "PDB", file.stem.split("_")[0])
        dfs.append(df_tmp)
    return pd.concat(dfs, ignore_index=True)


def harmonise_interactions(df_int):
    # Fill null values in one column with non-null values from another
    try: df_int['DONORIDX'] = df_int['DONORIDX'].combine_first(df_int['DON_IDX'])
    except: pass
    try: df_int['DONORIDX'] = df_int['DONORIDX'].combine_first(df_int['DONOR_IDX'])
    except: pass
    try: df_int['ACCEPTORIDX'] = df_int['ACCEPTORIDX'].combine_first(df_int['ACC_IDX'])
    except: pass
    try: df_int['ACCEPTORIDX'] = df_int['ACCEPTORIDX'].combine_first(df_int['ACCEPTOR_IDX'])
    except: pass

    # Use cosine law to get receptor-ligand distance for waterbridge interactions
    try:
        dist_aw = pd.to_numeric(df_int.get("DIST_A-W"), errors="coerce")
        dist_dw = pd.to_numeric(df_int.get("DIST_D-W"), errors="coerce")
        angle = pd.to_numeric(df_int.get("WATER_ANGLE"), errors="coerce")
        D = np.sqrt(np.square(dist_aw) + np.square(dist_dw) -
            2 * dist_aw * dist_dw * np.cos(np.deg2rad(angle))).round(2)
        df_int['DIST'] = df_int['DIST'].combine_first(D)
    except: pass
    try: df_int['DIST'] = df_int['DIST'].combine_first(df_int['DIST_D-A'])
    except: pass
    try: df_int['DIST'] = df_int['DIST'].combine_first(df_int['CENTDIST'])
    except: pass

    # Extract parameters based on interaction type
    protatomidx, ligatomidx = [], []
    for row in df_int.itertuples():
        if row.INT_TYPE in ('hbond', 'waterbridge'):
            if str(row.PROTISDON).upper() == "TRUE":
                protatomidx.append([row.DONORIDX])
                ligatomidx.append([row.ACCEPTORIDX])
            else:
                protatomidx.append([row.ACCEPTORIDX])
                ligatomidx.append([row.DONORIDX])
        elif row.INT_TYPE == 'hydrophobic':
            protatomidx.append([row.PROTCARBONIDX])
            ligatomidx.append([row.LIGCARBONIDX])
        elif row.INT_TYPE in ('pication', 'saltbridge', 'pistacking'):
            protatomidx.append(row.PROT_IDX_LIST.split(',') if isinstance(row.PROT_IDX_LIST, str) else [row.PROT_IDX_LIST])
            ligatomidx.append(row.LIG_IDX_LIST.split(',') if isinstance(row.LIG_IDX_LIST, str) else [row.LIG_IDX_LIST])
        elif row.INT_TYPE == 'halogen':
            protatomidx.append(row.ACC_IDX.split(',') if isinstance(row.ACC_IDX, str) else [row.ACC_IDX])
            ligatomidx.append(row.DON_IDX.split(',') if isinstance(row.DON_IDX, str) else [row.DON_IDX])
        elif row.INT_TYPE == 'metal':
            protatomidx.append(row.TARGET_IDX.split(',') if isinstance(row.TARGET_IDX, str) else [row.TARGET_IDX])
            ligatomidx.append(row.METAL_IDX.split(',') if isinstance(row.METAL_IDX, str) else [row.METAL_IDX])
        else:
            # Fallback to ensure same length
            protatomidx.append(None)
            ligatomidx.append(None)

    df_int['LIGATOMIDX'] = ligatomidx
    df_int['PROTATOMIDX'] = protatomidx

    col_map = {"PDB": "PDB", "RESNR": "RESNUM",
        "RESTYPE": "RESNAME", "RESCHAIN": "RESCHAIN",
        "RESNR_LIG": "LIGNUM", "RESTYPE_LIG": "LIGNAME",
        "RESCHAIN_LIG": "LIGCHAIN", "PROTATOMIDX": "PROTATOMIDX",
        "LIGATOMIDX": "LIGATOMIDX", "INT_TYPE": "INT_TYPE", "DIST": "DIST"}
    df_int = df_int[[k for k in col_map.keys()]].rename(columns=col_map)

    # Final check for missing values
    if df_int.isna().any().any():
        raise ValueError("something went wrong in harmonisation")

    return df_int


def analyse_pdb_files(pdb_files):
   
    with ProcessPoolExecutor() as executor:
        xml_results = list(tqdm(executor.map(get_interactions, pdb_files),
                total=len(pdb_files), desc="Analysing interactions"))

    df_list = []
    for xml_data in xml_results:
        for name, tree in xml_data.items():
            df = xml_to_df(tree)
            df['PDB'] = name.split(':')[0]
            df_list.append(df)
    df_int = pd.concat(df_list, ignore_index=True)

    df_int = harmonise_interactions(df_int)
    df_int["RESID"] = df_int["RESNAME"] + df_int["RESNUM"]
    df_int = df_int[["PDB", "RESID", "LIGNAME", "INT_TYPE"]].fillna("NA")
    return df_int


def get_ligands(pdb_file, select_loi=False):
    module_dir = Path(__file__).resolve().parent
    biolip_path = module_dir / "biolip_list.yaml"
    with open(biolip_path) as f:
        biolip_data = yaml.safe_load(f)
        biolip_list = biolip_data.get("biolip_list", biolip_data)

    # Parse the structure with PLIP
    complex_mol = PDBComplex()
    complex_mol.load_pdb(pdb_file)

    # Collect ligands detected by PLIP
    ligands = {}
    for lig in complex_mol.ligands:
        lig_id = f"{lig.hetid}:{lig.chain}:{lig.position}"
        loi = "Non-LOI" if lig.hetid in biolip_list else "LOI"
        ligands[lig_id] = loi
    
    if select_loi:
        ligands = {k: v for k, v in ligands.items() if v == "LOI"}

    return ligands