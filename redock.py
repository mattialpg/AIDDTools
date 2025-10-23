#!/usr/bin/env python3
import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

import glob, subprocess, argparse, os
from pathlib import Path
import prody
import numpy as np  

# ------------------ RECEPTOR PREPARATION ------------------
def prepare_receptor(pdb_file):
    """
    Loads the protein with ProDy, selects the protein atoms (optionally a chain),
    writes protein.pdb, and converts to protein.pdbqt with OpenBabel.
    Returns the path to protein.pdbqt in the current working directory.
    """
    pdb_file = Path(pdb_file)

    # If chain is provided, ProDy will parse only that chain.
    logger.info("Parsing protein with ProDy...")
    complex_ = prody.parsePDB(str(pdb_file), model=1, chain='A', verbosity='none')
    protein_obj = complex_.select('protein')

    # Export PDB to current folder as 'protein.pdb'
    pdb_out = Path(f"{pdb_file.stem}_protein.pdb")
    logger.info(f"Writing receptor PDB -> {pdb_out}")
    prody.writePDB(str(pdb_out), protein_obj)

    # Convert to PDBQT
    pdbqt_out = Path(f"{pdb_file.stem}_protein.pdbqt")
    logger.info(f"Converting to PDBQT with OpenBabel -> {pdbqt_out}")
    subprocess.run(
        f"obabel {pdb_out} -xr -O {pdbqt_out}",
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True)
    
    return complex_

def prepare_ligand(complex_, ligand_orig):
    """
    Converts the ligand to PDBQT format using OpenBabel.
    """
    ligand_obj = complex_.select(f"resname {ligand_orig}")

    # pdbqt_out = Path(f"{ligand_orig}.pdb")
    # prody.writePDB(str(pdbqt_out), ligand_obj)

    # Calculate docking box
    coords = ligand_obj.getCoords()
    min_coords, max_coords = np.min(coords, axis=0), np.max(coords, axis=0)
    centroid = np.round((min_coords + max_coords) / 2)
    edges = np.round(max_coords - min_coords)

    return centroid, edges


# ------------------ DOCKING ------------------
def run_docking(pdb_file, ligand_new_file, centroid, edges):
    """
    protein_pdb_file: input PDB (recommended). Will be converted to protein.pdbqt in CWD.
    ligand_file: ligand .pdbqt
    with_file: extra input (e.g., anchors) – placeholder in this minimal example.
    """
    vina = '/home/devshealth/CODE/smina.static'

    docked_file = Path(f"docked_{pdb_file.stem}.pdbqt")

    try:
        cmd = (
            f"{vina} --ligand {ligand_new_file} --receptor {pdbqt_file} --out {docked_file} "
            f"--center_x {centroid[0]} --center_y {centroid[1]} --center_z {centroid[2]} "
            f"--size_x {edges[0]} --size_y {edges[1]} --size_z {edges[2]} "
            f"--num_modes 6 --exhaustiveness 10"
        )
        subprocess.check_output(cmd, universal_newlines=True, shell=True)
        logger.info(f"Docked {docked_file}")
    except Exception as exc:
        logger.error(f"Cannot dock {docked_file}\n{exc}")

    logger.info("Docking completed")

# ------------------ MAIN ------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Redocking with AutoDock Vina")
    parser.add_argument("-protein", required=True, help="Protein PDB file (will be converted to protein.pdbqt in CWD)")
    parser.add_argument("-ligand_orig", required=True, help="Ligand file (.pdbqt)")
    parser.add_argument("-ligand_new", required=True, help="Extra input (anchors/YYY)")
    parser.add_argument("-vina", default="vina", help="Path to vina executable")
    parser.add_argument("-chain", default=None, help="Chain ID to use (equivalent to self.chain_lig)")

    args = parser.parse_args()

    # --- Prepare receptor BEFORE docking (your requested block) ---
    protein_pdb_file = Path(args.protein)
    complex_ = prepare_receptor(protein_pdb_file)

    centroid, edges = prepare_ligand(complex_, args.ligand_orig)

    subprocess.run(
        f"obabel -:'{args.ligand_new}' --gen3d -opdbqt -O ligand_new.pdbqt",
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True)

    pdbqt_file = Path(f"{protein_pdb_file.stem}_protein.pdbqt")
    ligand_new_file = Path(f"ligand_new.pdbqt")

    run_docking(pdbqt_file, ligand_new_file, centroid, edges)
