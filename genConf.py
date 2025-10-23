from rdkit import Chem
from rdkit.Chem import AllChem


def generate_conformers(mol, num_confs=None, prune_rms_thresh=None, energy_filter=False, post_rms_thresh=None):
    """
    Generate conformers for a molecule with pruning, energy filtering, and optional post-RMSD pruning.
    """
    num_rotatable_bonds = int(AllChem.CalcNumRotatableBonds(mol))
    mol = Chem.AddHs(mol)
    Chem.AssignAtomChiralTagsFromStructure(mol, replaceExistingTags=True)

    # Determine conformer count if automatic
    if not num_confs:
        if num_rotatable_bonds < 3:
            num_confs = 50
        elif num_rotatable_bonds > 6:
            num_confs = 300
        else:
            num_confs = num_rotatable_bonds ** 3

    # Generate conformers
    conformer_ids = AllChem.EmbedMultipleConfs(
        mol,
        numConfs=num_confs,
        pruneRmsThresh=prune_rms_thresh,
        randomSeed=1,
        useExpTorsionAnglePrefs=True,
        useBasicKnowledge=True
    )

    # Energy minimization and evaluation
    conformer_energies = []
    for conf_id in conformer_ids:
        props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
        if props is None:
            continue
        ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
        ff.Minimize()
        energy = float(ff.CalcEnergy())
        conformer_energies.append((energy, conf_id))

    # Energy filtering
    if energy_filter:
        mol, conformer_energies = filter_by_energy(mol, conformer_energies, energy_filter)

    # Post-RMSD filtering
    if post_rms_thresh:
        mol, conformer_energies = post_rmsd_filter(mol, conformer_energies, post_rms_thresh)

    return mol, conformer_energies, num_rotatable_bonds


def filter_by_energy(mol, conformer_energies, energy_window):
    """
    Keep conformers within an energy window (kcal/mol).
    """
    if not conformer_energies:
        return mol, []

    conformer_energies.sort(key=lambda x: x[0])
    min_energy = float(conformer_energies[0][0])
    energy_cutoff = min_energy + energy_window

    filtered_mol = Chem.Mol(mol)
    filtered_mol.RemoveAllConformers()
    filtered_mol.AddConformer(mol.GetConformer(int(conformer_energies[0][1])))

    kept_ids = [int(conformer_energies[0][1])]
    kept_energies = [float(conformer_energies[0][0])]

    for energy, conf_id in conformer_energies[1:]:
        if energy <= energy_cutoff:
            filtered_mol.AddConformer(mol.GetConformer(int(conf_id)))
            kept_ids.append(int(conf_id))
            kept_energies.append(float(energy))
        else:
            break

    return filtered_mol, list(zip(kept_energies, kept_ids))


def post_rmsd_filter(mol, conformer_energies, post_rms_thresh):
    """
    Remove conformers that are too similar (RMSD pruning after minimization).
    """
    if not conformer_energies:
        return mol, []

    conformer_energies.sort(key=lambda x: x[0])
    filtered_mol = Chem.Mol(mol)
    filtered_mol.RemoveAllConformers()

    kept_conf_ids = [conformer_energies[0][1]]
    kept_energies = [conformer_energies[0][0]]

    mol_no_h = Chem.RemoveHs(mol)

    for energy, conf_id in conformer_energies[1:]:
        keep_conf = True
        for kept_id in kept_conf_ids:
            rmsd = AllChem.GetBestRMS(mol_no_h, mol_no_h, probeConfId=conf_id, refConfId=kept_id)
            if rmsd < post_rms_thresh:
                keep_conf = False
                break
        if keep_conf:
            kept_conf_ids.append(conf_id)
            kept_energies.append(float(energy))

    for conf_id in kept_conf_ids:
        filtered_mol.AddConformer(mol.GetConformer(conf_id))

    return filtered_mol, list(zip(kept_energies, kept_conf_ids))
