"""Based on: MacFrag is an efficient molecule fragmentation method, which is capable of segmenting
large-scale molecules in a rapid speed and generating diverse fragments that are more
compliant with the Rule of Three.

https://github.com/yydiao1025/MacFrag
https://academic.oup.com/bioinformatics/article/39/1/btad012/6986129

Expand reactions as in:
https://github.com/kheikamp/modified_molBLOCKS/blob/master/extendedRECAP.txt"""


from rdkit import Chem
from rdkit.Chem.BRICS import BreakBRICSBonds
import networkx as nx
from copy import deepcopy
from rdkit.Chem import rdDepictor


def add_atom_indices(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol

def remove_atom_indices(mol):
    # cmol = deepcopy(mol)
    for atom in mol.GetAtoms():
        atom.ClearProp('molAtomMapNumber')
    return mol
def flatten(lst):
    return [item for sublist in lst for item in sublist]

def replace_dummies(mol, replace_with='H', keep_info=True):
    mol = deepcopy(mol)
    rwMol = Chem.RWMol(mol)

    if keep_info:
        for atom in mol.GetAtoms():
            atom.SetProp('_symbol', atom.GetSymbol())

    dummies = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == '*']

    if replace_with == 'H':
        for dummy in sorted(dummies, reverse=True):
            rwMol.ReplaceAtom(dummy, Chem.Atom(1))
    else:
        h = Chem.GetPeriodicTable().GetAtomicNumber(replace_with)
        try:
            for dummy in sorted(dummies, reverse=True):
                rwMol.GetAtomWithIdx(dummy).SetAtomicNum(h)
            rwMol = Chem.RemoveHs(rwMol)
            Chem.SanitizeMol(rwMol)
        except Chem.AtomValenceException:
            # Force replacement with single bond
            for dummy in sorted(dummies, reverse=True):
                bond = rwMol.GetAtomWithIdx(dummy).GetBonds()[0]
                bond.SetBondType(Chem.BondType.SINGLE)
                rwMol.GetAtomWithIdx(dummy).SetAtomicNum(h)
            rwMol = Chem.RemoveHs(rwMol)
            Chem.SanitizeMol(rwMol)

    return Chem.Mol(rwMol)

def get_all_connected_subgraphs(G):
    """Get all connected subgraphs by a recursive procedure"""
    
    def recursive_local_expand(node_set, possible, excluded, results, max_size):
        """Recursive function to add an extra node to the subgraph being formed"""
        results.append(node_set)
        if len(node_set) == max_size:
            return
        for j in possible - excluded:
            new_node_set = node_set | {j}
            excluded = excluded | {j}
            new_possible = (possible | set(G.neighbors(j))) - excluded
            recursive_local_expand(new_node_set, new_possible, excluded, results, max_size)

    # Iterate over connected components
    results = []
    for cc in sorted(nx.connected_components(G), key=len, reverse=True):
        max_size = len(cc)
        # Iterate over nodes in the connected component
        for i in cc:
            recursive_local_expand({i}, set(G.neighbors(i)), {i}, results, max_size)

    return [tuple(x) for x in sorted(results, key=len)]


def join_sets_with_common_elements(set_list):
    set_list = [set(sorted(x[0] + x[1])) for x in set_list]
    updated = True
    while updated:
        updated = False
        for i in range(len(set_list)):
            for j in range(i + 1, len(set_list)):
                if set_list[i] & set_list[j]:
                    # Join sets with common elements
                    set_list[i] |= set_list[j]
                    set_list.pop(j)
                    updated = True
                    break
    return [tuple(x) for x in set_list]

def break_bonds(mol, bonds):
    eMol = Chem.EditableMol(mol)
    for bond in bonds:
        bond_order = mol.GetBondBetweenAtoms(bond[0], bond[1]).GetBondType()
        eMol.RemoveBond(bond[0], bond[1])

        # Add a dummy atoms to replace the vacant positions
        dummy_atom = Chem.Atom(0)   # 0 represents a dummy atom
        dummy_atom.SetAtomicNum(0)  # Set atomic number to 0 for a dummy atom
        dummy_atom_idx = eMol.AddAtom(dummy_atom)
        eMol.AddBond(bond[0], dummy_atom_idx, bond_order)

        dummy_atom = Chem.Atom(0)   # 0 represents a dummy atom
        dummy_atom.SetAtomicNum(0)  # Set atomic number to 0 for a dummy atom
        dummy_atom_idx = eMol.AddAtom(dummy_atom)
        eMol.AddBond(bond[1], dummy_atom_idx, bond_order)

    res = eMol.GetMol()
    Chem.SanitizeMol(res)
    return res


def GraphDecomp(mol, maxBlocks=8, keep_AtomMapNumber=True, side_chain_size=4):
    """
    Parameters:
        maxBlocks: The maximum number of building blocks that the fragments contain
        maxSR: rings containing a number of atoms equal to or less than this value will not be fragmented
    """
    mol = add_atom_indices(mol)

    rings = mol.GetRingInfo().AtomRings()
    bond_matrix = Chem.GetAdjacencyMatrix(mol)

    # Find pairs of fused rings
    pairs_fused_rings = []
    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            if set(rings[i]) & set(rings[j]):
                pairs_fused_rings.append((rings[i], rings[j]))

    unfused_rings = [x for x in rings if x not in flatten(pairs_fused_rings)]
    fused_rings = join_sets_with_common_elements(pairs_fused_rings)
    new_rings = unfused_rings + fused_rings

    # Find bonds between new rings
    bonds = set()
    for ring in new_rings:
        for atom1 in ring:
            for atom2 in range(len(mol.GetAtoms())):
                if bond_matrix[atom1, atom2] == 1 and atom2 not in ring:
                    bonds.add(tuple(sorted((atom1, atom2))))
    
    # Do not break C==O bonds
    acylic = []
    for bond in bonds:
        a1 = mol.GetAtomWithIdx(bond[0]).GetSymbol()
        a2 = mol.GetAtomWithIdx(bond[1]).GetSymbol()
        b = mol.GetBondBetweenAtoms(bond[0], bond[1]).GetBondType()
        if a1 == 'C' and a2 == 'O' and b == Chem.rdchem.BondType.DOUBLE:
            acylic.append(bond)
    bonds = [b for b in bonds if b not in acylic]

    #? The following code gives different results with CCIs 9B7 and 6A4 
    # # Check presence of the pattern P-O-C 
    # for atom in mol.GetAtoms():
    #     if atom.GetSymbol() == 'P':
    #         for neigh1 in atom.GetNeighbors():
    #             if neigh1.GetSymbol() == 'O':
    #                 PO_pair = tuple(sorted((atom.GetIdx(), neigh1.GetIdx())))
    #                 if PO_pair in bonds:
    #                     bonds.remove(PO_pair)  # Remove P-O from bonds to keep intact the phosphates
    #                 for neigh2 in neigh1.GetNeighbors():
    #                     if neigh2.GetSymbol() == 'C':
    #                         CO_pair = tuple(sorted((neigh1.GetIdx(), neigh2.GetIdx())))
    #                         bonds.add(CO_pair)  # Add (P-O)-C bonds

    # Get molecular fragments (blocks)
    blocks = list(Chem.GetMolFrags(break_bonds(mol, bonds), asMols=True))

    for block in blocks:
        block_smi = Chem.MolToSmiles(block)
        nAtoms = block.GetNumAtoms(onlyExplicit=True) - block_smi.count('*')
        nRings = [list(x) for x in block.GetRingInfo().AtomRings()]
        nAtomsRings = [len(x) for x in nRings]

        # Classify blocks based on structural type  #TODO: add Decorator type
        if nAtomsRings:
            if nAtomsRings == [3] or nAtomsRings == [4]: type = 'T'  # Small rings (cyclopropane and cyclobutane)
            else: type = 'R'  # Rings
        elif 'P' in block_smi: type = 'P'  # Phosphate blocks
        elif block_smi.count('*') > 1: type = 'L'  # Linkers
        elif block_smi.count('*') == 1 and nAtoms >= side_chain_size: type = 'S'  # Side-chains
        else: type = 'X'  # Substituents

        # Assign block properties 
        block.SetProp('Type', type)
        block.SetProp('Smiles', block_smi)

    #! No recomposition
    if maxBlocks == 1 or not mol.GetRingInfo().AtomRings():  # Skip molecules with no rings
        if not keep_AtomMapNumber:
            [remove_atom_indices(block) for block in blocks]
        return blocks
    elif maxBlocks > len(blocks):
        maxBlocks = len(blocks)

    #! Graph-based implementation
    # Create a dictionary of {block: index}
    block_index = {tuple([a.GetAtomMapNum() for a in block.GetAtoms() if a.GetSymbol()!='*']): i
                   for i, block in enumerate(blocks)}

    # Create a dictionary of {bond: block}
    bond_block = {}
    for bond in bonds:
        ba1, ba2 = bond
        for block in block_index.keys():
            if ba1 in block:
                bond_block[ba1] = block
            if ba2 in block:
                bond_block[ba2] = block
    
    G = nx.Graph()
    # Add nodes with attributes
    for i, block in enumerate(blocks):
        G.add_node(i, type=block.GetProp('Type'))
    # Add edges with attributes
    for bond in bonds:
        ba1, ba2 = sorted(bond)
        node1, node2 = block_index[bond_block[ba1]], block_index[bond_block[ba2]]
        G.add_edge(node1, node2, bond=bond)

    # pos_nodes = nx.spring_layout(G)
    # pos_attrs = {}
    # for node, coords in pos_nodes.items():
    #     pos_attrs[node] = (coords[0], coords[1]+.09)
    # nx.draw_networkx(G)
    # import matplotlib.pyplot as plt
    # plt.show()

    # Get all connected induced subgraphs (CISs) and build a type dictionary
    all_cis = get_all_connected_subgraphs(G)
    all_cis = [x for x in all_cis if len(x) <= maxBlocks]
    node_attrs = nx.get_node_attributes(G, 'type')
    all_cis_dict = {cis: ''.join(map(node_attrs.get, cis)) for cis in all_cis}
    all_cis_dict = {tuple(sorted(k)): ''.join(sorted(v)) for k, v in all_cis_dict.items()}

    # Drop disallowed type combinations
    for cis, types in list(all_cis_dict.items()):
        if types.count('R') >= 2: del all_cis_dict[cis]            # Rings combinations
        elif 'R' in types and 'S' in types: del all_cis_dict[cis]  # Ring/side-chain combinations
        elif 'R' in types and 'L' in types: del all_cis_dict[cis]  # Ring/linker combinations
        elif 'P' in types and types != 'P': del all_cis_dict[cis]  # Phosphates joined with anything
        elif 'T' in types and types != 'T': del all_cis_dict[cis]  # Small-rings joined with anything

    # from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers
    # for block in blocks:
    #     # for bond in block.GetBonds():
    #     #     print(bond.GetStereo())
    #     print(list(EnumerateStereoisomers(block)))

    # Keep unique combinations of fragments
    keep_cis_dict = {}
    nodes_to_add = list(range(len(blocks)))  # All the original single fragments
    for cis, types in list(all_cis_dict.items())[::-1]:  # Start from longest CIS
        if all(n in nodes_to_add for n in cis):
            keep_cis_dict[cis] = ''.join(x for x in types)
            nodes_to_add = [x for x in nodes_to_add if x not in cis]
    keep_cis_dict = dict(sorted(keep_cis_dict.items()))

    # Get edges between CISs to keep
    edges_to_break = list(G.edges())
    for edge in G.edges():
        for unseparable_nodes in keep_cis_dict.keys():
            if set(edge).issubset(unseparable_nodes):
                edges_to_break.remove(edge)

    if not edges_to_break:
        mol.SetProp('Type', 'U')  # Unique fragment
        if not keep_AtomMapNumber:
            remove_atom_indices(mol)
        return [mol]

    # Convert breakable edges to bonds
    bonds_to_break = [G[e[0]][e[1]]['bond'] for e in edges_to_break]
    new_blocks = list(Chem.GetMolFrags(break_bonds(mol, bonds_to_break), asMols=True))

    #NOTE: The fragments created so far retain the original indices (idx)
    #> of the parent mol. The dummy atoms are added to the broken molecule
    #> and for this reason their indices follows those of the heavy atoms.
    #> This unordered indexing may cause problems, especially when mapping
    #> atom numbers between different mols. For this reason, it is safer
    #> to canonicalize the smiles an reconstrunct the fragment from it.  
    copy_blocks = [remove_atom_indices(deepcopy(x)) for x in new_blocks]
    copy_blocks = [Chem.MolFromSmiles(Chem.MolToSmiles(x)) for x in copy_blocks]

    # Project the original map num. on the newly indexed blocks
    atom_map_numbers = [[a.GetAtomMapNum() for a in block.GetAtoms()] for block in new_blocks]
    matches = [p.GetSubstructMatch(q) for p, q in zip(new_blocks, copy_blocks)]
    for block, match, mapnum in zip(copy_blocks, matches, atom_map_numbers):
        for i, m in enumerate(match):
            atom = block.GetAtomWithIdx(i)
            atom.SetAtomMapNum(mapnum[m])

    # Add fragment type to blocks
    for block, type in zip(copy_blocks, keep_cis_dict.values()):
        block.SetProp('Type', type)
        # Reset coordinates for display
        rdDepictor.Compute2DCoords(block)
        rdDepictor.StraightenDepiction(block)
    
    # Remove original ligand atom numbering
    if not keep_AtomMapNumber:
        for block in copy_blocks:
            remove_atom_indices(block)
        remove_atom_indices(mol)

    return copy_blocks


def remove_NIG(query_mol, interacting_atoms, frag_type=None):
    """ NIG = Non-Interacting Groups """
    try:
        if not frag_type or frag_type == 'R':
            # Fragment molecule
            query_mol = deepcopy(query_mol)
            frags = GraphDecomp(replace_dummies(query_mol), side_chain_size=1)
            frag_dict = {tuple([a.GetAtomMapNum() for a in frag.GetAtoms() if a.GetSymbol() != '*']):
                        frag.GetProp('Type') for frag in frags}
            
            if not frags:
                return query_mol

            side_chain_atoms = [x for x in frag_dict.keys() if frag_dict[x] != 'R']
            atoms_to_remove = flatten([x for x in side_chain_atoms if not any(i in interacting_atoms for i in x)])
            atoms_to_remove = [x for x in atoms_to_remove if query_mol.GetAtomWithIdx(x).GetSymbol() != '*']

            rwMol = Chem.RWMol(query_mol)
            for idx in sorted(atoms_to_remove, reverse=True):
                neighs = rwMol.GetAtomWithIdx(idx).GetNeighbors()
                rwMol.RemoveAtom(idx)
                # Fix hydrogen for neigbors atoms
                for neigh in neighs:
                    if neigh.GetIdx() not in atoms_to_remove:
                        neigh.SetNumExplicitHs(1)
            
            rwMol = Chem.RemoveHs(rwMol)
            Chem.SanitizeMol(rwMol)
            Chem.rdmolops.SanitizeFlags.SANITIZE_NONE
            return Chem.Mol(rwMol)
    
        elif frag_type == 'L':
            # Keep whole interacting groups
            atoms_to_reduce = [a.GetIdx() for a in query_mol.GetAtoms()
                            if a.GetSymbol() != '*'
                            and a.GetIdx() not in interacting_atoms]
            matches = query_mol.GetSubstructMatches(Chem.MolFromSmarts("O=S=O")) + \
                      query_mol.GetSubstructMatches(Chem.MolFromSmarts("C(=O)O")) + \
                      query_mol.GetSubstructMatches(Chem.MolFromSmarts("C(F)(F)F"))
            for match in matches:
                if any(idx in interacting_atoms for idx in match):
                    atoms_to_reduce = [idx for idx in atoms_to_reduce if idx not in match]
            # print(atoms_to_reduce)

            rwMol = Chem.RWMol(query_mol)

            atoms_to_remove = []
            for idx in atoms_to_reduce:
                atom = rwMol.GetAtomWithIdx(idx)
                matches = rwMol.GetSubstructMatches(Chem.MolFromSmarts('[O;$(O=[C,S])]')) + \
                          rwMol.GetSubstructMatches(Chem.MolFromSmarts('[N;$(N#C)]')) + \
                          rwMol.GetSubstructMatches(Chem.MolFromSmarts('[F;$(F-C)]'))
                          # rwMol.GetSubstructMatches(Chem.MolFromSmarts('[OH;$(OC(=O))]')) + 
                if atom.GetIdx() in flatten(matches):
                    atoms_to_remove.append(atom.GetIdx())
                else:
                    atom.SetAtomicNum(6)

            for idx in sorted(atoms_to_remove, reverse=True):
                rwMol.RemoveBond(idx, rwMol.GetAtomWithIdx(idx).GetNeighbors()[0].GetIdx())
                rwMol.RemoveAtom(idx)

            return rwMol.GetMol()

    except Exception as e:
        return query_mol