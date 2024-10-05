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

# SMARTS atomic environments for molecule fragmentation
environDefs = {
    'L1':  '[C;D3]([#0,#6,#7,#8])(=O)',  # Acyl –C(=O)– pattern
    'L2':  '[O;D2]-[#0,#6,#1]',  # Ether –O– pattern | originally L3
    'L3':  '[C;!D1;!$(C=*)]-[#6]',  # Aliphatic –C– pattern | originally L4
    'L31': '[C;D2]-[#6]',  # Aliphatic =C– pattern broken at the double bond

    'L4':  '[N;!D1;!$(N=*);!$(N-[!#6;!#16;!#0;!#1]);!$([N;R]@[C;R]=O)]',  # originally L5
    'L41': '[N;$([N;!R]=[#6])]',  # Aliphatic =N– pattern broken at the double bond
    'L8':  '[N;R;$(N(@C(=O))@[#6,#7,#8,#16])]',  # Cyclic amide pattern | originally L10

    'L5':  '[C;D2,D3]-C',  # –C=C– pattern broken in the middle | Originally L7
    'L51': 'C=C',  # –C=C– pattern broken at sides

    'L6':  '[C;!D1;!R;!$(C!-*)]',  #original L8
    'L61': '[C;R1;!D1;!$(C!-*)]',
    'L62': '[#6]=[N,S]-[#6,#7,#8,#16]',  # Aliphatic C=[N,S]–[C,N,S] pattern  !COMPARE WITH L113

    'L9':  '[S;D2](-[#0,#6])',  # Thioether –S– pattern | originally L11
    'L10': '[S;D4]([#0,#6])(=O)(=O)',  # Sulfonyl [*,C]–SO2–[*,C] pattern | originally L12

    'L14': '[c;$(c(:c):c)]', # Aromatic C–C–C pattern | originally L16
    'L12': '[c;$(c(:[c,n,o,s]):[n,o,s])]',  # Aromatic [C,N,O,S]–C–[N,O,S] pattern | originally L14
    'L7':  '[n;$(n(:[c,n,o,s]):[c,n,o,s])]',  # Aromatic [C,N,O,S]–N–[C,N,O,S] pattern | originally L9 | removed charge constraint
    
    'L11':  '[C;$(C(-;@[C,N,O,S])-;@[N,O,S])]',  # Aliphatic ring [C,N,O,S]–C–[N,O,S] pattern | originally L13
    'L111': '[C;R2;$(C(-;@[C,N,O,S])-;@[N,O,S])]',
    'L112': '[C;R1;$(C(-;@[C,N,O,S;R2])-;@[N,O,S;R2])]',
    'L113': '[C;$(C(-;@[C,N,O,S])=;@[N,O,S])]',  # Aliphatic ring [C,N,O,S]-C=[N,O,S] pattern
    'L114': '[#6]-[#6]=[#5]',  # Aliphatic ring [C,N,O,S]-C=[N,O,S] pattern
    
    'L13':  '[C;$(C(-;@[C,N])-;@C)]',  # Aliphatic ring C–[C,N]–C pattern | originally L15 | added central N
    'L131': '[C;R2;$(C(-;@C)-;@C)]',
    'L132': '[C;R1;$(C(-;@[C;R2])-;@[C;R2])]',
    'L133': '[C;$([#6](-;@C)=;@C)]',  # Aliphatic ring C–C=C pattern
    
    # Custom linkers
    'L15': '[#5,#6]C#C[#5,#6]',
    'L151': 'C#N',
    'L16': 'N=N',
    'L17': 'P(=O)(O)O',
    # 'L19': 'CC=N',
}

reactionDefs = (
    # L1
    [('1', '2', '-'),
     ('1', '4', '-'),
     ('1', '41', '-'),
     ('1', '8', '-'),
     ('1', '11', '-'),
     ('1', '12', '-'),
     ('1', '13', '-'),
     ('1', '14', '-')],

    # L2
    [('2', '3', '-'),
     ('2', '11', '-'),
     ('2', '12', '-'),
     ('2', '13', '-'),
     ('2', '14', '-')],

    # L3
    [('3', '4', '-'),
     ('3', '7', '-'),
     ('3', '9', '-'),
     ('3', '10', '-'),
     ('3', '11', '-'),
     ('31', '11', '='),
     ('3', '133', '-'),
     ('3', '14', '-')],

    # L4
    [('4', '5', '-'),
     ('4', '10', '-'),
     ('4', '12', '-'),
     ('4', '14', '-'),
     ('4', '11', '-'),
     ('4', '113', '-'),
     ('4', '13', '-'),
     ('41', '12', '-'),
    #  ('41', '12', '='),
     ('41', '13', '=')],
    
    # L5
    [('5', '11', '='),
    ('5', '13', '='),
    ('51', '4', '-'),
    ('51', '12', '-'),
    ('51', '13', '-'),
    ('51', '133', '-'),
    ('51', '14', '-')],

    # L6
    [('6', '7', '-'),
     ('6', '8', '-'),
     ('6', '11', '-;!@'),
     ('6', '12', '-'),
     ('6', '13', '-;!@'),
     ('6', '14', '-'),
     ('61', '111', '-;@'),
     ('61', '131', '-;@'),
     ('62', '113', '-'),
     ('62', '12', '-'),
     ('62', '14', '~')],

    # L7
    [('7', '11', '-'),
     ('7', '12', '-'),
     ('7', '13', '-'),
     ('7', '14', '-')],

    # L8
    [('8', '11', '-'),
     ('8', '12', '-'),
     ('8', '13', '-'),
     ('8', '14', '-')],

    # L9
    [('9', '11', '-'),
     ('9', '12', '-'),
     ('9', '13', '-'),
     ('9', '113', '-'),
     ('9', '133', '-'),
     ('9', '14', '-')],

    # L10
    [('10', '12', '-'),
     ('10', '133', '-'),
     ('10', '14', '-')],
     
    # L11
    [('11', '12', '-'),
     ('11', '13', '-;!@'),
     ('11', '14', '-'),
     ('112', '132', '-;@'),
     ('113', '113', '-'),
     ('113', '12', '-'),
     ('113', '13', '-'),
     ('113', '133', '-'),
     ('113', '14', '-')],

    # L12
    [('12', '12', '-'),
     ('12', '13', '-'),
     ('12', '131', '-'),
     ('12', '132', '-'),
     ('12', '133', '-'),
     ('12', '14', '-')],

    # L13
    [('13', '14', '-')],

    # L14
    [('12', '14', '-'),
    ('14', '14', '-'),
    # ('14', '19', '-')
    ],

    # L15
    [('15', '3', '-'),
    ('15', '5', '-'),
    ('15', '12', '-'),
    ('15', '13', '-'),
    ('15', '131', '-'),
    ('15', '132', '-'),
    ('15', '14', '-'),
    ('151', '3', '-'),
    ('151', '5', '-'),
    ('151', '12', '-'),
    ('151', '13', '-'),
    ('151', '131', '-'),
    ('151', '132', '-'),
    ('151', '14', '-')],

    # MY REACTIONS

    [('16', '14', '-'),
    ('17', '14', '-')],
)

# Create a dictionary of environments 
environMatchers = {}
for env, smart in environDefs.items():
    environMatchers[env] = Chem.MolFromSmarts(smart)

# Create all combinations of envs as defined in reactionDefs
bondMatchers = []
for compats in reactionDefs:
    tmp = []
    for i1, i2, bondType in compats:
        e1 = environDefs['L' + i1]
        e2 = environDefs['L' + i2]
        patt = f"[$({e1})]{bondType}[$({e2})]"
        patt = Chem.MolFromSmarts(patt)
        tmp.append((i1, i2, patt))
    bondMatchers.append(tmp)

def SSSRsize_filter(bond, maxSR=8):
    judge=True
    for i in range(3, maxSR+1):
        if bond.IsInRingSize(i):
            judge=False
            break           
    return judge
    
def searchBonds(mol, maxSR=8):
    # Check what reactions apply
    envMatches = {}   
    for env, patt in environMatchers.items():
        envMatches[env] = mol.HasSubstructMatch(patt)

    validBonds = set()
    for compats in bondMatchers:
        for i1, i2, patt in compats:
            if not envMatches['L' + i1] or not envMatches['L' + i2]:
                continue
            matches = mol.GetSubstructMatches(patt)
            # for m in matches:
            #     Draw.MolToImage(mol, highlightAtoms=m, size=(900,900)).show()
            #     exit()
            for match in matches:
                if match not in validBonds and match[::-1] not in validBonds:
                    bond = mol.GetBondBetweenAtoms(match[0], match[1])
                    if not bond.IsInRing():
                        validBonds.add(match)
                        yield ((match, (i1, i2)))
                    elif bond.IsInRing() and SSSRsize_filter(bond, maxSR=maxSR):
                        validBonds.add(match)
                        yield ((match, (i1, i2)))

def add_atom_indices(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol

def remove_atom_indices(mol):
    # cmol = deepcopy(mol)
    for atom in mol.GetAtoms():
        atom.ClearProp('molAtomMapNumber')
    return mol
    
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

#!####### MY NEW FUNCTIONS #########
def flatten(lst):
    return [item for sublist in lst for item in sublist]

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
#!####### ---------------- #########


def GraphDecomp(mol, maxBlocks=8, maxSR=6, keep_AtomMapNumber=True, keep_connectivity=False, method='default'):
    """
    Parameters:
        maxBlocks: The maximum number of building blocks that the fragments contain
        maxSR: rings containing a number of atoms equal to or less than this value will not be fragmented
    """
    mol = add_atom_indices(mol)

    if method == 'BRICS':
        #! Original code using BRICS as a fragmentation algorithm
        bonds = list(searchBonds(mol, maxSR=maxSR))
        blocks = list(Chem.GetMolFrags(BreakBRICSBonds(mol, bonds, keep_connectivity=False), asMols=True))
        labelled_bonds = bonds  # Retain L-labels for BRICS fragmentation
        bonds = [x[0] for x in bonds]  # Remove L-labels to prepare for new method
        #! ------------------------------------ #
    else:
        #! My version uses the following algorithm
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
        #! ------------------------------------ #

    for block in blocks:
        block_smi = Chem.MolToSmiles(block)
        nAtoms = block.GetNumAtoms(onlyExplicit=True) - block_smi.count('*')
        nRings = [list(x) for x in block.GetRingInfo().AtomRings()]
        nAtomsRings = [len(x) for x in nRings]

        # Classify blocks based on structural type  #TODO: add Decorator type
        if nAtomsRings:
            if nAtomsRings == [3] or nAtomsRings == [4]: type = 'T'  # Small rings (cyclopropane and -butane)
            else: type = 'R'  # Rings
        elif 'P' in block_smi: type = 'P'  # Phosphate blocks
        elif block_smi.count('*') > 1: type = 'L'  # Linkers
        elif block_smi.count('*') == 1 and nAtoms >= 4: type = 'S'  # Side-chains
        else: type = 'X'  # Substituents

        # Assign block properties 
        block.SetProp('Type', type)
        block.SetProp('Smiles', block_smi)

    #! No recomposition
    if maxBlocks == 1 or not mol.GetRingInfo().AtomRings():  # Skip molecules with no rings
        if not keep_AtomMapNumber:
            [remove_atom_indices(block) for block in blocks]
        return blocks
        print('h')
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

    if method == 'BRICS':
        bonds_to_break = [b for b in labelled_bonds for x in bonds_to_break if x in b]
        new_blocks = list(Chem.GetMolFrags(BreakBRICSBonds(mol, bonds_to_break, keep_connectivity=keep_connectivity), asMols=True))
    else:
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

from tools import general_methods as GenMtd
def remove_NIG(mol, interacting_atoms):
    """ NIG = Non-Interacting Groups """
    K = Chem.MolFromSmiles('[99*]'); D = Chem.MolFromSmiles('*')
    try:
        # Fragment molecule
        cmol = deepcopy(mol)
        ring_atoms = sorted(set(flatten(cmol.GetRingInfo().AtomRings())), reverse=True)
        qmol = GenMtd.replace_dummies(cmol)  # Remove dummies if any
        map = list(cmol.GetSubstructMatch(qmol))
        blocks = GraphDecomp(qmol, maxBlocks=1, keep_AtomMapNumber=True)

        if len(blocks) > 1:
            # Look for decorative substituent groups
            decorative_Xs = set()
            for block in blocks:
                # Keep acyl oxygen attached to rings  #TODO: This should be implemented inside GraphDecomp!
                # if Chem.MolToSmarts(remove_atom_indices(block)) in ['[#0]=[#8]', '[#8]=[#0]']:
                #     continue
                indices = [a.GetAtomMapNum() for a in block.GetAtoms() if a.GetSymbol()!='*']
                mapped_indices = sorted([map[x] for x in indices], reverse=True)
                if mapped_indices != ring_atoms:
                    decorative_Xs.add(tuple(mapped_indices))

            # Keep only non-interacting groups
            decorative_Xs = [x for x in decorative_Xs if not set(x).intersection(set(interacting_atoms))]
            decorative_Xs = sorted(set(flatten(decorative_Xs)), reverse=True)

            rwm = Chem.RWMol(cmol)
            for idx in decorative_Xs:
                neighs = [neigh.GetIdx() for neigh in cmol.GetAtomWithIdx(idx).GetNeighbors()]
                if set(neighs).intersection(set(ring_atoms)):
                    # Atoms directly bonded to ring
                    atomr = Chem.Atom(0)
                    atomr.SetIsotope(99)
                    atomr.SetNoImplicit(True)
                    rwm.ReplaceAtom(idx, atomr)
                else:
                    rwm.RemoveAtom(idx)
            new_mol = rwm.GetMol()
            new_mol = Chem.RemoveHs(new_mol)
            Chem.SanitizeMol(new_mol)
            new_mol = Chem.ReplaceSubstructs(new_mol, K, D, replaceAll=True)[0]
            return Chem.MolToSmiles(new_mol)
        else:
            raise Exception
    except Exception as e:
        # print(e)
        return Chem.MolToSmiles(mol)