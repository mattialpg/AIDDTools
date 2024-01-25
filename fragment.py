from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
AllChem.SetPreferCoordGen(True)
from openeye.oechem import *
from openeye.oemedchem import *
from openeye.oedepict import *

def getSubmolRadN(mol, radius):
    atoms=mol.GetAtoms()
    submols=[]
    for atom in atoms:
        env=Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom.GetIdx())
        amap={}
        submol=Chem.PathToSubmol(mol, env, atomMap=amap)
        subsmi=Chem.MolToSmiles(submol, rootedAtAtom=amap[atom.GetIdx()], canonical=False)
        submols.append(Chem.MolFromSmiles(subsmi, sanitize=False))
    return submols
	
def depict_molecule_with_fragment_highlights(image, mol, fragfunc):

    fraglist = [f for f in fragfunc(mol)]

    colorg = oechem.OELinearColorGradient()
    colorg.AddStop(oechem.OEColorStop(0, oechem.OEMediumYellow))
    colorg.AddStop(oechem.OEColorStop(len(fraglist), oechem.OEDarkBrown))

    disp = oedepict.OE2DMolDisplay(mol)

    highlight = oedepict.OEHighlightByLasso(oechem.OEWhite)
    highlight.SetConsiderAtomLabelBoundingBox(True)

    for fidx, frag in enumerate(fraglist):
        highlight.SetColor(colorg.GetColorAt(fidx))
        oedepict.OEAddHighlighting(disp, highlight, frag)

    oedepict.OERenderMolecule(image, disp)

mol = Chem.MolFromSmiles('c1ccnc(c1)NS(=O)(=O)c2ccc(cc2)/N=N/c3ccc(c(c3)C(=O)O)O')
depict_molecule_with_fragment_highlights('image.png', mol, OEGetFuncGroupFragments)


# img1 = Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=(900,600)); img1.show()
# mol1_f = Chem.FragmentOnBonds(mol, (0, 2, 4))
# img1 = Draw.MolsToGridImage([mol1_f], highlightAtomLists=[[0] for _ in range(len([mol1_f]))], subImgSize=(900,600)); img1.show()