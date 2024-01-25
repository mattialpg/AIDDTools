#
# calculation of synthetic accessibility score as described in:
#
# Estimation of Synthetic Accessibility Score of Drug-like Molecules based on Molecular Complexity and Fragment Contributions
# Peter Ertl and Ansgar Schuffenhauer
# Journal of Cheminformatics 1:8 (2009)
# http://www.jcheminf.com/content/1/1/8
#
# several small modifications to the original paper are included
# particularly slightly different formula for marocyclic penalty
# and taking into account also molecule symmetry (fingerprint density)
#
# for a set of 10k diverse molecules the agreement between the original method
# as implemented in PipelinePilot and this implementation is r2 = 0.97
#
# peter ertl & greg landrum, september 2013
#

import numpy as np
import pandas as pd
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from GraphDecomp import *

def readFragmentScores(fragment_database):
    df = pd.read_pickle(fragment_database)
    frag_scores = pd.Series(df.Fragment_Score.values, index=df.Isomeric_SMILES_FRAG).to_dict()
    return frag_scores


def calculateScore(mol, frag_scores):
    # Calculate fragment score
    blocks = GraphDecomp(mol, maxBlocks=15, maxSR=12, keep_AtomMapNumber=False)
    score1 = sum(frag_scores.get(Chem.MolToSmiles(block), 0) for block in blocks)

    # Calculate molecular features
    nAtoms = mol.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    ri = mol.GetRingInfo()
    nSpiros = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    nBridgeheads = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    nMacrocycles = sum(1 for x in ri.AtomRings() if len(x) > 8)

    # Calculate penalties
    sizePenalty = nAtoms**1.005 - nAtoms
    stereoPenalty = np.log2(nChiralCenters + 1)
    spiroPenalty = np.log2(nSpiros + 1)
    bridgePenalty = np.log2(nBridgeheads + 1)
    macrocyclePenalty = np.log2(2) if nMacrocycles > 0 else 0
    score2 = 0 - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty

    # Correction to make symmetrical molecules easier to synthetise
    score3 = np.log2(float(nAtoms) / len(blocks)) * 0.5 if nAtoms > len(blocks) else 0

    sascore = score1 + score2 + score3

    # # need to transform "raw" value into scale between 1 and 10
    # min = -4.0
    # max = 2.5
    # sascore = 11. - (sascore - min + 1) / (max - min) * 9.
    # # smooth the 10-end
    # if sascore > 8.:
    #     sascore = 8. + np.log2(sascore + 1. - 9.)
    # if sascore > 10.:
    #     sascore = 10.0
    # elif sascore < 1.:
    #     sascore = 1.0

    return sascore


def processMols(mols, fragment_database):
    frag_scores = readFragmentScores(fragment_database)
    for mol in mols:
        s = calculateScore(mol, frag_scores)
        # print(f"{Chem.MolToSmiles(mol)}\t\t{s:.03}")


if __name__ == '__main__':
    import sys
    import time

    t1 = time.time()
    readFragmentScores("fpscores")
    t2 = time.time()

    suppl = Chem.SmilesMolSupplier(sys.argv[1])
    t3 = time.time()
    processMols(suppl)
    t4 = time.time()

    print('Reading took %.2f seconds. Calculating took %.2f seconds' % ((t2 - t1), (t4 - t3)),
                file=sys.stderr)