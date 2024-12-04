import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from mordred import Calculator, descriptors

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, RepeatedKFold
from sklearn.feature_selection import VarianceThreshold, RFECV
from sklearn.linear_model import ElasticNetCV, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SequentialFeatureSelector
# from tpot import TPOTClassifier

"""For each descriptor in the module correlation matrix, we counted the number of descriptors with a correlation
coefficient of at least 0.7. We then ranked the descriptors from highest to lowest by number of correlated module
descriptors. Next, we selected the top descriptor as a feature descriptor and removed that feature descriptor and
its correlated descriptors from the correlation matrix. Last, we regenerated the correlation matrix and repeated
these steps until all descriptors in the module were either selected as a feature descriptor or deleted. If descriptors
had the same number of correlated descriptors and therefore the same ranking, they were selected by their order in Mordred.
We repeated this process for individual modules and then a final time after combining all feature descriptors. This
process generated the feature descriptor set for RFE. [10.1016/j.fuel.2022.123836]"""

# descriptor_list = ['nAcid', 'MW', 'Zagreb1', 'WPath']
# calc.descriptors = [d for d in calc.descriptors if str(d) in descriptor_list]  # re-register subset of descriptors
# calc.descriptors = [d for d in calc.descriptors if "ATS" not in str(d)]  # Ignore autocorrelation
# descr = [calc(mol) for mol in mols]

module_dict = {
    'ABCIndex': ['ABC', 'ABCGG'],
    'AcidBase': ['nAcid', 'nBase'],
    'Aromatic': ['nAromAtom', 'nAromBond'],
    'AtomCount': ['nAtom', 'nHeavyAtom', 'nSpiro', 'nBridgehead', 'nH', 'nB', 'nC', 'nN', 'nO', 'nS', 'nP', 'nF', 'nCl', 'nBr', 'nI', 'nX'],
    'AdjacencyMatrix': ['SpAbs_A', 'SpMax_A', 'SpDiam_A', 'SpAD_A', 'SpMAD_A','LogEE_A', 'SM1_A', 'VE1_A', 'VE2_A', 'VE3_A', 'VR1_A', 'VR2_A', 'VR3_A'],
    'BCUT': ['BCUTse-1h', 'BCUTse-1l', 'BCUTv-1h', 'BCUTv-1l', 'BCUTZ-1h', 'BCUTZ-1l', 'BCUTpe-1h', 'BCUTpe-1l', 'BCUTs-1h', 'BCUTs-1l', 'BCUTc-1h', 'BCUTc-1l', 'BCUTdv-1h', 'BCUTdv-1l', 'BCUTi-1h', 'BCUTi-1l', 'BCUTd-1h', 'BCUTd-1l',  'BCUTm-1h', 'BCUTm-1l', 'BCUTare-1h', 'BCUTare-1l', 'BCUTp-1h', 'BCUTp-1l'],
    'BaryszMatrix': ['SpAbs_Dzse', 'SpMax_Dzse', 'SpDiam_Dzse', 'SpAD_Dzse', 'SpMAD_Dzse', 'LogEE_Dzse', 'SM1_Dzse', 'VE1_Dzse', 'VE2_Dzse', 'VE3_Dzse', 'VR1_Dzse', 'VR2_Dzse', 'VR3_Dzse', 'SpAbs_Dzv', 'SpMax_Dzv', 'SpDiam_Dzv', 'SpAD_Dzv', 'SpMAD_Dzv', 'LogEE_Dzv', 'SM1_Dzv', 'VE1_Dzv', 'VE2_Dzv', 'VE3_Dzv', 'VR1_Dzv', 'VR2_Dzv', 'VR3_Dzv', 'SpAbs_DzZ', 'SpMax_DzZ', 'SpDiam_DzZ', 'SpAD_DzZ', 'SpMAD_DzZ', 'LogEE_DzZ', 'SM1_DzZ', 'VE1_DzZ', 'VE2_DzZ', 'VE3_DzZ', 'VR1_DzZ', 'VR2_DzZ', 'VR3_DzZ', 'SpAbs_Dzpe', 'SpMax_Dzpe', 'SpDiam_Dzpe', 'SpAD_Dzpe', 'SpMAD_Dzpe', 'LogEE_Dzpe', 'SM1_Dzpe', 'VE1_Dzpe', 'VE2_Dzpe', 'VE3_Dzpe', 'VR1_Dzpe', 'VR2_Dzpe', 'VR3_Dzpe', 'SpAbs_Dzi', 'SpMax_Dzi', 'SpDiam_Dzi', 'SpAD_Dzi', 'SpMAD_Dzi', 'LogEE_Dzi', 'SM1_Dzi', 'VE1_Dzi', 'VE2_Dzi', 'VE3_Dzi', 'VR1_Dzi', 'VR2_Dzi', 'VR3_Dzi', 'SpAbs_Dzm', 'SpMax_Dzm', 'SpDiam_Dzm', 'SpAD_Dzm', 'SpMAD_Dzm', 'LogEE_Dzm', 'SM1_Dzm', 'VE1_Dzm', 'VE2_Dzm', 'VE3_Dzm', 'VR1_Dzm', 'VR2_Dzm', 'VR3_Dzm', 'SpAbs_Dzare', 'SpMax_Dzare', 'SpDiam_Dzare', 'SpAD_Dzare', 'SpMAD_Dzare', 'LogEE_Dzare', 'SM1_Dzare', 'VE1_Dzare', 'VE2_Dzare', 'VE3_Dzare', 'VR1_Dzare', 'VR2_Dzare', 'VR3_Dzare', 'SpAbs_Dzp', 'SpMax_Dzp', 'SpDiam_Dzp', 'SpAD_Dzp', 'SpMAD_Dzp', 'LogEE_Dzp', 'SM1_Dzp', 'VE1_Dzp', 'VE2_Dzp', 'VE3_Dzp', 'VR1_Dzp', 'VR2_Dzp', 'VR3_Dzp'],
    'BertzCT': ['BertzCT'],
    'BondCount': ['nBonds', 'nBondsO', 'nBondsS', 'nBondsD', 'nBondsT', 'nBondsA', 'nBondsM', 'nBondsKS', 'nBondsKD', 'RNCG', 'RPCG',],
    'CarbonTypes': ['C1SP1', 'C2SP1', 'C1SP2', 'C2SP2', 'C3SP2', 'C1SP3', 'C2SP3', 'C3SP3', 'C4SP3', 'HybRatio'],
    'Chi': ['FCSP3', 'Xch-3d', 'Xch-4d', 'Xch-5d', 'Xch-6d', 'Xch-7d', 'Xch-3dv', 'Xch-4dv', 'Xch-5dv', 'Xch-6dv', 'Xch-7dv', 'Xc-3d', 'Xc-4d', 'Xc-5d', 'Xc-6d', 'Xc-3dv', 'Xc-4dv', 'Xc-5dv', 'Xc-6dv', 'Xpc-4d', 'Xpc-5d', 'Xpc-6d', 'Xpc-4dv', 'Xpc-5dv', 'Xpc-6dv', 'Xp-0d', 'Xp-1d', 'Xp-2d', 'Xp-3d', 'Xp-4d', 'Xp-5d', 'Xp-6d', 'Xp-7d', 'AXp-0d', 'AXp-1d', 'AXp-2d', 'AXp-3d', 'AXp-4d', 'AXp-5d', 'AXp-6d', 'AXp-7d', 'Xp-0dv', 'Xp-1dv', 'Xp-2dv', 'Xp-3dv', 'Xp-4dv', 'Xp-5dv', 'Xp-6dv', 'Xp-7dv', 'AXp-0dv', 'AXp-1dv', 'AXp-2dv', 'AXp-3dv', 'AXp-4dv', 'AXp-5dv', 'AXp-6dv', 'AXp-7dv'],
    'Constitutional': ['SZ', 'Sm', 'Sv', 'Sse', 'Spe', 'Sare', 'Sp', 'Si', 'MZ', 'Mm', 'Mv', 'Mse', 'Mpe', 'Mare', 'Mp', 'Mi'],
    'DetourMatrix': ['SpAbs_Dt', 'SpMax_Dt', 'SpDiam_Dt', 'SpAD_Dt', 'SpMAD_Dt', 'LogEE_Dt', 'SM1_Dt', 'VE1_Dt', 'VE2_Dt', 'VE3_Dt', 'VR1_Dt', 'VR2_Dt', 'VR3_Dt', 'DetourIndex'],
    'DistanceMatrix': ['SpAbs_D', 'SpMax_D', 'SpDiam_D', 'SpAD_D', 'SpMAD_D', 'LogEE_D', 'SM1_D', 'VE1_D', 'VE2_D', 'VE3_D', 'VR1_D', 'VR2_D', 'VR3_D'],
    'EState': ['NsLi', 'NssBe', 'NssssBe', 'NssBH', 'NsssB', 'NssssB', 'NsCH3', 'NdCH2', 'NssCH2', 'NtCH', 'NdsCH', 'NaaCH', 'NsssCH', 'NddC', 'NtsC', 'NdssC', 'NaasC', 'NaaaC', 'NssssC', 'NsNH3', 'NsNH2', 'NssNH2', 'NdNH', 'NssNH', 'NaaNH', 'NtN', 'NsssNH', 'NdsN', 'NaaN', 'NsssN', 'NddsN', 'NaasN', 'NssssN', 'NsOH', 'NdO', 'NssO', 'NaaO', 'NsF', 'NsSiH3', 'NssSiH2', 'NsssSiH', 'NssssSi', 'NsPH2', 'NssPH', 'NsssP', 'NdsssP', 'NsssssP', 'NsSH', 'NdS', 'NssS', 'NaaS', 'NdssS', 'NddssS', 'NsCl', 'NsGeH3', 'NssGeH2', 'NsssGeH', 'NssssGe', 'NsAsH2', 'NssAsH', 'NsssAs', 'NsssdAs', 'NsssssAs', 'NsSeH', 'NdSe', 'NssSe', 'NaaSe', 'NdssSe', 'NddssSe', 'NsBr', 'NsSnH3', 'NssSnH2', 'NsssSnH', 'NssssSn', 'NsI', 'NsPbH3', 'NssPbH2', 'NsssPbH', 'NssssPb', 'SsLi', 'SssBe', 'SssssBe', 'SssBH', 'SsssB', 'SssssB', 'SsCH3', 'SdCH2', 'SssCH2', 'StCH', 'SdsCH', 'SaaCH', 'SsssCH', 'SddC', 'StsC', 'SdssC', 'SaasC', 'SaaaC', 'SssssC', 'SsNH3', 'SsNH2', 'SssNH2', 'SdNH', 'SssNH', 'SaaNH', 'StN', 'SsssNH', 'SdsN', 'SaaN', 'SsssN', 'SddsN', 'SaasN', 'SssssN', 'SsOH', 'SdO', 'SssO', 'SaaO', 'SsF', 'SsSiH3', 'SssSiH2', 'SsssSiH', 'SssssSi', 'SsPH2', 'SssPH', 'SsssP', 'SdsssP', 'SsssssP', 'SsSH', 'SdS', 'SssS', 'SaaS', 'SdssS', 'SddssS', 'SsCl', 'SsGeH3', 'SssGeH2', 'SsssGeH', 'SssssGe', 'SsAsH2', 'SssAsH', 'SsssAs', 'SsssdAs', 'SsssssAs', 'SsSeH', 'SdSe', 'SssSe', 'SaaSe', 'SdssSe', 'SddssSe', 'SsBr', 'SsSnH3', 'SssSnH2', 'SsssSnH', 'SssssSn', 'SsI', 'SsPbH3', 'SssPbH2', 'SsssPbH', 'SssssPb', 'MAXsLi', 'MAXssBe', 'MAXssssBe', 'MAXssBH', 'MAXsssB', 'MAXssssB', 'MAXsCH3', 'MAXdCH2', 'MAXssCH2', 'MAXtCH', 'MAXdsCH', 'MAXaaCH', 'MAXsssCH', 'MAXddC', 'MAXtsC', 'MAXdssC', 'MAXaasC', 'MAXaaaC', 'MAXssssC', 'MAXsNH3', 'MAXsNH2', 'MAXssNH2', 'MAXdNH', 'MAXssNH', 'MAXaaNH', 'MAXtN', 'MAXsssNH', 'MAXdsN', 'MAXaaN', 'MAXsssN', 'MAXddsN', 'MAXaasN', 'MAXssssN', 'MAXsOH', 'MAXdO', 'MAXssO', 'MAXaaO', 'MAXsF', 'MAXsSiH3', 'MAXssSiH2', 'MAXsssSiH', 'MAXssssSi', 'MAXsPH2', 'MAXssPH', 'MAXsssP', 'MAXdsssP', 'MAXsssssP', 'MAXsSH', 'MAXdS', 'MAXssS', 'MAXaaS', 'MAXdssS', 'MAXddssS', 'MAXsCl', 'MAXsGeH3', 'MAXssGeH2', 'MAXsssGeH', 'MAXssssGe', 'MAXsAsH2', 'MAXssAsH', 'MAXsssAs', 'MAXsssdAs', 'MAXsssssAs', 'MAXsSeH', 'MAXdSe', 'MAXssSe', 'MAXaaSe', 'MAXdssSe', 'MAXddssSe', 'MAXsBr', 'MAXsSnH3', 'MAXssSnH2', 'MAXsssSnH', 'MAXssssSn', 'MAXsI', 'MAXsPbH3', 'MAXssPbH2', 'MAXsssPbH', 'MAXssssPb', 'MINsLi', 'MINssBe', 'MINssssBe', 'MINssBH', 'MINsssB', 'MINssssB', 'MINsCH3', 'MINdCH2', 'MINssCH2', 'MINtCH', 'MINdsCH', 'MINaaCH', 'MINsssCH', 'MINddC', 'MINtsC', 'MINdssC', 'MINaasC', 'MINaaaC', 'MINssssC', 'MINsNH3', 'MINsNH2', 'MINssNH2', 'MINdNH', 'MINssNH', 'MINaaNH', 'MINtN', 'MINsssNH', 'MINdsN', 'MINaaN', 'MINsssN', 'MINddsN', 'MINaasN', 'MINssssN', 'MINsOH', 'MINdO', 'MINssO', 'MINaaO', 'MINsF', 'MINsSiH3', 'MINssSiH2', 'MINsssSiH', 'MINssssSi', 'MINsPH2', 'MINssPH', 'MINsssP', 'MINdsssP', 'MINsssssP', 'MINsSH', 'MINdS', 'MINssS', 'MINaaS', 'MINdssS', 'MINddssS', 'MINsCl', 'MINsGeH3', 'MINssGeH2', 'MINsssGeH', 'MINssssGe', 'MINsAsH2', 'MINssAsH', 'MINsssAs', 'MINsssdAs', 'MINsssssAs', 'MINsSeH', 'MINdSe', 'MINssSe', 'MINaaSe', 'MINdssSe', 'MINddssSe', 'MINsBr', 'MINsSnH3', 'MINssSnH2', 'MINsssSnH', 'MINssssSn', 'MINsI', 'MINsPbH3', 'MINssPbH2', 'MINsssPbH', 'MINssssPb'],
    'EccentricConnectivityIndex': ['ECIndex'],
    'ExtendedTopochemicalAtom': ['ETA_alpha', 'AETA_alpha', 'ETA_shape_p', 'ETA_shape_y', 'ETA_shape_x', 'ETA_beta', 'AETA_beta', 'ETA_beta_s', 'AETA_beta_s', 'ETA_beta_ns', 'AETA_beta_ns', 'ETA_beta_ns_d', 'AETA_beta_ns_d', 'ETA_eta', 'AETA_eta', 'ETA_eta_L', 'AETA_eta_L', 'ETA_eta_R', 'AETA_eta_R', 'ETA_eta_RL', 'AETA_eta_RL', 'ETA_eta_F', 'AETA_eta_F', 'ETA_eta_FL', 'AETA_eta_FL', 'ETA_eta_B', 'AETA_eta_B', 'ETA_eta_BR', 'AETA_eta_BR', 'ETA_dAlpha_A', 'ETA_dAlpha_B', 'ETA_epsilon_1', 'ETA_epsilon_2', 'ETA_epsilon_3', 'ETA_epsilon_4', 'ETA_epsilon_5', 'ETA_dEpsilon_A', 'ETA_dEpsilon_B', 'ETA_dEpsilon_C', 'ETA_dEpsilon_D', 'ETA_dBeta', 'AETA_dBeta', 'ETA_psi_1', 'ETA_dPsi_A', 'ETA_dPsi_B'],
    'FragmentComplexity': ['fragCpx'],
    'Framework': ['fMF'],
    'HydrogenBond': ['nHBAcc', 'nHBDon'],
    'InformationContent': ['IC0', 'IC1', 'IC2', 'IC3', 'IC4', 'IC5', 'TIC0', 'TIC1', 'TIC2', 'TIC3', 'TIC4', 'TIC5', 'SIC0', 'SIC1', 'SIC2', 'SIC3', 'SIC4', 'SIC5', 'BIC0', 'BIC1', 'BIC2', 'BIC3', 'BIC4', 'BIC5', 'CIC0', 'CIC1', 'CIC2', 'CIC3', 'CIC4', 'CIC5', 'MIC0', 'MIC1', 'MIC2', 'MIC3', 'MIC4', 'MIC5', 'ZMIC0', 'ZMIC1', 'ZMIC2', 'ZMIC3', 'ZMIC4', 'ZMIC5'],
    'KappaShapeIndex': ['Kier1', 'Kier2', 'Kier3'],
    'Lipinski': ['Lipinski', 'GhoseFilter'],
    'McGowanVolume': ['VMcGowan'],
    'MoeType': ['LabuteASA', 'PEOE_VSA1', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'SMR_VSA1', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 'SlogP_VSA10', 'SlogP_VSA11', 'EState_VSA1', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'EState_VSA10', 'VSA_EState1', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'VSA_EState9'],
    'MolecularDistanceEdge': ['MDEC-11', 'MDEC-12', 'MDEC-13', 'MDEC-14', 'MDEC-22', 'MDEC-23', 'MDEC-24', 'MDEC-33', 'MDEC-34', 'MDEC-44', 'MDEO-11', 'MDEO-12', 'MDEO-22', 'MDEN-11', 'MDEN-12', 'MDEN-13', 'MDEN-22', 'MDEN-23', 'MDEN-33'],
    'MolecularId': ['MID', 'AMID', 'MID_h', 'AMID_h', 'MID_C', 'AMID_C', 'MID_N', 'AMID_N', 'MID_O', 'AMID_O', 'MID_X', 'AMID_X'],
    'PathCount': ['MPC2', 'MPC3', 'MPC4', 'MPC5', 'MPC6', 'MPC7', 'MPC8', 'MPC9', 'MPC10', 'TMPC10', 'piPC1', 'piPC2', 'piPC3', 'piPC4', 'piPC5', 'piPC6', 'piPC7', 'piPC8', 'piPC9', 'piPC10', 'TpiPC10'],
    'Polarizability': ['apol', 'bpol'],
    'RingCount': ['nRing', 'n3Ring', 'n4Ring', 'n5Ring', 'n6Ring', 'n7Ring', 'n8Ring', 'n9Ring', 'n10Ring', 'n11Ring', 'n12Ring', 'nG12Ring', 'nHRing', 'n3HRing', 'n4HRing', 'n5HRing', 'n6HRing', 'n7HRing', 'n8HRing', 'n9HRing', 'n10HRing', 'n11HRing', 'n12HRing', 'nG12HRing', 'naRing', 'n3aRing', 'n4aRing', 'n5aRing', 'n6aRing', 'n7aRing', 'n8aRing', 'n9aRing', 'n10aRing', 'n11aRing', 'n12aRing', 'nG12aRing', 'naHRing', 'n3aHRing', 'n4aHRing', 'n5aHRing', 'n6aHRing', 'n7aHRing', 'n8aHRing', 'n9aHRing', 'n10aHRing', 'n11aHRing', 'n12aHRing', 'nG12aHRing', 'nARing', 'n3ARing', 'n4ARing', 'n5ARing', 'n6ARing', 'n7ARing', 'n8ARing', 'n9ARing', 'n10ARing', 'n11ARing', 'n12ARing', 'nG12ARing', 'nAHRing', 'n3AHRing', 'n4AHRing', 'n5AHRing', 'n6AHRing', 'n7AHRing', 'n8AHRing', 'n9AHRing', 'n10AHRing', 'n11AHRing', 'n12AHRing', 'nG12AHRing', 'nFRing', 'n4FRing', 'n5FRing', 'n6FRing', 'n7FRing', 'n8FRing', 'n9FRing', 'n10FRing', 'n11FRing', 'n12FRing', 'nG12FRing', 'nFHRing', 'n4FHRing', 'n5FHRing', 'n6FHRing', 'n7FHRing', 'n8FHRing', 'n9FHRing', 'n10FHRing', 'n11FHRing', 'n12FHRing', 'nG12FHRing', 'nFaRing', 'n4FaRing', 'n5FaRing', 'n6FaRing', 'n7FaRing', 'n8FaRing', 'n9FaRing', 'n10FaRing', 'n11FaRing', 'n12FaRing', 'nG12FaRing', 'nFaHRing', 'n4FaHRing', 'n5FaHRing', 'n6FaHRing', 'n7FaHRing', 'n8FaHRing', 'n9FaHRing', 'n10FaHRing', 'n11FaHRing', 'n12FaHRing', 'nG12FaHRing', 'nFARing', 'n4FARing', 'n5FARing', 'n6FARing', 'n7FARing', 'n8FARing', 'n9FARing', 'n10FARing', 'n11FARing', 'n12FARing', 'nG12FARing', 'nFAHRing', 'n4FAHRing', 'n5FAHRing', 'n6FAHRing', 'n7FAHRing', 'n8FAHRing', 'n9FAHRing', 'n10FAHRing', 'n11FAHRing', 'n12FAHRing', 'nG12FAHRing'],
    'RotatableBond': ['nRot', 'RotRatio'],
    'SLogP': ['SLogP', 'SMR'],
    'TopoPSA': ['TopoPSA(NO)', 'TopoPSA'],
    'TopologicalCharge': ['GGI1', 'GGI2', 'GGI3', 'GGI4', 'GGI5', 'GGI6', 'GGI7', 'GGI8', 'GGI9', 'GGI10', 'JGI1', 'JGI2', 'JGI3', 'JGI4', 'JGI5', 'JGI6', 'JGI7', 'JGI8', 'JGI9', 'JGI10', 'JGT10'],
    'TopologicalIndex': ['Radius', 'Diameter', 'TopoShapeIndex', 'PetitjeanIndex'],
    'VdwVolumeABC': ['Vabc'],
    'VertexAdjacencyInformation': ['VAdjMat'],
    'WalkCount': ['MWC01', 'MWC02', 'MWC03', 'MWC04', 'MWC05', 'MWC06', 'MWC07', 'MWC08', 'MWC09', 'MWC10', 'TMWC10', 'SRW02', 'SRW03', 'SRW04', 'SRW05', 'SRW06', 'SRW07', 'SRW08', 'SRW09', 'SRW10', 'TSRW10'],
    'Weight': ['MW', 'AMW'],
    'WienerIndex': ['WPath', 'WPol'],
    'ZagrebIndex': ['Zagreb1', 'Zagreb2', 'mZagreb1', 'mZagreb2']}

def calc_descriptors(mols, descriptor_list=None):
    #TODO: Maybe subdivide calculation into chunks of n descriptors
    #TODO: and write (append) the output file by chunks. This may save memory
    #TODO: and speed up the process
    global module_dict

    # Register all descriptors
    calc = Calculator(descriptors, ignore_3D=True)
    # Re-register subset of descriptors
    if not descriptor_list:
        nested_list = list(module_dict.values())
        descriptor_list = [x for sublist in nested_list for x in sublist]
    calc.descriptors = [d for d in calc.descriptors if str(d) in descriptor_list]

    descr = []
    for mol in tqdm(mols, desc='Calculating descriptors'):
        try: descr.append(calc(mol))
        except: descr.append([])
    df_descr = pd.DataFrame(descr, columns=descr[0].asdict().keys())
    df_descr = df_descr.apply(pd.to_numeric, errors='coerce')   # Set non-numeric values to NaN
    return df_descr

def filter_by_var(df_descriptors, nonzero_thrd=0.1):
    """ Df must be normalised. Standardise for example with StandardScaler,
    i.e. remove the mean and scale to unit variance, is wrong because each
    feature will have variance 1. """

    global module_dict
    nonzero_list = []
    for key in module_dict.keys():
        columns = [c for c in module_dict[key] if c in df_descriptors.columns]
        df = df_descriptors[columns]

        if df.empty or df is None:
            pass
        elif len(columns) < 3:
            nonzero_list.extend(columns)
        else:
            # Remove constant features
            selector = VarianceThreshold(nonzero_thrd)
            selector.fit(df)
            selector.get_feature_names_out
            nonzero_df = df[df.columns[selector.get_support(indices=True)]]
            nonzero_list.extend(nonzero_df.columns.to_list())
    print(f"\nRound 1: Selected {len(nonzero_list)} non-zero variance features out of {df_descriptors.shape[1]}.")
    return nonzero_list

def filter_by_corr(df_descriptors, descriptor_list, corr_thrd=0.9): 
    """Iteratively remove highly correlated features as described
    by Comesana et al. (DOI: 10.1016/j.fuel.2022.123836)"""

    global module_dict
    uncorr_list1 = []
    for key in module_dict.keys():
        columns = [c for c in module_dict[key] if c in descriptor_list]
        df = df_descriptors[columns]

        corr_matrix = df.corr(method='spearman').abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))  # Select upper triangle of correlation matrix
        while np.any(upper > corr_thrd):
            corr_matrix = df.corr(method='spearman').abs()  # Create correlation matrix
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))  # Select upper triangle of correlation matrix

            plt.matshow(corr_matrix)
            plt.show()

            max_correlated = (upper > corr_thrd).apply(np.count_nonzero).idxmax(axis=0)  # Label of the most correlated feature
            correlated = upper.index[upper[max_correlated] > corr_thrd].tolist()  # Labels of features correlated with the previous one
            df = df.drop(columns=correlated).drop(columns=max_correlated)
            uncorr_list1.append(max_correlated)
            if df.empty: break
    print(f"Round 2: Selected {len(uncorr_list1)} uncorrelated features out of {len(descriptor_list)}.")

    # Filter again all the remaining features
    uncorr_list2 = []
    df = df_descriptors[uncorr_list1]
    corr_matrix = df.corr(method='spearman').abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    while np.any(upper > corr_thrd):
        corr_matrix = df.corr(method='spearman').abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        max_correlated = (upper > corr_thrd).apply(np.count_nonzero).idxmax(axis=0)
        correlated = upper.index[upper[max_correlated] > corr_thrd].tolist()
        df = df.drop(columns=correlated).drop(columns=max_correlated)
        uncorr_list2.append(max_correlated)
        if df.empty: break
    print(f"Round 3: Selected {len(uncorr_list2)} uncorrelated features out of {len(uncorr_list1)}.")
    return uncorr_list2

def reduce_by_RFECV(X, y, problem_type):
    if problem_type == 'classification':
        model = DecisionTreeClassifier()
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        scoring = 'accuracy'
        estimator_list = [
            # LogisticRegression(),
            DecisionTreeClassifier(),
            # SVC(kernel='linear'),
            # GradientBoostingClassifier(),
            ]
    elif problem_type == 'regression':
        model = DecisionTreeRegressor()
        cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
        scoring = 'r2'
        estimator_list = [
            RandomForestRegressor(),
            # DecisionTreeRegressor(),
            # GradientBoostingRegressor(),
            # ElasticNetCV(),
            ]
        
    # Evaluate estimator and store results
    for estim in estimator_list:
        name = estim.__class__.__name__
        print(f"\nApplying {name}...")
        selector = RFECV(estimator=estim)
        pipe = Pipeline(steps=[('s', selector),('m', model)])

        # Run and evaluate model
        pipe.fit(X, y)
        scores = cross_val_score(pipe, X, y, scoring=scoring, cv=cv, n_jobs=-1)

        df_results = pd.DataFrame([selector.support_, selector.ranking_]).T
        df_results.columns = ['Support', 'Ranking']
        df_results['Descriptor'] = X.columns.values
        df_results = df_results[['Ranking', 'Descriptor', 'Support']].sort_values(by='Ranking')
        # print(df_results)

        df_reduced = df_results[df_results['Support'] == True]
        reduced_list = df_reduced['Descriptor'].values.tolist()
        
        # Report performance
        print(f"  MAE: {scores.mean():.3f} ({scores.std():.3f})")
        print(f"  Optimal number of features: {selector.n_features_}")
        print(f"  {reduced_list}")
    return reduced_list

def reduce_by_SBFE(X, y, problem_type, drop=10):
    """Step-Backward Feature Elimination"""
    
    if problem_type == 'classification':
        estimator = LogisticRegression()
        cv = StratifiedKFold(n_splits=5, shuffle=True)
        scoring = 'accuracy'
    elif problem_type == 'regression':
        estimator = DecisionTreeRegressor()
        cv = RepeatedKFold(n_splits=2, n_repeats=10)
        scoring = 'r2'

    # model = RandomForestClassifier()
    ranks = []
    X1 = X.copy()
    while len(X1.columns) > drop:
        n = len(X1.columns)-drop
        selector = SequentialFeatureSelector(estimator,
                                            n_features_to_select=n,
                                            direction='backward',
                                            scoring='accuracy',
                                            cv=cv,
                                            n_jobs=-1)
        # pipe = Pipeline([('s', selector), ('m', model)])

        # Run and evaluate model
        selector.fit(X1, y)
        # scores = cross_val_score(pipe, X, y, scoring='accuracy', n_jobs=-1)

        dropped_cols = sorted(list(set(X1.columns) - set(selector.get_feature_names_out())), key=str.lower)
        X1 = X1[selector.get_feature_names_out()]
        ranks.append(dropped_cols)
    ranks.append(list(X1.columns))
    ranks = ranks[::-1]

    # # Find the best combination of descriptors
    # model = TPOTClassifier(generations=5, population_size=20, scoring='accuracy', verbosity=2, n_jobs=-1)
    # descr_list = []
    # for n in range(len(ranks)):
    #     descr_list.extend(ranks[n])
    #     print(descr_list)
    #     X2 = X[descr_list]
    #     model.fit(X2, y)
    #     # model.export(f"tpot_pipeline_{n}.py")

    return ranks[::-1]

def reduce_by_kxy(X, y, problem_type):
    import kxy
    X.kxy.variable_selection(y, problem_type=problem_type)

# plot curve for estimaotr

    # Plot model performance for comparison
    # plt.boxplot(results, labels=names, showmeans=True)
    # plt.show()

    # # Compute the curves
    # x = self.n_feature_subsets_
    # means = self.cv_scores_.mean(axis=1)
    # sigmas = self.cv_scores_.std(axis=1)

    # # Plot one standard deviation above and below the mean
    # self.ax.fill_between(x, means - sigmas, means + sigmas, alpha=0.25)

    # # Plot the curve
    # self.ax.plot(x, means, "o-")

    # # Plot the maximum number of features
    # self.ax.axvline(
    #     self.n_features_,
    #     c="k",
    #     ls="--",
    #     label="n_features = {}\nscore = {:0.3f}".format(
    #         self.n_features_, self.cv_scores_.mean(axis=1).max()
    #     ),
    # )

    # # Set the title of the figure
    # self.set_title("RFECV for {}".format(self.name))

    # # Add the legend
    # self.ax.legend(frameon=True, loc="best")

    # # Set the axis labels
    # self.ax.set_xlabel("Number of Features Selected")
    # self.ax.set_ylabel("Score")