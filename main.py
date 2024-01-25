import os, sys, glob, subprocess
import numpy as np
import pandas as pd
from natsort import natsorted

from sequencelib import *
from dockinglib import *
from analysislib import *

######### SEQUENCE ANALYSIS #########
# consensus = \
# 'AGVLWDVPSPPPVGKAELEDGAYRIKQKGILGYSQIGAGVYKEGTFHTMWHVTRGAVLMH\
# KGKRIEPSWADVKKDLISYGGGWKLEGEWKEGEEVQVLALEPGKNPRAVQTKPGLFKTNT\
# GTIGAVSLDFSPGTSGSPIVDKKGKVVGLYGNGVVTRSGAYVSAIANTEKSIEDNPEIED\
# DIFRK'
# fasta_from_pdb('*.pdb')
# plot_alignment('seq_all.fasta', type='mutations')
#####################################

######### MANIPULATE PDBs #########
download_pdb('pdb_list.txt')
# align_pdb('*.pdb','3U1I.pdb')
###################################

######### INTERACTION ANALYSIS #########
# split_reclig('*pdb')#, header=True, orig_folder='./')
# prepare_lig('ligands.csv', draw='known_ligands.png')
# df1 = df_interactions('*.pdb', outfile='interactions.csv', rename_apo=True)
# df2 = intmap('interactions.csv', pivot='RESN')
# heatmap(df2, savename='heatmap_complete.png', square=False)
# df2 = intmap('interactions.csv', pivot='RESTYPE')
# heatmap(df2, savename='heatmap_restype.png', square=False)
# df3 = filter_int(df2)
# heatmap(df3, savename='heatmap_filtered.png')
# heatmap('intmap_cov.txt', savename='heatmap_cov.png')
# heatmap('intmap_noncov.txt', savename='heatmap_noncov.png')
########################################

######### DOCKING ANALYSIS #########
# pdblist = glob.glob1(os.getcwd(), '*out*pdb')
# for pdb in pdblist:
	# merge_reclig('7L11_receptor.pdb', pdb, pdb.replace('out','complex'))

# df1 = interaction_df('*complex*', 'interactions.csv')
# df2 = intmap('interactions.csv')
# heatmap(df2, savename='heatmap.png')
# heatmap(df2, ref=r"C:\Users\Mattia\MEGA\DEVSHEALTH\COVID-19\MEDIOID\intmap_noncov.txt", savename='heatmap_compare.png')
####################################

######### MD ANALYSIS #########
# df1 = interaction_df('*.pdb', 'interactions.csv')
# df1 = interaction_df('md_100_040.pdb', 'i.csv')
# print(df1)
# df2 = intmap('interactions.csv')
# timemap(df2, savename='timemap.png', title='Nirmatrelvir')
####################################

# intmap(dfcov, 'int_cov.png')
# intmap('int_noncov.txt', 'int_noncov.png')

# pdblist = glob.glob1(os.getcwd(), '*sdf')

# dfsmiles = lig2smile('*sdf')
# prepare_lig('smiles.txt', image=True)

# # reclig_splitter('7L11')
# prepare_rec('7L11_receptor')

# prepare_lig('hits.txt', image=True)

###
# while read f; do vina --receptor ${f}_receptor.pdbqt --ligand ${f}_ligand.pdbqt --config docking.conf; done < targets.txt
###


