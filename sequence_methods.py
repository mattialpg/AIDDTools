import os, glob, math
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt

def fasta_from_pdb(pdblist, consensus=None):
	"""
	pdblist  (list): list of pdb files
	consensus (str): consensus sequence
	OUTPUT   (file): text file with all the fasta sequencies 
	"""
	from textwrap import wrap
	from Bio import SeqIO
	from natsort import natsorted

	if '*' in pdblist: pdblist = natsorted([x for x in glob.glob1(os.getcwd(), pdblist)])
	else: pdblist = [pdblist] if isinstance(pdblist, str) else pdblist
	
	with open('seq_all.fasta', 'w') as f1, open('seq_mut.fasta', 'w') as f2:
		if consensus is not None:
			# Write consensus seq. in the full-sequencies file 
			txtcons = wrap(str(consensus), width=80)
			f1.write('>CONS \n')
			f1.write('\n'.join(txtcons) + '\n')
			# Write consensus seq. in the mutated-sequencies file
			f2.write('>CONS \n')
			f2.write('\n'.join(txtcons) + '\n')
		
		for pdb in pdblist:
			fasta = [str(x.seq) for x in list(SeqIO.parse(pdb, 'pdb-atom'))]
			txtfasta = ''.join(fasta)
			wrapfasta = wrap(txtfasta, width=80)
			f1.write('>' + pdb.split('.')[0] + '\n')
			f1.write('\n'.join(wrapfasta) + '\n')
			
			if txtfasta != consensus:
				f2.write('>' + pdb.split('.')[0] + '\n')
				f2.write('\n'.join(wrapfasta) + '\n')
	return

def plot_alignment(fasta_file, type='mutations'):
	"""
	Plot multiple sequence alignment
	To include the custom color scheme "tcoffee" copy the json file to
	C:/Users/Mattia/anaconda3/envs/my-chem/Lib/site-packages/biotite/sequence/graphics/color_schemes
	"""
	import biotite.sequence as seq
	import biotite.sequence.io.fasta as fasta
	import biotite.sequence.align as align
	import biotite.sequence.graphics as graphics
	from matplotlib.colors import LinearSegmentedColormap

	fasta_file = fasta.FastaFile.read(fasta_file)
	sequences = [seq.ProteinSequence(seq_str) for seq_str in fasta_file.values()]
	matrix = align.SubstitutionMatrix.std_protein_matrix()
	names = [header.replace('>','').split('|')[0] if '|' in header else header.replace('>','')\
				for header, string in fasta_file.items()]

	# Add custom font to matplotlib
	from matplotlib import font_manager
	font_dirs = ['C:/Users/Mattia/MEGA/DEVSHEALTH/Algoritmi/drug_design/condensed_font']
	font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
	for font_file in font_files:
		font_manager.fontManager.addfont(font_file)
	plt.rcParams['font.family'] = 'Consolas'

	# alignment, order, tree, distances = 
	A = align.align_multiple(sequences, matrix)
	# alignment = alignment[:, order]	# Order alignment according to the guide tree
	print(A[0][0]);exit()

	lines = min(25,len(sequences))	# number of lines in each plot
	for i in range(0,len(sequences),lines):
		fig, ax = plt.subplots(1, figsize=(16,9))
		
		# Plot alignment with highlighted mutations
		if type == 'mutations':
			# Box color interpolated between reddish and white
			cmap = LinearSegmentedColormap.from_list('custom', colors=[(1.0, 0.3, 0.3), (1.0, 1.0, 1.0)])
			graphics.plot_alignment_similarity_based(ax, alignment,#labels=names[i:i+lines], 
												symbols_per_line=math.ceil(len(alignment)/2),
												cmap=cmap, label_size=13, symbol_size=13, 
												show_numbers=True)
		# Plot alignment with residue-based colours
		if type == 'residues':
			graphics.plot_alignment_type_based(ax, alignment[i:i+lines], labels=names[i:i+lines], 
												symbols_per_line=155, color_scheme='tcoffee',
												label_size=13, symbol_size=13, #line_spacing=0.5,
												show_line_position=True, color_symbols=True)

		fig.tight_layout()
		plt.savefig('fasta_' + str(int(i/lines)) + '.png', dpi=400)