"""
Author: Jack Duryea
November, 2018
Waterland Lab
Computational Epigenetics Section
Baylor College of Medicine

Converts bins to DSS format,
here's an example:

chr	pos	N	X
chr12	10588	4	1
chr12	10589	21	16
chr12	10601	5	4
chr12	10602	26	22
chr12	10606	7	6
chr12	10607	26	23
...


where chr is the chromosome,
ps is the position, N is the number of reads covering
that CpG, and X is the number of methylated reads at that
position.


Example Usage:
python bins_to_DSS.py 19 /home/jduryea/CpGNet/data/neuron_ListerMukamel_GSM1173786_mm_neun_pos/bin_data/NEUN_POSchr19_2CpGbins.p > outfile.txt


python bins_to_DSS.py 19 /home/jduryea/CpGNet/data/neuron_ListerMukamel_GSM1173786_mm_neun_pos/bin_data/NEUN_POSchr19_2CpGbins.p > imputed_lm_neuron_chr19_dss.txt


"""
# 

from __future__ import print_function
import sys
sys.path.append('/home/jduryea/CpGNet/PRELIM')

sys.path.append('/home/jduryea/CpGNet/util')
#sys.path.append('/home/jduryea/CpGNet/model')
from PReLIM import PReLIM
#from CpG_Net import CpGNet
#from CpG_Net import CpG_Bin
from tqdm import tqdm
import numpy as np
import cPickle as p
from PReLIM import CpGBin

chr_number = sys.argv[1]
density = sys.argv[2]

# header
print("chr\tpos\t\tN\tX")

#python bins_to_DSS.py 19 /home/jduryea/CpGNet/data/glia_ListerMukamel_GSM1173787_mm_neun_neg/bin_data/NEUN_NEG_chr19bins.p
# "python bins_to_DSS.py 12 /home/jduryea/CpGNet/data/neuron_ListerMukamel_GSM1173786_mm_neun_pos/bin_data/NEUN_POSchr12_2CpGbins.p > dss_data/imputed_lm_neuron_2cpgs_dss_data_chr12.txt"

# "python bins_to_DSS.py 1 3 > dss_data/imputed_lm_neuron_2cpgs_dss_data_chr12.txt"


data_file = "/home/jduryea/CpGNet/data/glia_ListerMukamel_GSM1173787_mm_neun_neg/bin_data/NEUN_NEGchr"+chr_number+"_"+density+"CpGbins.p"
#data_file = "/home/jduryea/CpGNet/data/neuron_ListerMukamel_GSM1173786_mm_neun_pos/bin_data/NEUN_POSchr"+chr_number+"_"+density+"CpGbins.p"
model_file = "glia_"+density+"_cpgs_prelim_model"
#model_file = "/home/jduryea/CpGNet/models/glia_models/glia"+density+"cpg.h5"
#model_file = "/home/jduryea/CpGNet/models/neuron_models/neuron"+density+"cpg.h5"
#print("data file:" + data_file)
#print("model file:" + model_file)
# load bins
#print("loading")
bins = p.load(open(data_file,"r"))
#print("done loading")
#python bins_to_DSS.py 19 /home/jduryea/CpGNet/data/neuron_ListerMukamel_GSM1173786_mm_neun_pos/bin_data/NEUN_POSchr19bins.p
# CpG Net
#CPG_DENSITY = 2

#model_file = "/home/jduryea/CpGNet/scripts/train_scripts/lm_neuron/NEUN_POS_CpGNetWeights_chr"+chr_number+"_density2.h5"
#model_file = "/home/jduryea/CpGNet/scripts/train_scripts/lm_glia/NEUN_NEG_CpGNetWeights_chr"+chr_number+"_density2.h5"

net = PReLIM(density)
#net = None
net.loadWeights(model_file)

# whether or not to use imputation, alonge with our minimum require confidence
use_imputation = True
confidence = 0.6

#print("imputing")

# impute more efficiently
imputed_matrices = net.impute_many([bin_.matrix for bin_ in bins])
#print("done imputing")

for i in range(len(bins)):
	bin_ = bins[i]
	matrix = bin_.matrix
	if use_imputation and np.count_nonzero(matrix==-1) > 0:
		imputed_matrix = imputed_matrices[i]
		
		matrix = imputed_matrix

		# filter out confidence levels that are too low
		matrix[np.sqrt( np.square((matrix-0.5)*2)) < confidence ]=-1
		matrix = np.round(matrix)
		# print ("imputed_matrix:", matrix)
		# print ("original matrix:", bin_.matrix)
		# print("----------------")

	positions = bin_.cpgPositions
	if len(positions) != matrix.shape[1]:
		print("oops")
		continue

	for coln in range(len(positions)):
		col = matrix[:,coln]
		x_methy_reads =  np.sum(col[col!=-1])
		n_total_reads = np.count_nonzero(col!=-1)
		line_data = "chr" + chr_number + "\t" + str(int(positions[coln])) + "\t\t" + str(int(n_total_reads)) + "\t" + str(int(x_methy_reads))
		print (line_data)
# For each bin, get the matrix and compute the DSS required data for each CpG
# for bin_ in tqdm(bins):
# 	matrix = bin_.matrix
# 	# should use imputation only when specified and more than one missing cpg
# 	if use_imputation and np.count_nonzero(matrix==-1) > 0:
# 		#print ("before imputing:")
# 		#print (matrix)
# 		#print("--------------------------")
# 		imputed_matrix = net.impute(matrix)
		
# 		matrix = imputed_matrix
# 		# filter out confidence levels that are too low
# 		matrix[np.sqrt( np.square((matrix-0.5)*2)) < confidence ]=-1
# 		matrix = np.round(matrix)
		#print("after imputation:")
		#print (matrix)

	# positions = bin_.cpgPositions
	# if len(positions) != matrix.shape[1]:
	# 	print("oops")
	# 	continue

	# for coln in range(len(positions)):
	# 	col = matrix[:,coln]
	# 	x_methy_reads =  np.sum(col[col!=-1])
	# 	n_total_reads = np.count_nonzero(col!=-1)
	# 	line_data = "chr" + chr_number + "\t" + str(int(positions[coln])) + "\t\t" + str(int(n_total_reads)) + "\t" + str(int(x_methy_reads))
	# 	print (line_data)










