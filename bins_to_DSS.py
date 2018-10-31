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
"""
# 

from __future__ import print_function
import sys
sys.path.append('/home/jduryea/CpGNet/util')
from CpG_Bin import Bin

import numpy as np
import cPickle as p


chr_number = sys.argv[1]
data_file = sys.argv[2]


# load bins
bins = p.load(open(data_file,"r"))

# header
print("chr\tpos\t\tN\tX")

# For each bin, get the matrix and compute the DSS required data for each CpG
for bin_ in bins:
	matrix = bin_.matrix
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










