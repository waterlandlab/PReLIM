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
"""


from __future__ import print_function
from CpG_Net import CpGBin
import sys
import numpy as np
import cPickle as p


chr_number = sys.argv[1]
data_file = sys.argv[2]


# load bins
bins = p.load(open(data_file,"r"))

# header
print("chr\tpos\tN\tX")

# For each bin, get the matrix and compute the DSS required data for each CpG
for bin_ in bins[:100]:
	matrix = bin_.matrix
	positions = bin_.cpgPositions
	if len(positions) != matrix.shape[1]:
		print("oops")
		continue

	for coln in range(len(positions)):
		col = matrix[:,coln]
		x_methy_reads =  np.sum(col[col!=-1])
		n_total_reads = np.count_nonzero(col!=-1)
		line_data = "chr" + chr_number + "\t" + str(positions[i]) + "\t" + str(n_total_reads) + "\t" + str(x_methy_reads)
		print (line_data)










