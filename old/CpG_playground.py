"""
Author: 
Jack Duryea
Waterland Lab
Computational Epigenetics Section
Baylor College of Medicine

April 2018

Examples of how to work with CpG_Net and CpG_Bin

"""

from CpG_Net import CpGNet
from CpG_Bin import Bin
import numpy as np
import cPickle as pickle



# 1. How to create a bin
cpgMatrix = np.array([[1,1,0],[-1,0,0],[1,1,0],[0,0,0],[1,-1,0]])
binStartInc = 100 # left most genomic position
binEndInc = 199 # right most genomic position
binSize = 100 # bin width (in base pairs)

mybin = Bin(matrix=cpgMatrix, binStartInc=100, binEndInc= 199, cpgPositions=[105, 110, 120], chromosome="19", binSize=100, verbose=True, tag1="Created for a demo")


# 2. Get matrix and metadata
print "CpG matrix:       \n", mybin.matrix
print "leftmost position:  ", mybin.binStartInc
print "rightmost position: ", mybin.binEndInc
print "CpG Density:        ", mybin.cpgDensity
print "Read Depth:         ", mybin.readDepth
print "Chromosome:         ", mybin.chromosome
print "Species:            ", mybin.species
print "Custom tag:         ", mybin.tag1





# 5. Imputation example in a bin with 5 cpgs and 2 reads

DENSITY = 5
net = CpGNet(cpgDensity=DENSITY)
net.loadWeights("CpGNet_3cpg_weights.h5")
# the cpg matrix
matrix = np.array([[0,0,0,1,-1],[0,0,0,0,0]],dtype=float)

# positions of each cpg
pos = np.array([1002,1004,1040,1050,1070])

# left most position of bin
bin_start_pos = 1000

# right most position of bin
bin_end_pos = 1100 

# make the imputation, 
predicted_matrix = net.impute(matrix, pos, bin_start_pos, bin_end_pos)

# example output
# array([[ 0.        ,  0.        ,  0.        ,  0.        ,  0.04091179],
#        [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ]])








