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
cpgMatrix = np.array([[1,0],[0,1],[-1,1],[0,-1],[1,1]])
binStartInc = 100 # left most genomic position
binEndInc = 199 # right most genomic position
binSize = 100 # bin width (in base pairs)

mybin = Bin(matrix=cpgMatrix, binStartInc=100, binEndInc= 199, cpgPositions=[105, 110], chromosome="19", binSize=100, verbose=True, tag1="Created for a demo")


# 2. Get matrix and metadata
print "CpG matrix:       \n", mybin.matrix
print "leftmost position:  ", mybin.binStartInc
print "rightmost position: ", mybin.binEndInc
print "CpG Density:        ", mybin.cpgDensity
print "Read Depth:         ", mybin.readDepth
print "Chromosome:         ", mybin.chromosome
print "Species:            ", mybin.species
print "Custom tag:         ", mybin.tag1


# 3. Saving and loading 
pickle.dump(mybin, open("binfile.p","wb"))
binFromDisk = pickle.load(open("binfile.p","rb"))


# 4. Feature extraction 
bins = [mybin]# more bins]
net = CpGNet(cpgDensity=2)
X, y = net.collectFeatures(bins) # extract features

print "X:",X
print "Y:",y










