Jack Duryea (duryea@bcm.edu)   
Waterland Lab  
Computational Epigenetics Section  
Baylor College of Medicine  

April 2018


# CpGNet
A neural network approach to CpG imputation

## CpG_Bin
A python object that stores information about CpG bins


## Tutorial

## 1. How to create a bin
cpgMatrix = np.array([[1,1],[-1,0],[1,1],[0,0],[1,-1]])  
binStartInc = 100 # left most genomic position  
binEndInc = 199 # right most genomic position  
binSize = 100 # bin width (in base pairs)  

mybin = Bin(matrix=cpgMatrix, binStartInc=100, binEndInc= 199, cpgPositions=[105, 110], chromosome="19", binSize=100, verbose=True, tag1="Created for a demo")  


## 2. Get matrix and metadata
print "CpG matrix:       \n", mybin.matrix  
print "leftmost position:  ", mybin.binStartInc  
print "rightmost position: ", mybin.binEndInc  
print "CpG Density:        ", mybin.cpgDensity  
print "Read Depth:         ", mybin.readDepth  
print "Chromosome:         ", mybin.chromosome  
print "Species:            ", mybin.species  
print "Custom tag:         ", mybin.tag1  


## 3. Saving and loading 
pickle.dump(mybin, open("binfile.p","wb"))  
binFromDisk = pickle.load(open("binfile.p","rb"))  
  

## 4. Feature extraction  and CpGNet training
bins = [mybin]# more bins]  
net = CpGNet(cpgDensity=2)  
X, y = net.collectFeatures(bins) # extract features  

notMissing = y!=-1  
X_train = X[notMissing]     
y_train = y[notMissing]  

net.fit(X_train, y_train, weight_file ="jack-april-2018-chr19-mm10.h5", epochs=1000)  

## 5. Imputation
from CpG_Net import CpGNet
from CpG_Bin import Bin
import numpy as np 

DENSITY = 5
net = CpGNet(cpgDensity=DENSITY)  
net.loadWeights("CpGNet_5cpg_weights.h5")
### the cpg matrix
matrix = np.array([[0,0,0,1,-1],[0,0,0,0,0]],dtype=float)

### positions of each cpg
pos = np.array([1002,1004,1040,1050,1070])

### left most position of bin
bin_start_pos = 1000

### right most position of bin
bin_end_pos = 1100 

### make the imputation, 
predicted_matrix = net.impute(matrix, pos, bin_start_pos, bin_end_pos)
