"""
Author: 
Jack Duryea
Waterland Lab
Baylor College of Medicine

CpG-Net imputes missing CpG methylation
states in CpG matrices.

"""

from sklearn.preprocessing import Imputer
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import normalize 
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm

import math
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

from scipy import stats
import pandas as pd
import numpy as np
import cPickle as p
import matplotlib.pyplot as plt
import seaborn as sn
import warnings
from sklearn.model_selection import train_test_split


# bin data type
from CpG_Bin import Bin

warnings.simplefilter("ignore", DeprecationWarning)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    import md5, sha
import os
import sys




class CpGNet():
	def __init__(self, cpgDensity=2):
		self.model = None
		self.cpgDensity = cpgDensity
		# default weights


	def fit(self, X_train, y_train, 
			epochs=10,
			batch_size=32,
			val_split=0.2,
			weight_file = "CpGNetWeights2.h5"
			):

		x_input_dim = X_train.shape[1]

		self.model = Sequential()
		self.model.add(Dense(1000, activation='relu',input_dim=x_input_dim))
		self.model.add(Dropout(0.1))

		self.model.add(Dense(1000, activation='relu', input_dim=1000))
		self.model.add(Dropout(0.1))

		#output
		self.model.add(Dense(1, activation='hard_sigmoid'))

		self.model.compile(optimizer='adam',
		              loss='binary_crossentropy',
		              metrics=['binary_crossentropy'])
		earlystopper = EarlyStopping(patience=5, verbose=1)
		
		checkpointer = ModelCheckpoint('CpGNetWeights', monitor='val_loss', verbose=1, save_best_only=True, mode="min")

		self.model.fit(X_train, y_train, 
			epochs=epochs, 
			batch_size=batch_size, 
			callbacks=[earlystopper, checkpointer], 
			validation_split=val_split, 
			verbose=True, 
			shuffle=True)


	def predict(self, X):
		return self.model.predict(X)


	def impute(self, Bins):






	# Returns a matrix encoding of a CpG matrix
	def encode_input_matrix(m):
	    matrix = np.copy(m)
	    n_cpgs = matrix.shape[1]
	    matrix += 1 # deal wiht -1s
	    base_3_vec = np.power(3, np.arange(n_cpgs-1,-1,-1))
	    #
	    encodings = np.dot(base_3_vec, matrix.T)
	    #
	    encoded_vector_dim = np.power(3, n_cpgs)
	    encoded_vector = np.zeros(encoded_vector_dim)
	    #
	    for x in encodings:
	        encoded_vector[int(x)] += 1
	    #
	    num_reads = encodings.shape[0]
	    #
	    # Now we normalize
	    encoded_vector = normalize(encoded_vector, norm="l1")
	    return encoded_vector[0], num_reads


   	METHYLATED = 1
	UNMETHYLATED = 0
	MISSING = -1
	methylated = 1
	unmethylated = 0
	unknown = -1

	def get_mean(sub_matrix, current_cpg_state):
	        num_methy = np.count_nonzero(sub_matrix == METHYLATED)
	        num_unmethy = np.count_nonzero(sub_matrix == UNMETHYLATED)
	        
	        if current_cpg_state == METHYLATED:
	            num_methy -= 1
	        num_methy = max(0,num_methy)
	        if current_cpg_state == UNMETHYLATED:
	            num_unmethy -= 1
	        num_unmethy = max(0,num_unmethy )
	        if float(num_methy + num_unmethy)==0:
	            return -2

	        # # return based on prior
	        # if float(num_methy)/float(num_methy + num_unmethy) == 2:
	        #     print "yo"
	        #     print "num_methy: ",num_methy
	        #     print "num_unmethy: ", num_unmethy
	        return float(num_methy)/float(num_methy + num_unmethy)
	        
	# finds the majority class of the given column, discounting the current cpg
	def get_column_mean(matrix, col_i, current_cpg_state):
	    sub=matrix[:,col_i]
	    return get_mean(sub, current_cpg_state)

	# finds the majority class of the given read, discounting the current cpg
	def get_read_mean(matrix, read_i, current_cpg_state):
	    sub=matrix[read_i,:]
	    return get_mean(sub, current_cpg_state)
	
	# Returns X, y
	# note: y can contain the labels 1,0, -1
	def collectFeatures(self, bins):
		X = []
		Y = []
		for M_i in tqdm(range(len(bins))):
		    Bin = bins[M_i]
		    M = Bin.matrix
		    numReads = M.shape[0]
		    density = M.shape[1]
		    positions = Bin.positions
		    
		    col_means = np.nanmean(np.where(M != -1, M, np.nan), axis=0)
		    row_means = np.nanmean(np.where(M != -1, M, np.nan), axis=1)
		    for i in range(numReads):
		        for j in range(density):
		            state = M[i,j]
	                cur_col_mean = get_column_mean(M, j, -1)
	                
	                read_mean = get_read_mean(M, i, state)
	                
	                adjacent_state_mean = None
	                if j == 0: # left edge
	                    adjacent_state_mean = M[i, 1]
	                else if j == density-1: # right edge
						adjacent_state_mean = M[i, density-2]
	            	else:
	                    adjacent_state_mean = (M[i, j-1] + M[i, j+1])/float(2)

	                encoding = encode_input_matrix(M)[0]

	                # record the relative differences in CpG positions
	                differences = []
	                for i in range(1, density):
	                	differences.append(positions[i]-positions[i-1])
	                data = [adjacent_state_mean, cur_col_mean, read_mean] + list(encoding)+ differences # + list(encoding)
	                X.append(data)

	                Y.append(state)

		X = np.array(X)
		Y = np.array(Y)
		Y.astype(int)





#### Usage ###
"""
from CpGNet import CpGNet

net = CpGNet() # load the model
bins = loadData() # load the data, not part of CpGNet. "bins" is an array of CpG_bin objects


## For training
X, y = net.collectFeatures(bins) # extract features
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42) # train/test split
net.fit(X_train, y_train) # train


# For making validation predictions
X, y = net.collectFeatures(bins)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
net.fit(X_train, y_train)
predictions = net.predict(X_test) # make predictions
accuracy = np.sum(np.round(predictions) == y_test)/float(len(y_test)) # compute accuracy

## For imputing
ImputedBins = net.impute(bins)
"""





















