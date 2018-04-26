"""
Author: 
Jack Duryea
Waterland Lab
Computational Epigenetics Section
Baylor College of Medicine

April 2018

CpG-Net imputes missing CpG methylation
states in CpG matrices.

"""

# standard imports
from scipy import stats
import pandas as pd
import numpy as np
import cPickle as p
import matplotlib.pyplot as plt
import warnings
import math
import numpy as np
import os
import sys
from tqdm import tqdm


# sklearn imports
from sklearn.preprocessing import normalize 
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures


# keras imports
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

# bin data type
from CpG_Bin import Bin

# warnings suck, turn thme off
warnings.simplefilter("ignore", DeprecationWarning)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    import md5, sha




class CpGNet():
	def __init__(self, cpgDensity=2):
		self.model = None
		self.cpgDensity = cpgDensity
		self.METHYLATED = 1
		self.UNMETHYLATED = 0
		self.MISSING = -1
		self.methylated = 1
		self.unmethylated = 0
		self.unknown = -1

	def fit(self, 
			X_train, 
			y_train, 
			epochs=10,
			batch_size=32,
			val_split=0.2,
			weight_file = "CpGNetWeights2.h5"
			):
		"""
		Inputs: 
		1. X_train,     numpy array, Contains feature vectors.
		2. y_train,     numpy array, Contains labels for training data.
		3. epochs,      integer,     Number of epochs to train for. Can be cut short if using early stopping.
		4. batch_size,  integer,     Size of each training batch.
		5. val_split,   float        Between 0 and 1, the proportion of X_train that is used for validation loss.
		6. weight_file, string,      The name of the file to save the model weights to.
	
		Outputs: 
		None, saves a weight file

		Usage: 
		CpGNet.fit(X_train, y_train)	
		
		"""

		x_input_dim = X_train.shape[1]

		self.model = Sequential()
		self.model.add(Dense(10, activation='relu',input_dim=x_input_dim))
		self.model.add(Dropout(1.0))

		self.model.add(Dense(10, activation='relu', input_dim=10))
		self.model.add(Dropout(1.0))

		#output
		self.model.add(Dense(1, activation='sigmoid'))

		self.model.compile(optimizer='adam',
		              loss='binary_crossentropy',
		              metrics=['binary_crossentropy'])
		earlystopper = EarlyStopping(patience=5, verbose=1)
		
		checkpointer = ModelCheckpoint('CpGNetWeights', monitor='val_loss', verbose=1, save_best_only=True, mode="min")

		self.model.fit(X_train, y_train, 
			epochs=epochs, 
			batch_size=batch_size, 
			#callbacks=[earlystopper, checkpointer], 
			validation_split=val_split, 
			verbose=True, 
			shuffle=True)


	# Load a saved model 
	def loadWeights(self, weight_file):
		"""
		Inputs:
		1. weight_file, string, name of file with saved model weights

		Outputs:
		None

		Effects:
		self.model is loaded with the provided weights
		"""
		self.model = load_model(weight_file)

	# Make a prediction on the provided feature vectors using the trained model
	def predict(self, X):
		"""
		Inputs: 
		1. X, numpy array, contains feature vectors
		
		Outputs: 
		1. 1-d numpy array of predicted labels

		Usage: 
		y_pred = CpGNet.predict(X)	

		"""
		return self.model.predict(X)


	def impute(self, Bins, confidence_threshold=0.1):
		"""
		Inputs: 
		1. Bins, list, contains CpG_Bins
		2. confidence_threshold, float in [0,1], the required confidence of each imputation
			defined as twice the distance from 0.5 (the decision boundary).

		Outputs: 
		1. List of CpG_Bins with missing data imputed

		Usage: 
		CpGNet.impute()	

		"""
		num_succesful_imputations = 0
		num_failed_imputations = 0

		for Bin in Bins:
			matrix = Bin.matrix
			features = self.collectFeatures([Bin])
			print "features:",features[0]
			pred = self.predict(features[0])
			pred_i = 0
			# look at each value in the matrix
			for i in range(matrix.shape[0]):
				for j in range(matrix.shape[1]):

					state = matrix[i][j]
					print "state: ",state
					print "pred:  ",pred[pred_i]
					if state == self.MISSING:
						if np.abs(pred[pred_i] - 0.5) * 2 > confidence_threshold:
							matrix[i][j] = np.round(pred[pred_i])
							num_succesful_imputations += 1
						else:
							num_failed_imputations += 1


					pred_i += 1

		return num_succesful_imputations, num_failed_imputations









	### Helper functions, for private use only ###

	# Returns a matrix encoding of a CpG matrix
	def encode_input_matrix(self,m):
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





	# finds the majority class of the given column, discounting the current cpg
	def get_column_mean(self, matrix, col_i, current_cpg_state):
	    sub=matrix[:,col_i]
	    return self.get_mean(sub, current_cpg_state)

	# finds the majority class of the given read, discounting the current cpg
	def get_read_mean(self, matrix, read_i, current_cpg_state):
	    sub=matrix[read_i,:]
	    return self.get_mean(sub, current_cpg_state)

	def get_mean(self, sub_matrix, current_cpg_state):
	        num_methy = np.count_nonzero(sub_matrix == self.METHYLATED)
	        num_unmethy = np.count_nonzero(sub_matrix == self.UNMETHYLATED)
	        
	        if current_cpg_state == self.METHYLATED:
	            num_methy -= 1
	        num_methy = max(0,num_methy)
	        if current_cpg_state == self.UNMETHYLATED:
	            num_unmethy -= 1
	        num_unmethy = max(0,num_unmethy )
	        if float(num_methy + num_unmethy)==0:
	            return -2

	        return float(num_methy)/float(num_methy + num_unmethy)
	        
	
	
	# Returns X, y
	# note: y can contain the labels 1,0, -1
	def collectFeatures(self, bins):
		X = []
		Y = []
		for Bin in bins:
			M = Bin.matrix
			numReads = M.shape[0]
			density = M.shape[1]
			positions = Bin.cpgPositions

			col_means = np.nanmean(np.where(M != -1, M, np.nan), axis=0)
			row_means = np.nanmean(np.where(M != -1, M, np.nan), axis=1)
			for i in range(numReads):
				for j in range(density):
					state = M[i,j]
					Y.append(state)

					cur_col_mean = self.get_column_mean(M, j, -1)

					read_mean = self.get_read_mean(M, i, state)

					adjacent_state_mean = None
					if j == 0: # left edge
						adjacent_state_mean = M[i, 1]
					elif j == density-1: # right edge
						adjacent_state_mean = M[i, density-2]
					else:
						adjacent_state_mean = (M[i, j-1] + M[i, j+1])/float(2)

					encoding = self.encode_input_matrix(M)[0]

					# # record the relative differences in CpG positions
					differences = []
					for pos in range(1, density):
						differences.append(positions[pos]-positions[pos-1])
					data = [adjacent_state_mean, cur_col_mean, read_mean] + list(encoding)+ differences # + list(encoding)
					X.append(data)


		X = np.array(X)
		Y = np.array(Y)
		Y.astype(int)
		return X, Y





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




















