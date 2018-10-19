from __future__ import print_function

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

try:
	import cPickle as p
except ModuleNotFoundError:
	import pickle as p

import matplotlib.pyplot as plt
import warnings
import math
import numpy as np
import os
import sys
from tqdm import tqdm
import copy
import time
from random import shuffle
from collections import defaultdict
import random
# sklearn imports
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from keras.layers.advanced_activations import LeakyReLU

# keras imports
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint


# warnings suck, turn thme off
if sys.version_info[0] < 3:
	warnings.simplefilter("ignore", DeprecationWarning)
	with warnings.catch_warnings():
		warnings.filterwarnings("ignore", category=DeprecationWarning)
		import md5, sha


# TODO: most of these fields are redundant in our application
class CpGBin():
	""" 
	Constructor for a bin

	Inputs:

		matrix: numpy array, the bin's CpG matrix.
		binStartInc: integer, the starting, inclusive, chromosomal index of the bin.
		binEndInc: integer, the ending, inclusive, chromosomal index of the bin.
		cpgPositions: array of integers, the chromosomal positions of the CpGs in the bin.
		sequence: string, nucleotide sequence (A,C,G,T)
		encoding: array, a reduced representation of the bin's CpG matrix
		missingToken: integer, the token that represents missing data in the matrix.
		chromosome: string, the chromosome this bin resides in.
		binSize: integer, the number of base pairs this bin covers
		species: string, the speices this bin belongs too.
		verbose: boolean, print warnings, set to "false" for no error checking and faster speed

		tag1: anything, for custom use.
		tag2: anything, for custom use.
	"""
	def __init__(self, 
			matrix, 
			#relative_positions
			binStartInc=None, 
			binEndInc=None, 
			cpgPositions=None, 
			sequence="",
			encoding=None, 
			missingToken= -1, 
			chromosome=None, 
			binSize=100, 
			species="MM10", 
			verbose=True, 
			tag1=None, 
			tag2=None):


		self.cpgDensity = matrix.shape[1]
		self.readDepth = matrix.shape[0]
		
		

		self.matrix = np.array(matrix, dtype=float)
		self.binStartInc = binStartInc
		self.binEndInc = binEndInc
		self.cpgPositions = cpgPositions
		self.sequence = sequence
		self.missingToken = missingToken
		self.chromosome = chromosome
		self.binSize = binSize
		self.species = species
		self.tag1 = tag1
		self.tag2 = tag2




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


	def train(self, bin_matrices, weight_file="CpGNetWeights.h5"):
		# bin_matrices: a list of cpg matrices 
		bins = []

		# convert to bin objects for ease of use
		for matrix in bin_matrices:
			mybin = CpGBin(matrix=matrix)
			bins.append(mybin)
		
		# find bins with no missing data
		complete_bins = _filter_missing_data( bins )
		shuffle(complete_bins)
		
		# apply masks
		masked_bins = _apply_masks( complete_bins, bins )

		# extract features
		X, y = self._collectFeatures( masked_bins ) 

		# Train the neural network model
		self.fit(X, y, weight_file = weight_file)


	def fit(self,
			X_train,
			y_train,
			epochs=10,
			batch_size=32,
			val_split=0.2,
			weight_file="CpGNetWeights.h5"
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

		# Hidden layers
		self.model.add(Dense(1000, activation='linear',input_dim=x_input_dim))
		self.model.add(LeakyReLU(alpha=.001))
		self.model.add(Dropout(0.5))

		self.model.add(Dense(800, activation='linear',input_dim=x_input_dim))
		self.model.add(LeakyReLU(alpha=.001))
		self.model.add(Dropout(0.5))

		self.model.add(Dense(500, activation='linear'))
		self.model.add(LeakyReLU(alpha=.001))
		self.model.add(Dropout(0.5))

		self.model.add(Dense(100, activation='linear'))
		self.model.add(LeakyReLU(alpha=.001))
		self.model.add(Dropout(0.5))

		# Output layer predicts methylation status of a single CpG
		self.model.add(Dense(1, activation='sigmoid'))

		adam = keras.optimizers.Adam(lr=0.00001)

		self.model.compile(optimizer=adam,
					  loss='binary_crossentropy',
					  metrics=['accuracy'])

		earlystopper = EarlyStopping(patience=5, verbose=1)
		checkpointer = ModelCheckpoint(weight_file, monitor='val_acc', verbose=1, save_best_only=True, mode="max")

		# Displays the model's structure
		print (self.model.summary())


		return self.model.fit(X_train, y_train, 
			epochs=epochs, 
			batch_size=batch_size, 
			callbacks=[earlystopper, checkpointer], 
			validation_split=val_split, 
			verbose=True, 
			shuffle=True)









	# Return a vector of predicted classes 
	def predict_classes(self, X):
		"""
		Inputs: 
		1. X, numpy array, contains feature vectors
		
		Outputs: 
		1. 1-d numpy array of prediction values

		Usage: 
		y_pred = CpGNet.predict_classes(X)  

		"""
		return self.model.predict_classes(X)
	
	# Return a vector of probabilities for methylation
	def predict(self, X):
		"""
		Inputs: 
		1. X, numpy array, contains feature vectors
		
		Outputs: 
		1. 1-d numpy array of predicted class labels

		Usage: 
		y_pred = CpGNet.predict(X)  

		"""
		return self.model.predict(X)


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

	

	# TODO: vectorize this computation?
	# Imputes missing values in Bins
	def impute(self, matrix):
		"""
		Inputs: 
		1. matrix, a 2d np array, dtype=float, representing a CpG matrix, 1=methylated, 0=unmethylated, -1=unknown
		
		Outputs: 
		1. A 2d numpy array with predicted probabilities of methylation

		"""
		X = []

		numReads = matrix.shape[0]
		density = matrix.shape[1]

		nan_copy = np.copy(matrix)
		nan_copy[nan_copy == -1] = np.nan
		column_means = np.nanmean(nan_copy, axis=0)
		row_means = np.nanmean(nan_copy, axis=1)
		
		encoding = self._encode_input_matrix(matrix)[0]

		for i in range(numReads):
			for j in range(density):
				observed_state = matrix[i, j]
				
				if observed_state != -1:
					continue

				# encoding is the matrix encoding vector
				# encoding = self._encode_input_matrix(observed_matrix)[0]

				# # record the relative differences in CpG positions
				# rel_pos = []
				# rel_pos.append((positions[0] - bin_start) / 100.0)  ## distance to left bin edge
				# for pos in positions:
				# 	rel_pos.append((pos - bin_start) / 100.0)
				# rel_pos.append((bin_end - positions[-1] + 1) / 100.0)  ## distance to left bin edge

				row_mean = row_means[i]
				col_mean = column_means[j]
				# j is the current index in the row
				# M[] is the current row data
				row = np.copy(matrix[i])
				row[j] = -1
				# data = [j]  + list(encoding) + differences
				#data = [row_mean] + [col_mean] + rel_pos + [row_mean] + [col_mean] + [j] + list(row)  # list(encoding)
				data = [row_mean] + [col_mean] +  [i, j] + list(row) +  list(encoding)
				X.append(data)

		X = np.array(X)
		predictions = self.predict(X)
		k = 0 # keep track of prediction index for missing states
		predicted_matrix = np.copy(matrix)
		for i in range(predicted_matrix.shape[0]):
			for j in range(predicted_matrix.shape[1]):
				if predicted_matrix[i, j] == -1:
					predicted_matrix[i, j] = predictions[k]
					k += 1

		return predicted_matrix







	### Helper functions, for private use only ###

	# Returns a matrix encoding of a CpG matrix
	def _encode_input_matrix(self, m):
		matrix = np.copy(m)
		n_cpgs = matrix.shape[1]
		matrix += 1  # deal with -1s
		base_3_vec = np.power(3, np.arange(n_cpgs - 1, -1, -1))
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
		print ("encoded vector:",encoded_vector)
		encoded_vector_norm = normalize([encoded_vector], norm="l1")
		print ("encoded vector normalized",encoded_vector_norm)
		return encoded_vector_norm[0], num_reads

	# finds the majority class of the given column, discounting the current cpg
	
	def _get_column_mean(self, matrix, col_i, current_cpg_state):
		sub = matrix[:, col_i]
		return self._get_mean(sub, current_cpg_state)

	# finds the majority class of the given read, discounting the current cpg
	def _get_read_mean(self, matrix, read_i, current_cpg_state):
		sub = matrix[read_i, :]
		return self._get_mean(sub, current_cpg_state)

	def _get_mean(self, sub_matrix, current_cpg_state):
		num_methy = np.count_nonzero(sub_matrix == self.METHYLATED)
		num_unmethy = np.count_nonzero(sub_matrix == self.UNMETHYLATED)

		if current_cpg_state == self.METHYLATED:
			num_methy -= 1
		num_methy = max(0, num_methy)
		if current_cpg_state == self.UNMETHYLATED:
			num_unmethy -= 1
		num_unmethy = max(0, num_unmethy)
		if float(num_methy + num_unmethy) == 0:
			return -2

		return float(num_methy) / float(num_methy + num_unmethy)

	# Returns X, y
	# note: y can contain the labels 1,0, -1
	def _collectFeatures(self, bins):
		X = []
		Y = []
		for Bin in tqdm(bins):
			observed_matrix = Bin.tag2["observed"]
			truth_matrix = Bin.tag2["truth"]
			encoding = self._encode_input_matrix(observed_matrix)[0]

			numReads = observed_matrix.shape[0]
			density = observed_matrix.shape[1]
			#positions = Bin.cpgPositions
			nan_copy = np.copy(observed_matrix)
			

			nan_copy[nan_copy == -1] = np.nan 
			column_means = np.nanmean(nan_copy,axis=0)
			row_means = np.nanmean(nan_copy,axis=1)

			for i in range(numReads):
				for j in range(density):
					observed_state = observed_matrix[i,j]
					if observed_state != -1:
						continue

					state = truth_matrix[i,j]
					Y.append(state)
					
					
					# # record the relative differences in CpG positions
					# rel_pos = []
					# rel_pos.append((positions[0] - Bin.binStartInc)/100.0) ## distance to left bin edge
					# for pos in positions:
					#     rel_pos.append((pos - Bin.binStartInc)/100.0)
					# rel_pos.append((Bin.binEndInc - positions[-1] + 1)/100.0) ## distance to left bin edge

					#rel_pos = Bin.relative_positions
					row_mean = row_means[i]
					col_mean = column_means[j]
					# j is the current index in the row
					# M[] is the current row data
					# encoding is the matrix encoding vector
					# differences is the difference in positions of the cpgs
					row = np.copy(observed_matrix[i])
					row[j] = -1

					#data = [row_mean] + [col_mean] + rel_pos +  [i, j] + list(row) +  list(encoding)
					data = [row_mean] + [col_mean] +  [i, j] + list(row) +  list(encoding)
					X.append(data)


		X = np.array(X)
		Y = np.array(Y)
		Y.astype(int)
		return X, Y

# returns a list of bins similar to the input
# but matrix rows with missing values are removed
def _filter_bad_reads(bins):
	filtered_bins = []
	for Bin in bins:
		newBin = copy.deepcopy(Bin)
		matrix = newBin.matrix

		# find rows with missing values
		counts = np.count_nonzero(matrix == -1, axis=1)
		idx = counts == 0
		matrix_filtered = matrix[idx]
		newBin.matrix = matrix_filtered
		filtered_bins.append(newBin)

	return filtered_bins

# returns a mapping of dimensions to list of masks that can be used on data
# of that size.
# the missing pattern is in matrix form.
# -1 is missing, 2 is known
def _extract_masks( bins):
	masks = defaultdict(lambda: [])
	for Bin in tqdm(bins):
		matrix = np.copy(Bin.matrix)
		matrix[matrix >= 0] = 2

		#min_missing = 10
		min_missing = 1 # must have at least 1 missing value
		if np.count_nonzero(matrix == -1) >= min_missing:
			masks[matrix.shape].append(matrix)

	return masks

  
def _apply_masks( filtered_bins, all_bins ):
	
	masks = _extract_masks( all_bins )
	ready_bins = []

	for Bin in filtered_bins:
		truth_matrix = Bin.matrix
		m_shape = truth_matrix.shape
		if m_shape in masks:
			if len( masks [ m_shape ] ) > 0:
				mask = random.choice(masks[m_shape])
				observed = np.minimum(truth_matrix, mask)
				Bin.tag2 = {"truth":truth_matrix, "observed":observed, "mask":mask}
				ready_bins.append(Bin)

	return ready_bins

# get a set of bins with no missing data
def _filter_missing_data( bins, min_read_depth=1 ):
	cpg_bins_complete = _filter_bad_reads(bins)
	# secondary depth filter
	cpg_bins_complete_depth = [bin_ for bin_ in cpg_bins_complete if bin_.matrix.shape[0] >= min_read_depth]
	return cpg_bins_complete_depth


