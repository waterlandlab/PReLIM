from __future__ import print_function

"""
Author: 
Jack Duryea
Waterland Lab
Computational Epigenetics Section
Baylor College of Medicine

PReLIM: Preceise Read Level Imputation of Methylation

PReLIM imputes missing CpG methylation
states in CpG matrices.

"""

# standard imports
from scipy import stats
import numpy as np
import warnings
import numpy as np
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Pickle
try:
	import cPickle as p
except ModuleNotFoundError:
	import pickle as p


# warnings suck, turn them off
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




class PReLIM():
	def __init__(self, cpgDensity=2):
		"""
        :param cpgDensity: the density of the bins that will be used 
        """

		self.model = None
		self.cpgDensity = cpgDensity
		self.METHYLATED = 1
		self.UNMETHYLATED = 0
		self.MISSING = -1
		self.methylated = 1
		self.unmethylated = 0
		self.unknown = -1





	# Train a model
	def train(self, bin_matrices, model_file="no", verbose=False):
		"""
        :param bin_matrices: list of cpg matrices
        :param model_file: The name of the file to save the model to. If None, then create a file name that includes a timestamp. If you don't want to save a file, set this to "no"
        :param verbose: prints more info if true
        """

		X,y = self.get_X_y(bin_matrices, verbose=verbose)
		
		# Train the neural network model
		self.fit(X,y, model_file=model_file, verbose=verbose)
		



	def fit(self,
			X_train,
			y_train,
			n_estimators = [10, 50, 100, 500, 1000],
			cores = -1,
			max_depths = [1, 5, 10, 20, 30],
			model_file=None,
			verbose=False
			):

		"""
		Train a random forest model using grid search on a feature matrix (X) and class labels (y)

		Usage: 
		model.fit(X_train, y_train)	

        :param X_train: numpy array, Contains feature vectors.
        :param y_train: numpy array, Contains labels for training data.
        :param n_estimators: list, the number of estimators to try during a grid search.
        :param max_depths: list, the maximum depths of trees to try during a grid search.
        :param cores: integer, the number of cores to use during training, helpful for grid search.
        :param model_file:  string,The name of the file to save the model to. 
			If None, then create a file name that includes a timestamp.
			If you don't want to save a file, set this to "no"

        :return: The trained sklearn model
        """

		grid_param = {  
		 "n_estimators": n_estimators,
		  "max_depth": max_depths,
		}

		# Note: let the grid search use a lot of cores, but only use 1 for each forest
		# since dispatching can take a lot of time
		rf = RandomForestClassifier(n_jobs=1)
		self.model = GridSearchCV(rf, grid_param, n_jobs=2, cv=5, verbose=verbose)
		self.model.fit(X_train, y_train)


		# save the model
		if model_file == "no":
			return self.model

		if not model_file:
			model_file = "PReLIM_model" + str(time.time())

		p.dump(self.model, open(model_file,"wb"))

		return self.model




	# Feature collection directly from bins
	def get_X_y(self, bin_matrices, verbose=False):
		"""
        :param bin_matrices: list of CpG matrices
        :param verbose: prints more info if true
        :return: feature matrix (X) and class labels (y)
        """

		bins = []

		# convert to bin objects for ease of use
		for matrix in bin_matrices:
			mybin = CpGBin( matrix=matrix )
			bins.append( mybin )
		
		# find bins with no missing data
		complete_bins = _filter_missing_data( bins )
		shuffle( complete_bins )
		
		# apply masks
		masked_bins = _apply_masks( complete_bins, bins )

		# extract features
		X, y = self._collectFeatures( masked_bins ) 
		return X, y 


	# Return a vector of predicted classes 
	def predict_classes(self, X):
		"""
		Predict the classes of the samples in the given feature matrix
		
		Usage:
		y_pred = CpGNet.predict_classes(X)  
        
        :param X: numpy array, contains feature vectors
        :param verbose: prints more info if true
        :return: 1-d numpy array of predicted classes
        """

		return self.model.predict(X)
	
	# Return a vector of probabilities for methylation
	def predict(self, X):
		"""
		Predict the probability of methylation for each sample in the given feature matrix
		
		Usage:
		y_pred = CpGNet.predict(X)  
        
        :param X: numpy array, contains feature vectors
        :param verbose: prints more info if true
        :return: 1-d numpy array of prediction values
        """

		return self.model.predict_proba(X)[:,1]


	def predict_proba(self, X):
		"""
		Predict the classes of the samples in the given feature matrix
		Same as predict, just a convenience to have in case of differen styles
		
		Usage:
		y_pred = CpGNet.predict_classes(X)  
        
        :param X: numpy array, contains feature vectors
        :param verbose: prints more info if true
        :return: 1-d numpy array of predicted classes
        """

		return self.model.predict_proba(X)[:1]


	# Load a saved model
	def loadWeights(self, model_file):
		"""
		self.model is loaded with the provided weights

		:param model_file: string, name of file with a saved model
		"""

		self.model = p.load(open(model_file,"rb"))

	


	

	# Imputes missing values in Bins
	def impute(self, matrix):
		"""
		Impute the missing values in a CpG matrix. Values are filled with the 
		predicted probability of methylation.

		:param matrix: a 2d np array, dtype=float, representing a CpG matrix, 1=methylated, 0=unmethylated, -1=unknown
        :return: A 2d numpy array with predicted probabilities of methylation
		"""

		X = self._get_imputation_features(matrix)
		
		if len(X) == 0: # nothing to impute
			return matrix

		predictions = self.predict(X)

		k = 0 # keep track of prediction index for missing states
		predicted_matrix = np.copy(matrix)
		for i in range(predicted_matrix.shape[0]):
			for j in range(predicted_matrix.shape[1]):
				if predicted_matrix[i, j] == -1:
					predicted_matrix[i, j] = predictions[k]
					k += 1

		return predicted_matrix






	# Extract all features for all matrices so we can predict in bulk, this is where the speedup comes from
	def impute_many(self, matrices):
		'''
		Imputes a bunch of matrices at the same time to help speed up imputation time.
	
		:param matrices: array-like (i.e. list), where each element is a 2d np array, dtype=float, representing a CpG matrix, 1=methylated, 0=unmethylated, -1=unknown
        :return: A List of 2d numpy arrays with predicted probabilities of methylation for unknown values.
		'''
		
		X = np.array([features for matrix_features in [self._get_imputation_features(matrix) for matrix in matrices] for features in matrix_features])
		
		if len(X) == 0:
			return matrices
		
		predictions = self.predict(X)


		predicted_matrices = []


		k = 0 # keep track of prediction index for missing states, order is crucial!
		for matrix in matrices:
			predicted_matrix = np.copy(matrix)
			for i in range(predicted_matrix.shape[0]):
				for j in range(predicted_matrix.shape[1]):
					if predicted_matrix[i, j] == -1:
						predicted_matrix[i, j] = predictions[k]
						k += 1
			predicted_matrices.append(predicted_matrix)

		return predicted_matrices




	### Helper functions, for private use only ###

	# get a feature matrix for the given cpg matrix
	def _get_imputation_features(self,matrix):
		'''
		Returns a vector of features needed for the imputation of this matrix
		Each sample is an individual CpG, and the features are
		the row mean, the column mean, the position of the cpg in the matrix,
		the row, and the relative proportions of each methylation pattern 

		:param matrix: a 2d np array, dtype=float, representing a CpG matrix, 1=methylated, 0=unmethylated, -1=unknown
        :return: A feature vector for the matrix
		'''
		
		X = []

		numReads = matrix.shape[0]
		density = matrix.shape[1]

		nan_copy = np.copy(matrix)
		nan_copy[nan_copy == -1] = np.nan

		# get the column and row means
		column_means = np.nanmean(nan_copy, axis=0)
		row_means = np.nanmean(nan_copy, axis=1)
		
		encoding = self._encode_input_matrix(matrix)[0]

		# iterate over all values in the matrix
		for i in range(numReads):
			for j in range(density):
				observed_state = matrix[i, j]
				
				# only record missing values 
				if observed_state != -1:
					continue

				row_mean = row_means[i]
				col_mean = column_means[j]
				
				row = np.copy(matrix[i])
				row[j] = -1

				# features for a single sample
				data = [row_mean] + [col_mean] +  [i, j] + list(row) +  list(encoding)
				
				X.append(data)

		# list to np array
		X = np.array(X)

		return X


	# Returns a matrix encoding of a CpG matrix
	def _encode_input_matrix(self, m):
		"""
		:param m: a 2d np array, dtype=float, representing a CpG matrix, 1=methylated, 0=unmethylated, -1=unknown
        :return: list of relative proportions of each type of methylation pattern, number of reads
		"""
		matrix = np.copy(m)
		n_cpgs = matrix.shape[1]
		matrix += 1  # deal with -1s
		base_3_vec = np.power(3, np.arange(n_cpgs - 1, -1, -1))
		
		encodings = np.dot(base_3_vec, matrix.T)
		
		encoded_vector_dim = np.power(3, n_cpgs)
		encoded_vector = np.zeros(encoded_vector_dim)
		
		for x in encodings:
			encoded_vector[int(x)] += 1
		
		num_reads = encodings.shape[0]
		
		# Now we normalize
		encoded_vector_norm = normalize([encoded_vector], norm="l1")
		return encoded_vector_norm[0], num_reads

	# finds the majority class of the given column, discounting the current cpg
	def _get_column_mean(self, matrix, col_i, current_cpg_state):
		"""
		:param matrix: a 2d np array, dtype=float, representing a CpG matrix, 1=methylated, 0=unmethylated, -1=unknown
		:param col_i: integer, the column index
		:param current_cpg_state: the cpg to discount
        :return: the mean value of column col_i, discounting current_cpg_state
		"""
		sub = matrix[:, col_i]
		return self._get_mean(sub, current_cpg_state)

	# finds the majority class of the given read, discounting the current cpg
	def _get_read_mean(self, matrix, read_i, current_cpg_state):
		"""
		:param matrix: a 2d np array, dtype=float, representing a CpG matrix, 1=methylated, 0=unmethylated, -1=unknown
		:param read_i: integer, the row index
		:param current_cpg_state: the cpg to discount
        :return: the mean value of row read_i, discounting current_cpg_state
		"""
		sub = matrix[read_i, :]
		return self._get_mean(sub, current_cpg_state)


	# Return the mean of sub matrix, discounting the current cpg methylation state
	def _get_mean(self, sub_matrix, current_cpg_state):
		'''
		:param sub_matrix: a list of individual cpgs
		:param current_cpg_state: the cpg to discount
		:return: the mean value of the list, discounting current_cpg_state
		'''
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
		"""
		Given a list of cpg bins, collect features for each artificially masked CpG
		and record the hidden value as the class label.

		:param matrix: bins: list of CpG bins that contain CpG matrices
        :return: feature matrix X and class labels y
		"""
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
					
					
					# row and column means
					row_mean = row_means[i]
					col_mean = column_means[j]

					# j is the current index in the row
					
					# encoding is the matrix encoding vector
					# differences is the difference in positions of the cpgs
					row = np.copy(observed_matrix[i])
					row[j] = -1

					data = [row_mean] + [col_mean] +  [i, j] + list(row) +  list(encoding)
					
					X.append(data)


		X = np.array(X)
		Y = np.array(Y)
		Y.astype(int)
		return X, Y



#### Helper functions ####

# Returns a list of bins similar to the input
# but matrix rows with missing values are removed
def _filter_bad_reads(bins):
	"""
	Given a list of cpg bins, remove reads with missing values
	so we can mask them.
	
	:param matrix: bins: list of CpG bins that contain CpG matrices
    :return: bins, but all reads wiht missing values have been removed
	"""
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

# Returns a mapping of dimensions to list of masks that can be used on data
# of that size. the missing pattern is in matrix form.
# -1 is missing, 2 is known
def _extract_masks( bins):
	"""
	Given a list of cpg bins, return a list matrices that
	represent the patterns of missing values, or "masks"
	
	:param matrix: bins: list of CpG bins that contain CpG matrices
    :return: list of matrices that represent the patterns of missing values
	"""
	masks = defaultdict(lambda: [])
	for Bin in tqdm(bins):
		matrix = np.copy(Bin.matrix)
		matrix[matrix >= 0] = 2

		#min_missing = 10
		min_missing = 1 # must have at least 1 missing value
		if np.count_nonzero(matrix == -1) >= min_missing:
			masks[matrix.shape].append(matrix)

	return masks

# Extract masks from original matrices and apply them to the complete matrices
def _apply_masks( filtered_bins, all_bins ):
	"""
	Given a list of filtered cpg bins and a list of all the bins,
	extract masks from the original bins and apply them to the filtered bins.
	
	:param filtered_bins: bins with no reads with missing values.
	:param all_bins: list of CpG bins that contain CpG matrices
	:return: list of matrices that represent the patterns of missing values
	"""
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

# Get a list of bins with no missing data
def _filter_missing_data( bins, min_read_depth=1 ):
	"""
	Given a list of filtered cpg bins and a list of all the bins,
	extract masks from the original bins and apply them to the filtered bins.
	
	:param bins: list of CpG bins that contain CpG matrices
	:param min_read_depth: minimum number of reads needed for a bin to be complete.
	:return: remove reads with missing values from bins

	"""
	cpg_bins_complete = _filter_bad_reads(bins)
	# secondary depth filter
	cpg_bins_complete_depth = [bin_ for bin_ in cpg_bins_complete if bin_.matrix.shape[0] >= min_read_depth]
	return cpg_bins_complete_depth


