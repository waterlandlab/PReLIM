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

from collections import defaultdict

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

# bin data type
from .CpG_Bin import Bin

# warnings suck, turn thme off
if sys.version_info[0] < 3:
    warnings.simplefilter("ignore", DeprecationWarning)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
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
            weight_file="CpGNetWeights2.h5"
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
        convInput = Input(shape=(max_depth,CPG_DENSITY,1), dtype='float', name='input2')

        filter_size = CPG_DENSITY
        stride = filter_size

        convLayer = Dropout(0.2)(convInput) # dropout input to simulate noise
        convLayer = Conv2D(32, kernel_size=(2,2), strides=2, padding="same",activation="linear")(convLayer)
        convLayer = LeakyReLU(alpha=.001)(convLayer)
        convLayer = Conv2D(16, kernel_size=(2,2), strides=2, padding="same",activation="linear")(convLayer)
        convLayer = LeakyReLU(alpha=.001)(convLayer)
        convLayer = Conv2D(8, kernel_size=(2,2), strides=2, padding="same",activation="linear")(convLayer)
        convLayer = LeakyReLU(alpha=.001)(convLayer)

        #convLayer = MaxPooling2D()(convLayer)

        convLayer = Flatten()(convLayer)

        #convLayer = Flatten()(convInput)
        #convLayer = Dense(1000, activation="relu")(convLayer)

        # Numerical Module
        numericalInput = Input(shape=(Y[0].size,), dtype='float', name='input1')
        layer1 = Dropout(0.2)(numericalInput) # dropout on input to simulate noise
        layer1 = Dense(1000, activation="linear")(layer1)
        layer1 = LeakyReLU(alpha=.01)(layer1)


        layer1 = Dense(100, activation="linear")(layer1)
        layer1 = Dropout(0.5)(layer1)
        layer1 = LeakyReLU(alpha=.01)(layer1)

        layer1 = Dense(10, activation="linear")(layer1)
        layer1 = LeakyReLU(alpha=.01)(layer1)

        # Combined Module

        combined = keras.layers.concatenate([convLayer, layer1])
        combined = Dense(1000, activation="linear")(combined)
        combined = LeakyReLU(alpha=.01)(combined)
        combined = Dropout(0.5)(combined)

        combined = Dense(800, activation="linear")(combined)
        combined = LeakyReLU(alpha=.01)(combined)
        combined = Dropout(0.5)(combined)
        combined = Dense(400, activation="linear")(combined)

        combined = LeakyReLU(alpha=.01)(combined)
        combined = Dropout(0.5)(combined)
        combined = Dense(1, activation="sigmoid")(combined)

        # x_input_dim = X_train.shape[1]

        # self.model = Sequential()
        # self.model.add(Dense(1000, activation='linear', input_dim=x_input_dim))
        # self.model.add(LeakyReLU(alpha=.0001))

        # self.model.add(Dense(800, activation='linear', input_dim=x_input_dim))
        # self.model.add(LeakyReLU(alpha=.0001))

        # self.model.add(Dropout(0.5))

        # self.model.add(Dense(500, activation='linear'))
        # self.model.add(LeakyReLU(alpha=.0001))

        # self.model.add(Dense(100, activation='linear'))
        # self.model.add(LeakyReLU(alpha=.0001))
        

        # # output
        # self.model.add(Dense(1, activation='sigmoid'))

        # adam = keras.optimizers.Adam(lr=0.0001)

        # self.model.compile(optimizer=adam,
        #                    loss='binary_crossentropy',
        #                    metrics=['accuracy'])
        # earlystopper = EarlyStopping(patience=5, verbose=1)

        # weight_file = "CpGNet_" + str(self.cpgDensity) + "cpg_weights.h5"
        # checkpointer = ModelCheckpoint(weight_file, monitor='val_acc', verbose=1, save_best_only=True, mode="max")

        # return self.model.fit(X_train, y_train,
        #                       epochs=epochs,
        #                       batch_size=batch_size,
        #                       callbacks=[earlystopper, checkpointer],
        #                       validation_split=val_split,
        #                       verbose=True,
        #                       shuffle=True)

    def predict(X, y):
        return self.model.predict(X, y)

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

    # Imputes a matrix, useful when not much information is known
    # positional data is still needed

    # Imputes missing values in Bins
    def impute(self, matrix, positions, bin_start, bin_end):
        """
		Inputs: 
		1. matrix, a 2d np array representing a CpG matrix, 1=methylated, 0=unmethylated, -1=unknown
		2. positions, a 1d np array containing the chromosomal positions, left to right, of each cpg in the bin
        3. bin_start, integer, the leftmost position of the bin, inclusive
        4. bin_end, integer, the rightmost position of the bin, inclusive

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

        for i in range(numReads):
            for j in range(density):
                observed_state = matrix[i, j]
                if observed_state != -1:
                    continue

                # encoding is the matrix encoding vector
                # encoding = self.encode_input_matrix(observed_matrix)[0]

                # # record the relative differences in CpG positions
                rel_pos = []
                rel_pos.append((positions[0] - bin_start) / 100.0)  ## distance to left bin edge
                for pos in positions:
                    rel_pos.append((pos - bin_start) / 100.0)
                rel_pos.append((bin_end - positions[-1] + 1) / 100.0)  ## distance to left bin edge

                row_mean = row_means[i]
                col_mean = column_means[j]
                # j is the current index in the row
                # M[] is the current row data
                row = np.copy(matrix[i])
                row[j] = -1
                # data = [j]  + list(encoding) + differences
                data = [row_mean] + [col_mean] + rel_pos + [row_mean] + [col_mean] + [j] + list(row)  # list(encoding)
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
    def encode_input_matrix(self, m):
        matrix = np.copy(m)
        n_cpgs = matrix.shape[1]
        matrix += 1  # deal wiht -1s
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
        encoded_vector = normalize(encoded_vector, norm="l1")
        return encoded_vector[0], num_reads

    # finds the majority class of the given column, discounting the current cpg
    def get_column_mean(self, matrix, col_i, current_cpg_state):
        sub = matrix[:, col_i]
        return self.get_mean(sub, current_cpg_state)

    # finds the majority class of the given read, discounting the current cpg
    def get_read_mean(self, matrix, read_i, current_cpg_state):
        sub = matrix[read_i, :]
        return self.get_mean(sub, current_cpg_state)

    def get_mean(self, sub_matrix, current_cpg_state):
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
    def collectFeatures(self, bins):
        X = []
        Y = []
        for Bin in tqdm(bins):
            observed_matrix = Bin.tag2["observed"]
            truth_matrix = Bin.tag2["truth"]
            numReads = observed_matrix.shape[0]
            density = observed_matrix.shape[1]
            positions = Bin.cpgPositions

            nan_copy = np.copy(observed_matrix)
            nan_copy[nan_copy == -1] = np.nan
            column_means = np.nanmean(nan_copy, axis=0)
            row_means = np.nanmean(nan_copy, axis=1)

            for i in range(numReads):
                for j in range(density):
                    observed_state = observed_matrix[i, j]
                    if observed_state != -1:
                        continue

                    state = truth_matrix[i, j]
                    Y.append(state)

                    # encoding = self.encode_input_matrix(observed_matrix)[0]

                    # # record the relative differences in CpG positions
                    rel_pos = []
                    rel_pos.append((positions[0] - Bin.binStartInc) / 100.0)  ## distance to left bin edge
                    for pos in positions:
                        rel_pos.append((pos - Bin.binStartInc) / 100.0)
                    rel_pos.append((Bin.binEndInc - positions[-1] + 1) / 100.0)  ## distance to left bin edge

                    row_mean = row_means[i]
                    col_mean = column_means[j]
                    # j is the current index in the row
                    # M[] is the current row data
                    # encoding is the matrix encoding vector
                    # differences is the difference in positions of the cpgs
                    row = np.copy(observed_matrix[i])
                    row[j] = -1
                    # data = [j]  + list(encoding) + differences
                    data = [row_mean] + [col_mean] + rel_pos + [row_mean] + [col_mean] + [j] + list(row)  # list(encoding)
                    X.append(data)

        X = np.array(X)
        Y = np.array(Y)
        Y.astype(int)
        return X, Y

    # returns a list of bins similar to the input
    # but matrix rows with missing values are removed
    def filter_bad_reads(self, bins):
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
    def extract_masks(self, bins):
        masks = defaultdict(lambda: [])
        for Bin in tqdm(bins):
            matrix = np.copy(Bin.matrix)
            matrix[matrix >= 0] = 2

            min_missing = 10
            if np.count_nonzero(matrix == -1) > min_missing:
                masks[matrix.shape].append(matrix)

        return masks

    def advanced_feature_collect(self, bins):
        X = []  # matrices
        Y = []  # numerical data
        Z = []  # labels

        for Bin in tqdm(bins):
            observed_matrix = Bin.tag2["observed"]

            truth_matrix = Bin.tag2["truth"]
            numReads = observed_matrix.shape[0]
            density = observed_matrix.shape[1]
            positions = Bin.cpgPositions

            # copy so we can compute means while ignoring missing values
            nan_copy = np.copy(observed_matrix)
            nan_copy[nan_copy == -1] = np.nan
            column_means = np.nanmean(nan_copy, axis=0)
            row_means = np.nanmean(nan_copy, axis=1)

            for i in range(numReads):
                for j in range(density):

                    observed_state = observed_matrix[i, j]

                    # only record missing states
                    if observed_state != -1:
                        continue
                    state = truth_matrix[i, j]

                    rel_pos = []
                    rel_pos.append((positions[0] - Bin.binStartInc) / 100.0)  ## distance to left bin edge
                    for pos in positions:
                        rel_pos.append((pos - Bin.binStartInc) / 100.0)
                    rel_pos.append((Bin.binEndInc - positions[-1] + 1) / 100.0)  ## distance to left bin edge

                    row_mean = row_means[i]
                    col_mean = column_means[j]
                    # give positions and current index of cpg
                    numerical_data = list(observed_matrix[i]) + [row_mean] + [col_mean] + rel_pos + [j] + [i] + [
                        density] + [numReads]

                    X.append(observed_matrix)
                    Y.append(numerical_data)
                    Z.append(state)

        # convert to np arrays
        X = np.array(X)
        Y = np.array(Y)
        Z = np.array(Z)
        Z.astype(int)  # labels need to be ints
        return X, Y, Z

