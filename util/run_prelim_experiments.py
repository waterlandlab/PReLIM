'''
Jack Duryea
December 2018
Baylor College of Medicine
Waterland Lab

This script makes it easy to run experiments regarding PRELIM,
such as K-Fold validation, etc

'''
import sys

sys.path.append('/home/jduryea/CpGNet/util')
sys.path.append('/home/jduryea/CpGNet/model')
sys.path.append('/home/jduryea/CpGNet/data')


from datautil import datautil
from CpGNet import CpGNet
from CpG_Bin import Bin

import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from random import shuffle
from sklearn.metrics import roc_curve, auc
from datautil import datautil
import random

import sys
import argparse
from sklearn.model_selection import StratifiedKFold

def run_kfold(density, file):

	net = CpGNet(CPG_DENSITY)
	f = open(filename, "rb")
	all_bins = pickle.load( f )	
	shuffle(all_bins)


	bins = []

	# convert to bin objects for ease of use
	for matrix in all_bins:
		mybin = CpGBin( matrix=matrix )
		bins.append( mybin )
	
	# find bins with no missing data
	complete_bins = _filter_missing_data( bins )
	shuffle( complete_bins )
	
	# apply masks
	masked_bins = _apply_masks( complete_bins, bins )

	# extract features
	X, y = net._collectFeatures( masked_bins ) 

	num_vals = 10000 # cut down for speed
	X = X[:num_vals]
	y = y[:num_vals]

	# Run k-fold
	k = 5
	kFold = StratifiedKFold(n_splits = k)
	current_fold = 1
    for train, test in kFold.split(X, y):


		# Train the neural network model
		net.fit( X[train], y[train], weight_file = "kFold_test"+str(current_fold) )
		accuracy = net.get_accuracy( X[test], y[test] )

		print ("current fold: ", current_fold,", accuracy:" accuracy)

		strat += 1



def run():
	parser = argparse.ArgumentParser()

	parser.add_argument( "-k", required=False, help="run K-Fold validation", action='store_true' )
	parser.add_argument("density", help = "cpg density") # integer
	parser.add_argument("binfile", help="file for cpg bins")
	# parser.add_argument( "-p", help="invokes parser and reports on success or failure", action='store_true' )
	# parser.add_argument( "-r", help="prints human readable version of the internal representation", action='store_true' )
	# parser.add_argument('filename')

	# num_flags = args.p + args.s + args.r # easy way to find number of flgas

	# # Too many flags, implicitly resort to highest priority flag
	# if num_flags > 1:
	# 	print_message( "Dear sir or madam: The number of flags you have used exceeds the number of acceptable flags. Resorting to flag with highest priority [h=1st, r=2nd, p=3rd, s=4th].")

	# try:
	# 	file = open( args.filename, "rb" )
	# 	scanner = create_scanner( file )

	# 	# Go through commands: These ifs must be done in order of decreasing priority of the flag
	# 	# so that high priority flags are hit first
		
	# 	if args.r: # display the internal representation
	# 		parser = parse( scanner )
	# 		display_IR( parser )

	# 	elif args.p: # Parse the file. Shows all parsing errors, or reports success


	args = parser.parse_args() 
	density = int(args.density)
	file = args.binfile


	if args.k:
		run_kfold(density, file)






run()