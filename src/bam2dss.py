

import sys

import argparse
import numpy as np
#from PReLIM import PReLIM
import PReLIM
import random
from collections import defaultdict
from clubcpg.ParseBam import BamFileReadParser
import multiprocessing as mp
import time
from tqdm import tqdm

from itertools import product


# NOTE: it is recommended that you use this on a chromosome-by-chromosome basis


class PReLIM_Bam2DSS():

	def __init__(self):
		self.MIN_IMPUTATION_DENSITY = 2
		self.MAX_IMPUTATION_DENSITY = 6
		self.BIN_SIZE = 100
		self.ALL_CHROMES = ["chr" + str(i) for i in range(1,23)] + ["chrX", "chrY"]


	def parse_arguments(self):
		"""
		Parses command line arguments
		:return: parsed argument object 
		"""

		arg_parser = argparse.ArgumentParser(description="hello there!")
		arg_parser.add_argument("-i", required=True, help="input bam file to be analyzed (must be indexed with samtools already)")
		arg_parser.add_argument("-o", required=True, help="output file")
		arg_parser.add_argument("-c", required=False, help="name of the chromosome to be analyzed (e.g. -b chrX). If not provided, use all chromosomes")
		arg_parser.add_argument("-p", required=False, action="store_true", help="Use PReLIM if this flag is on, skip imputation otherwise")
		arg_parser.add_argument("-m", required=False, help="Model prefix to use. Models must be stored in path and be titled [prefix]_[cpg_density]cpg_revision_prelim_model")

		args = arg_parser.parse_args()
		

		return args

	def get_chromosome_lengths(self, parser, chrome=None):
			"""
			Get dictionary containing lengths of the chromosomes. Uses bam file for reference
			:return: Dictionary of chromosome lengths, ex: {"chrX": 222222}
			"""
			d = dict(zip(parser.OpenBamFile.references, parser.OpenBamFile.lengths))
			d = {k:d[k] for k in d if "chr" in k}

			# only picked one chromosome
			print("input chromes:",chrome)
			if chrome:
				return {chrome: d[chrome]}
		
			return d

	def generate_bins_list(self, chromosome_len_dict):
		"""
		Get a dict of lists of all bins according to desired bin size for all chromosomes in the passed dict
		:param chromosome_len_dict: A dict of chromosome length sizes from get_chromosome_lenghts, cleaned up by remove_scaffolds() if desired
		:return: dict with each key being a chromosome. ex: chr1
		"""
		all_bins = defaultdict(list)
		for chro, length in chromosome_len_dict.items():

			bins = list(np.arange(self.BIN_SIZE, length + self.BIN_SIZE, self.BIN_SIZE))
			bins = ["_".join([chro, str(x)]) for x in bins]
			all_bins[chro].extend(bins)

		return all_bins

	def bin2coords(self, bin_name):
		"""
		Given a bin name, return the chr name, start, and end of the bin
		:param bin_name: the bin name to process (e.g. "chr1_1200")
		:return chr start
		:return chr name, bin start coord, and bin end coord
		"""
		chr_name, end = bin_name.split("_")
		end = int(end)
		start = end - self.BIN_SIZE # assumes bin size of 100
		return chr_name, start, end


	

	def get_bin_matrix(self, parser, bin_name, return_cpgs = False):
		"""
		Given a bin name and a parser, return the matrix associated with that bin, include CpG pos if necessary
		:param parser: bam file parser
		:param bin_name: string, bin name to use (e.g. "chr1_1200")
		:param return_cpgs: boolean, whether or not to return CpGs
		:return the CpG matrix corresponding to the given bin
		"""

		chro, _, bin_loc = self.bin2coords(bin_name)

		try:
			# Parse the file to get reads in the bin and form them into a matrix
			reads = parser.parse_reads(chro, bin_loc - 99, bin_loc)
			reads = parser.correct_cpg_positions(reads)
			df = parser.create_matrix(reads)

			df.dropna(inplace=True, how="all")
			cpgs = list(df.columns)
			matrix = self.matrix_nan_to_neg(df.values)
			
			if self.impute:
				matrix = self.impute_matrix(matrix)

			if return_cpgs:
				return matrix, cpgs
			else:
				return matrix

		except:
			if return_cpgs:
				return np.array([]),[]
			else:
				return np.array([])

	def matrix_nan_to_neg(self, raw_matrix):
		"""
		Convert nans in a matrix to -1's
		:param raw_matrix: a numpy array, may contain nans
		:return the input matrix but all nans are now -1's
		"""
		raw_matrix_plus1 = raw_matrix + 1 
		matrix = np.nan_to_num(raw_matrix_plus1) - 1
		return matrix


	def get_matrices_for_training(self, parser, bin_name_list, num_samples_per_density=10000):
		"""
		Return a mapping from matrix density ot a list of matrices that can be used to train a PReLIM model 
		of that density.

		:param parser: a bam file parser
		:param bin_name_list: a list of bin names to parse
		:param num_samples_per_density: int, the number of samples 

		:return  a mapping from matrix density ot a list of matrices
		"""

		# initialize mapping from density to list of matrices with that density
		densities = list( range( self.MIN_IMPUTATION_DENSITY, self.MAX_IMPUTATION_DENSITY ) )
		matrices_by_density = {}
		for den in densities:
			matrices_by_density[den] = []


		bin_name_list = list(bin_name_list) # make a copy

		random.shuffle(bin_name_list) # make sure we get good representation

		for bin_name in bin_name_list:
			matrix,cpgs = self.get_bin_matrix(parser, bin_name, return_cpgs=True)
			matrix_density = self.get_matrix_density(matrix)
			if matrix_density in matrices_by_density:
				matrices_by_density[matrix_density].append(matrix)
			
			# do we have enough samples?
			if min([len(v) for v in matrices_by_density.values()]) == num_samples_per_density:
				break

		return matrices_by_density


	def train_models(self, matrices_by_density):
		"""

		"""


		densities = list( range( self.MIN_IMPUTATION_DENSITY, self.MAX_IMPUTATION_DENSITY ) )
	
		self.matrices_by_density = matrices_by_density

		# train in parallel
		num_cpus = mp.cpu_count()
		pool = mp.Pool(num_cpus)
		models = pool.map(self.train_on_density, densities)

		# Create a map from density to model
		model_map = {}
		for model, den in zip(models, densities):
			model_map[den] = model

		return model_map

	def train_on_density(self, den):
		"""
		"""
		model = PReLIM(cpgDensity=den) 

		if len(self.matrices_by_density[den]) < 2: # not enough matrices to train on
			print("OOPS! not enough matrices to train for density: " + str(den))
			return model

		model.train(self.matrices_by_density[den], model_file="no")
		return model

	def train(self, parser, bin_name_list):
		"""
		"""
		matrices_by_density = self.get_matrices_for_training(parser, bin_name_list)
		models = self.train_models(matrices_by_density)
		return models


	def get_matrix_density(self, matrix):
		"""
		"""
		try:
			return matrix.shape[1] 
		except:
			return 0


	def impute_matrix(self, matrix):
		"""
		"""
		try:
			den = self.get_matrix_density(matrix)

			if den in self.models:
				model = self.models[den]
				imputed = model.impute_class(matrix)
				return imputed
			else:
				return matrix
		except:
			return matrix



	def toDssFormat(self, matrix_and_position, chrome):
		"""
		"""
		matrix = matrix_and_position[0]

		if len(matrix) == 0:
			return ""
		
		else:
			positions = matrix_and_position[1]
			methy_counts = np.sum(matrix>0,axis=0,dtype=int)
			read_counts = np.sum(matrix>=0,axis=0)

			return ["{}\t{}\t{}\t{}".format(chrome, pos, r, m) for pos, m, r in zip(positions, methy_counts, read_counts)]

		

	def binName2dssFormat_chunk(self, bin_name_chunk):
		"""
		"""
		results = []
		


		parser = BamFileReadParser(self.input_bam_file, quality_score = 0)

		for bin_name in bin_name_chunk:
			# get the bin matrix
			bin_matrix = self.get_bin_matrix(parser, bin_name, return_cpgs=True)
			# get the chrome name
			chrome = bin_name.split("_")[0]
			# convert to dss format
			dss_format = self.toDssFormat(bin_matrix, chrome)
			# append
			results += (dss_format)
		

		return results

		


	def print_results(self, results, out_file):
		"""
		Prints out the results in DSS format.
		results should be a list of DSS formated lines
		"""
		header = "{}\t{}\t{}\t{}\n".format("chr","pos","N","X")
		with open(out_file,"w") as f:
			f.write(header)
			for res in results:
				for x in res:
					line = "{}\n".format(x)
					f.write(line)

	def load_models(self, model_prefix):
		"""
		"""
		models = {}

		for cpg_density in range(2, 7):
			weight_file = "{}_{}cpg_revision_prelim_model".format(model_prefix, cpg_density)
			model = PReLIM(cpgDensity=cpg_density)
			model.loadWeights(weight_file)

			models[cpg_density] = model
		
		return models


	def run(self):
		"""
		"""

		# Get command line arguments
		args = self.parse_arguments()
		self.input_bam_file = args.i
		self.chromes = args.c
		self.impute = args.p
		self.outfile=args.o

		# Get a parser and chromosome information
		parser = BamFileReadParser(self.input_bam_file, quality_score= 0)
		chromosome_lens = self.get_chromosome_lengths(parser, self.chromes)

		print("got parser")
		# get a list of all the bin names
		bin_name_list = []
		for chrom in chromosome_lens:
			chrome_bin_list = self.generate_bins_list(chromosome_lens)[chrom] 
			bin_name_list += chrome_bin_list


		print("got bin name list")
		#bin_name_list = bin_name_list[100000:101000] # TODO: don't do this!!!

		# Get PReLIM models
		if args.p:
			# load existing models
			if args.m:
				self.models = self.load_models(args.m)
				print("loaded models")
			# train new models
			else:			
				self.models = self.train(parser, bin_name_list)
		else:
			print("not imputing, no need for models")


		# parallelize the imputation
		num_cpus =  int(mp.cpu_count()/2)
		num_bins = len(bin_name_list)
		chunk_size = int(num_bins / num_cpus)
		bin_name_chunks = [bin_name_list[i: i + chunk_size] for i in range(0, num_bins, chunk_size)]
		
		pool = mp.Pool(num_cpus)
		results = pool.map(self.binName2dssFormat_chunk, bin_name_chunks)
		
		self.print_results(results, self.outfile)
		

		
if __name__ == "__main__":
	pbd =  PReLIM_Bam2DSS()
	pbd.run()







