"""
Jack Duryea 
(duryea@bcm.edu)
Computational Epigenetics Sector
Waterland Lab, BCM 2017

AXTELL

Read-level Methylation Extractor


Changelog:

UPDATE 7-31-2017
Includes a column that shows which CpGs in each read contribute to that read's methylation status  

UPDATE 10-30-2017
Fixes a bug where reads report CpGs at sligthly different locations, causing extra missing values to appear
Cleaned up the code and removed uncessary pieces of code


UPDATE 4-26-2018
Added capability to store objects as Bin objects

Sample usage:
python axtell.py -c chr19 -s debug.txt -o debugout.csv

"""

from scipy.stats.mstats import gmean
from collections import defaultdict
import multiprocessing as mp
from CpG_Bin import Bin
import pandas as pd
import numpy as np
import linecache
import argparse
import sklearn
import math
import sys
import os
import ctypes
import cPickle 


# Reads a same file containing a sequence of reads (strings) from a sam file
def get_reads(datafile):
	"""
	Input: 
	1. datafile - name of a sam file

	Output: 
	1. A list of reads starting positions and corresponding methylation call strings
	"""

	file = open(datafile)
	data = [x for x in list(file) if x[0] != "@"]
	file.close()
	return data


# Finds the relative location of a CpG in a bin
def get_rel_pos(x):
	"""
	Input
	1. x - absolute position of a CpG in chromosome. Integer.
	
	Output
	1. relative position of CpG in a bin. Integer. 
	"""
	if x % bin_size == 0:
		return bin_size
	else:
		return x % bin_size

# Returns 0 if a read reports methylated CpGs
# at the C, -1 if it reports at the G, and 0 otherwise
def reports_at_G(methy_read, seq_read):
	if len(methy_read) != len(seq_read):
		return 0
	for i in range(len(methy_read)):
		if methy_read[i] in ["Z", "z"]:
			if seq_read[i] in ["G","g","A","a"]:
				return -1
	return 0


# Returns the bin that would contain a given dinucleotide position, 1-based coordinates
def get_bin_name(cpg_pos):
	bin_num = (cpg_pos-1)/bin_size # integer division is key
	bin_name = (bin_num*bin_size) + bin_size
	return bin_name



# Adds lists of read methylation values to the corresponding bins
def update_data_frame(bin_data):
	#print "appending tco dataframe"
	# maps from bin name to a list of average methylation values
	bin_read_methylation = defaultdict(lambda:[])
	bin_methy_contribution = defaultdict(lambda:[])
	for read_datem in bin_data:
		for bin_datem in read_datem:
			bin_name = bin_datem[0]
			avg_methy = bin_datem[1]
			cpg_contribution = bin_datem[2]
			bin_read_methylation[bin_name].append(avg_methy)
			bin_methy_contribution[bin_name].append(cpg_contribution)

	# Turn into a dataframe 
	bin_names = list(bin_read_methylation.keys())
	bin_names.sort()
	methy_values = []
	contribution_values = []
	for bin_name in bin_names:
		methy_values.append(bin_read_methylation[bin_name])
		contribution_values.append(bin_methy_contribution[bin_name])

	data_dict = {"Bin Name": bin_names, "Average Methylation Values":methy_values, "Read Contributions":contribution_values}
	df = pd.DataFrame.from_dict(data_dict)
	df = df[["Bin Name", "Average Methylation Values","Read Contributions"]]
	return df


# Runs parallelization
def analyze_sam_file(filename):
	"""
	Input:
	1. data_length: integer, number of reads in the input file
	"""
	reads = get_reads(filename)
	bin_data = []

	# Create shared memory for multiprocessing
	d = np.array(reads)
	shared_array_base = mp.Array(ctypes.c_double, d.size)
	shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
	shared_array = d
	shared_array = shared_array.reshape(d.shape)


	for i in range(len(reads)):
		bin_data.append(compute_read_methylation(i,def_param=shared_array))

	df = update_data_frame(bin_data)
	return df


# Computes the overlap between a read and a bin. If the overlap contains x or more CpG sites,
# it is considered informative. The bin name and average read methylation for those
# overlapping CpG sites are computed. Returned is a set of bin names and corresponding average methylation
# that this read is contributing
#def compute_read_methylation(index, def_param=shared_array, bin_size=200): Parallel version
def compute_read_methylation(read):

	# Determine if the read is informative for any of the bins it overlaps
	informative_bins = [] 

	data = read.split()
	
	start_pos = int(data[3])	
	seq = data[9]
	methy_call = data[16][5:] # change 16 back to 13 for Jduryea's reads, 16 for scott
	cigar = data[5]
	MAPQ = data[4]
	if MAPQ < 3: # bad mapping quality,
		return []
	if cigar.count("M") > 1: # sequence doesn't directly align
		return []
		 
		methy_call = update_methy_with_cigar(methy_call,cigar)
	# correction factor for reads that report at Guanine
	guanine_correction_factor = reports_at_G(methy_call,seq)
	start_pos += guanine_correction_factor


	# Place CpG's in corresponding bins
	bin_cpg_data = defaultdict(lambda:[])
	for i, dn in enumerate(methy_call):
		if dn in ["Z","z"]: # if nucleotide is a C
			pos = start_pos+i
			bin_name = get_bin_name(pos)
			# cpg_rel_pos  =  get_rel_pos(pos) # the cpgs position relative to it's bin
			cpg_tup = (pos, int(dn=="Z")) # Tuple representing position and methy of CpG locus

			bin_cpg_data[bin_name].append(cpg_tup) # 1 if methylated, 0 else


	
	# We must cover at least half the CpGs
	for bin_, cytosines in bin_cpg_data.items():
		bin_name = chromosome_name + "_"+str(bin_)
		#num_cpgs_in_bin = bin_cpg_count_mapping[bin_name]
		read_mean_methy = np.mean([float(x[1]) for x in cytosines])
		# Filtering criterion, turned off for now
		# if num_cpgs_in_bin >= 4 and len(cytosines) >= math.ceil(num_cpgs_in_bin/2.0): # Must cover at least half the CpGs in the bin
		# 	informative_bins.append( (bin_, read_mean_methy, (read_mean_methy,cytosines)) )
		# elif num_cpgs_in_bin == 2 and len(cytosines) == 2:
		# 	informative_bins.append( (bin_, read_mean_methy, (read_mean_methy,cytosines)) )
		# elif num_cpgs_in_bin == 3 and len(cytosines) == 3:
			
		informative_bins.append( (bin_, read_mean_methy, (read_mean_methy,cytosines)) )


	return informative_bins

def parse_cigar(cigar):
    cigar_list = []
    prev_idx = 0
    for i in range(len(cigar)):
        if cigar[i].isalpha(): # If the character is a letter
            component = cigar[prev_idx:i+1]
            prev_idx = i+1
            cigar_list.append(component)
    return cigar_list


# Adapts the methylation call to account for the CIGAR string, adds "." or removes them based
def update_methy_with_cigar(methy_call, cigar):
    updated_methy_call = ""
    parsed_cigar = parse_cigar(cigar) # Breaks the cigar string into its components
    for component in parsed_cigar:
        if "M" in component: # All base pairs are matching
            length = int(component[:component.index("M")])
            substring = methy_call[:length]
            methy_call = methy_call[length:]

            updated_methy_call += substring
        if "I" in component:
            length = int(component[:component.index("I")])
            methy_call = methy_call[length:]
        if "D" in component:
            length = int(component[:component.index("D")])

            updated_methy_call += "."*length
    return updated_methy_call     
     

# Creates a cpg matrix based on the read contribution data in a bin
def make_cpg_matrix(cpg_data):
	

	num_reads = len(cpg_data)



	positions = set([])

	# scan positions is first sweep
	for read_index in range(num_reads):
		read = cpg_data[read_index]
		for cpg_i in range(len(read[1])):
			cpg = read[1][cpg_i]
			position = cpg[0]
			methyStatus = cpg[1]
			positions.update({position})

	
	# not all reads contain all positions, so finding all possible positions are necessary
	positions = sorted(positions)

	num_cpgs = len(positions)


	# now create the cpg matrix
	matrix = -1 * np.ones((num_reads, num_cpgs))

	for read_index in range(num_reads):
		read = cpg_data[read_index]
		for cpg_i in range(len(read[1])):
			cpg = read[1][cpg_i]
			position = cpg[0]
			methyStatus = cpg[1]
			matrix[read_index, positions.index(position)] = methyStatus

	bin_name = get_bin_name(positions[0])
	bin_name -= bin_size # correction
	binStart = bin_name
	binEnd = bin_name + bin_size - 1

	mybin = Bin(matrix=matrix, binStartInc=bin_name, binEndInc= binEnd, cpgPositions=positions, chromosome=chromosome_name, binSize=bin_size, verbose=True, tag1="Jack's demo data, April 2018")
	
	print "---------------"
	print "matrix\n", mybin.matrix
	print "bin start:", mybin.binStartInc
	print "bin end:", mybin.binEndInc
	print "positions:",positions




	return mybin


#	mybin = Bin()
	


	# cpg_matrix = np.zeros((len(cpg_data), bin_size))#.fill(-1)
	# cpg_matrix.fill(-1)
	# positions = set([])
	# for read_index in range(len(cpg_data)):
	# 	read = cpg_data[read_index]
	# 	for cpg in read[1]:
	# 		position = cpg[0]
	# 		positions.update({position})
	# 		methyStatus = cpg[1]
	# 		cpg_matrix[read_index,position-1] = methyStatus
	
	# ## filter column
	# cpg_matrix = np.transpose(cpg_matrix)
	# cpg_matrix = cpg_matrix[np.sum(cpg_matrix, axis=1) != -1 *len(cpg_data)]

	# return np.transpose(cpg_matrix), list(sorted(positions))





# Given a sorted list and a target value, efficiently find the element in the list closest in abs value to the target
def inexact_binary_search(alist, target):
    # alist - a sorted list, ascending order (left is low right is high)
    # target - the item we are looking for
    l = len(alist)
    # Base case
    if l == 1:
        return alist[0]
    # Recursive case
    else:
        # Split list in half
        L = alist[:l/2]
        R = alist[l/2:]
        if abs(L[-1] - target) < abs(R[0]-target):
            return inexact_binary_search(L,target)
        elif L[-1] == R[0]:
                if target <= L[-1]:
                    return inexact_binary_search(L,target)
                else:
                    return inexact_binary_search(R,target)
        else:
            return inexact_binary_search(R,target)


# Cleans the dataframe and uses annotation information
def clean_data(dataframe, chromosome_name):
	"""
	Input:
	1. dataframe: a pandas dataframe with bins and associated list of averages for read methylation
	2. annotation: a dataframe with chromosomal annotations
	3. chromosome_name: string, name of the chromosome being analyzed

	Output:
	1. A cleaned dataframe
	"""

	# Parse list of average read methylation values
	dataframe["Bin Name"] = dataframe["Bin Name"].apply(str)
	dataframe["Bin Name"] = chromosome_name+"_"+dataframe["Bin Name"]

	# Get CpG counts from annotation
	# cpg_counts = []
	# for bin_ in list(dataframe["Bin Name"]):
	#     count = bin_cpg_count_mapping[bin_]
	#     cpg_counts.append(count)
	# dataframe["CpG Count"] = cpg_counts
	return dataframe


# Filters a dataframe by cpg counts in bin and read coverage
def filter_data(dataframe, min_read_coverage, min_cpg_count):
	"""
	Inputs:
	1. dataframe, a pandas dataframe
	2. min_read_coverage, integer, the number of overlapping reads required to keep the bin
	3. min_cpg_count, integer, the number of CpGs in a bin required to keep it

	Output:
	1. A filtered data frame
	"""

	# Filter bins by number of reads and CpG counts
	# df_filtered = dataframe[dataframe["CpG Count"]>= min_cpg_count]
	df_filtered = dataframe[dataframe["Average Methylation Values"].apply(len) >= min_read_coverage]
	return df_filtered


# Given an index, this returns the position
# and methy call associated with the read in this 
# position of the dataframe. This function helps us in 
# multiprocessing that we don't have to copy over all the
# data for each process
def get_read_from_index(index):
	line = linecache.getline(sam_file, index).split("\s+") # Get line efficiently
	# TODO: should we clear the cache at some point?
	linecache.clearcache()
	if using_super_reads:
		return (line[0],line[1])
	else:
		return (line[3], line[16][5:])


# Similar purpose to the function get_read_from_index,
# Returns the average methylation values
def get_methylation_patter_from_index(index):
	return read_methylation_df["Average Methylation Values"].iloc[0]



# Command line arguments
des = "Axtell was written by Jack Duryea (duryea@bcm.edu), BCM 2017, Waterland Lab. This software is a custom methylation extractor that parses the methylation calls in a SAM file and built"
des += "for the purpose of analyzing read-specific methylation in bis-seq data split by chromosome. "
des += "For each read in the file, the program find the number of CpGs in the read that overlap a 200bp bin. If that number meets a certain filtering criterion, the program finds the average methylation of those CpGs and add this value to the bin. "
des += "Once the read-average methylation values have been computed, mixture model analysis can be run on these values for each bin using GMM and optimized with EM. "
des += "The output is a csv file. The first column is the bin name, the second column is the list of average read methylation values"

parser = argparse.ArgumentParser(description=des)
parser.add_argument("-c", required=True, help="The name of the chromosome being analyzed, e.g. 'chrY' ")
parser.add_argument("-a", required=False, help="The path of the annotation file for the chromosome being analyzed. Needs to include 200bp bin names and CpG counts per bin. Column names should be 'Bin Name' and 'CpG Count' respectively. e.g. '/home/chrYannotation.csv' ")
parser.add_argument("-s", required=True, help="The SAM file to be analyzed, must be mapped with Bismark")
parser.add_argument("-o", required=True, help="The name of the output file")
parser.add_argument("-b", required=False, default=100, help="The bin size, default is 100")
parser.add_argument("-sr", required=False, default=0, help="Set to 1 if using super reads, keep as is if using a normal SAM file")

args = parser.parse_args()

chromosome_name = args.c
annotation_file = args.a
sam_file = args.s
using_super_reads = (int(args.sr) == 1)
output_file = args.o
bin_size = int(args.b)



# Program entry
if __name__ == "__main__":

	# Get data from sam file
	reads = get_reads(sam_file)
	bin_data = []
	for read in reads:
		results = compute_read_methylation(read)
		bin_data.append(results)   	


	read_methylation_df = update_data_frame(bin_data)

	# Clean the data and append annotation data
	read_methylation_df = clean_data(read_methylation_df, chromosome_name)

	# Filter by coverage
	bins = []
	for cpg_data in read_methylation_df["Read Contributions"]:
		mybin = make_cpg_matrix(cpg_data)
		bins.append(mybin)



	# # save pickel data
	with open(output_file,'wb') as fp:
		cPickle.dump(bins,fp)

	


	print "Analysis complete"


# python axtell.py -c chr19 -s debug.txt -o debugout.csv











