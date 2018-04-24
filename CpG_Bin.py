"""
Author: 
Jack Duryea
Waterland Lab
Baylor College of Medicine

A Python object representing information about a 
bin and its meta data. Very useful for storing information and working with it.
Can be saved as a pickle file
"""


class Bin():
	""" 
	Constructor for a bin

	matrix: numpy array, the bin's CpG matrix.
	binStartInc: integer, the starting, inclusive, chromosomal index of the bin.
	binEndInc: integer, the ending, inclusive, chromosomal index of the bin.
	cpgPositions: array of integers, the chromosomal positions of the CpGs in the bin.
	encoding: array, a reduced representation of the bin's CpG matrix
	missingToken: integer, the token that represents missing data in the matrix.
	chromosome: string, the chromosome this bin resides in.
	binSize: integer, the number of base pairs this bin covers
	species: string, the speices this bin belongs too.
	tag: string, for custom use.




	"""
	def __init__(self, matrix, binStartInc, binEndInc, cpgPositions, encoding=None, missingToken= -1, chromosome="19", binSize=100, species="HG38", tag="customDescriptor"):
		self.cpgDensity = matrix.shape[0]

		assert binSize > 0, "invalid bin size"
		assert binStartInc < binEndInc, "invalid start and end indices"
		assert binEndInc - binStartInc == binSize - 1
		# TODO: add more assertions

		assert len(cpgPositions) == self.cpgDensity, "wrong number of positions"

		if ! (species == "HG38") or ! (species = "MM10"):
			print "Warning, you are not supplying a common species type. You've been warned"	

		self.matrix = matrix
		self.binStartInc = binStartInc
		self.binEndInc = binEndInc
		self.cpgPositions = cpgPositions
		self.missingToken = missingToken
		self.chromosome = chromosome
		self.binSize = binSize
		self.species = species
		self.tag = tag


	






