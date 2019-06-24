
===============
Using PReLIM
===============

Getting Started
================

.. code-block:: python

	from PReLIM import PReLIM
	import numpy as np

	# Note: This toy example does not contain enough data for the model
	# to actually make imputations due to train/validation splits. More bins must be used. 

	# Step 1: Collect methylation matrices, 1 is methylated, 0 is unmethylated, -1 is unknown. Each column is a cpg site, each row is a read
	bin1 = np.array([[1,0],[0,-1],[-1,1],[0,0]],dtype=float)
	bin2 = np.array([[1,0],[1,0],[-1,1],[0,0],[0,1],[1,1],[0,0]],dtype=float)
	bin3 = np.array([[-1,1],[0,-1],[-1,1],[0,0]],dtype=float)
	
	# Put bins into a list
	bins = [bin1, bin2, bin3]

	# Step 2: Created a model with correnct density for given bins
	model = PReLIM(cpgDensity=2)

	# Step 3: Train the model and save it if you wish
	model.train(bins, model_file="my_model_file")

	# Step 4: Use model for imputating a single bin, or...
	imputed_bin1 = model.impute(bin1)

	# ... use batch imputation to impute many bins at once (recommended)
	imputed_bins = model.impute_many(bins)