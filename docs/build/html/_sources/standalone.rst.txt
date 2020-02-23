===========================
Standalone Imputation Tool
===========================

Requirements
=============

* Python 3.5+ (could maybe get by on 3.0+ but don't risk it)
* Any \*nix operating system
* All the stuff in the requirements file
* ClubCpG (just for the BAM file parsing)

Usage
=============

PReLIM has a standlone tool that can be used to convert BAM files directly
into DSS format without using ClubCpG. It is easy to use and can train new models
or use existing ones. Note: It is highly recommended to do this on a chromosome-by-chromosome
basis!

1. BAM to DSS without imputation

	python bam2dss.py -i input.bam, -c chr19 -o output.txt

2. BAM to DSS with imputation using existing models

	python bam2dss.py -i input.bam, -c chr19 -o output.txt -p -m model_prefix

3. BAM to DSS with imputation using new models (you'll probably be using this one)

	python bam2dss.py -i input.bam, -c chr19 -o output.txt -p


