from setuptools import setup

setup(name="prelim",
      version="0.1.12",
      description="Package to impute missing CpG methylation data", 
      author="Jack Duryea",
      author_email="jack.duryea@bcm.edu",
      license='MIT',
      packages=['src'],

      install_requires=[
          'pysam', 
          'numpy', 
          'matplotlib>3,<3.1', 
          'scikit-learn==0.21.2',
          'joblib',
          'seaborn', 
          'scipy', 
          'pandas',
          'pickle',
          'random', 
          'fastcluster', 
          'pebble', 
          'tqdm'],

     

)