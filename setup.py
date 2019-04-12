from setuptools import setup

setup(name="PReLIM",
      version="0.2",
      description="",
      author="Jack Duryea",
      author_email="",
      license="MIT",
      packages=['prelim'],
      install_requires=['scikit-learn', 'numpy', 'pandas', 'tqdm'],
      include_package_data=True,
)